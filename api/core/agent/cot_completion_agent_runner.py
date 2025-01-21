import json
from collections.abc import Sequence
from typing import Optional

from core.agent.cot_agent_runner import CotAgentRunner
from core.model_runtime.entities.message_entities import (
    AssistantPromptMessage,
    PromptMessage,
    TextPromptMessageContent,
    ToolPromptMessage,
    UserPromptMessage,
)
from core.model_runtime.utils.encoders import jsonable_encoder


class CotCompletionAgentRunner(CotAgentRunner):
    def _organize_instruction_prompt(self) -> str:
        """
        Organize instruction prompt
        """
        if self.app_config.agent is None:
            raise ValueError("Agent configuration is not set")
        prompt_entity = self.app_config.agent.prompt
        if prompt_entity is None:
            raise ValueError("prompt entity is not set")
        first_prompt = prompt_entity.first_prompt

        system_prompt = (
            first_prompt.replace("{{instruction}}", self._instruction)
            .replace("{{tools}}", json.dumps(jsonable_encoder(self._prompt_messages_tools)))
            .replace("{{tool_names}}", ", ".join([tool.name for tool in self._prompt_messages_tools]))
        )

        return system_prompt

    def _organize_historic_prompt(self, current_session_messages: Optional[list[PromptMessage]] = None) -> str:
        """
        Organize historic prompt
        """
        historic_prompt_messages = self.history_prompt_messages or []
        historic_prompt = ""

        for message in historic_prompt_messages:
            if isinstance(message, UserPromptMessage):
                if message.content is None:
                    continue
                if isinstance(message.content, str):
                    historic_prompt += f"Question: {message.content}\n\n"
                elif isinstance(message.content, Sequence):
                    content_text = " ".join([
                        content.data for content in message.content 
                        if isinstance(content, TextPromptMessageContent)
                    ])
                    historic_prompt += f"Question: {content_text}\n\n"
            elif isinstance(message, AssistantPromptMessage):
                if message.content is None:
                    continue
                if isinstance(message.content, str):
                    historic_prompt += message.content + "\n\n"
                elif isinstance(message.content, Sequence):
                    content_text = " ".join([
                        content.data for content in message.content 
                        if isinstance(content, TextPromptMessageContent)
                    ])
                    historic_prompt += content_text + "\n\n"
                elif isinstance(message.content, TextPromptMessageContent):
                    historic_prompt += message.content.data + "\n\n"
            elif isinstance(message, ToolPromptMessage):
                if message.content is None:
                    continue
                if isinstance(message.content, str):
                    historic_prompt += f"Tool Response: {message.content}\n\n"
                elif isinstance(message.content, TextPromptMessageContent):
                    historic_prompt += f"Tool Response: {message.content.data}\n\n"

        return historic_prompt

    def _organize_prompt_messages(self) -> list[PromptMessage]:
        """
        Organize prompt messages
        """
        # organize system prompt
        system_prompt = self._organize_instruction_prompt()

        # organize historic prompt messages
        historic_prompt = self._organize_historic_prompt()

        # organize current assistant messages
        agent_scratchpad = self._agent_scratchpad
        assistant_prompt = ""
        for unit in agent_scratchpad or []:
            if unit.is_final():
                assistant_prompt += f"Final Answer: {unit.agent_response}"
            else:
                assistant_prompt += f"Thought: {unit.thought}\n\n"
                if unit.action_str:
                    assistant_prompt += f"Action: {unit.action_str}\n\n"
                if unit.observation:
                    assistant_prompt += f"Observation: {unit.observation}\n\n"

        # query messages
        query_prompt = f"Question: {self._query}"

        # join all messages
        prompt = (
            system_prompt.replace("{{historic_messages}}", historic_prompt)
            .replace("{{agent_scratchpad}}", assistant_prompt)
            .replace("{{query}}", query_prompt)
        )

        return [UserPromptMessage(content=[TextPromptMessageContent(data=prompt)])]
