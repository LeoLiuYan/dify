"""Unit tests for the CotCompletionAgentRunner class.

This module contains unit tests for the CotCompletionAgentRunner class, which is responsible
for organizing and managing prompts in a Chain of Thought completion agent context.

The tests cover three main methods:
1. _organize_instruction_prompt
2. _organize_historic_prompt
3. _organize_prompt_messages

Each test ensures proper handling of:
- Prompt organization and formatting
- Error cases and edge conditions
- Integration with parent class functionality
"""

import json
from collections.abc import Sequence
from typing import Optional, cast
from unittest.mock import MagicMock, patch

import pytest

from core.model_runtime.entities.message_entities import PromptMessageContent

from core.agent.cot_completion_agent_runner import CotCompletionAgentRunner
from core.agent.entities import AgentEntity, AgentScratchpadUnit
from core.app.apps.agent_chat.app_config_manager import AgentChatAppConfig
from core.app.entities.app_invoke_entities import ModelConfigWithCredentialsEntity
from core.model_runtime.entities.message_entities import (
    AssistantPromptMessage,
    PromptMessage,
    PromptMessageTool,
    SystemPromptMessage,
    TextPromptMessageContent,
    ToolPromptMessage,
    UserPromptMessage,
)
from core.model_runtime.entities.model_entities import ModelFeature
from core.model_runtime.utils.encoders import jsonable_encoder
from core.tools.entities.tool_entities import ToolParameter, ToolRuntimeVariablePool
from models.model import Message, MessageAgentThought


@pytest.fixture
def mock_app_config():
    """Fixture providing a mocked AgentChatAppConfig with necessary attributes."""
    config = MagicMock(spec=AgentChatAppConfig)
    
    # Required attributes for BaseAgentRunner
    config.app_id = "test_app_id"
    config.dataset = MagicMock()
    config.dataset.dataset_ids = []
    config.dataset.retrieve_config = None
    config.additional_features = MagicMock()
    config.additional_features.show_retrieve_source = False
    
    # Required attributes for CotCompletionAgentRunner
    config.agent = MagicMock()
    config.agent.prompt = MagicMock()
    config.agent.prompt.first_prompt = (
        "System: You are a helpful AI assistant.\n\n"
        "Instructions: {{instruction}}\n\n"
        "Available tools: {{tools}}\n\n"
        "Tool names: {{tool_names}}\n\n"
        "Chat history:\n{{historic_messages}}\n\n"
        "Current conversation:\n{{agent_scratchpad}}\n\n"
        "{{query}}"
    )
    return config


@pytest.fixture
def mock_model_config():
    """Fixture providing a mocked ModelConfigWithCredentialsEntity."""
    config = MagicMock(spec=ModelConfigWithCredentialsEntity)
    config.provider = "test_provider"
    config.model = "test_model"
    config.parameters = {}
    return config


@pytest.fixture
def mock_prompt_tools():
    """Fixture providing a list of mock PromptMessageTool instances."""
    tool1 = PromptMessageTool(
        name="calculator",
        description="A calculator tool for basic math operations",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The math expression to evaluate"
                }
            },
            "required": ["expression"]
        }
    )
    tool2 = PromptMessageTool(
        name="weather",
        description="A tool to get weather information",
        parameters={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The location to get weather for"
                }
            },
            "required": ["location"]
        }
    )
    return [tool1, tool2]


@pytest.fixture
def mock_db_session():
    """Mock database session for testing."""
    with patch('extensions.ext_database.db.session') as mock_session:
        # Mock the query chain for Message
        mock_message_query = MagicMock()
        mock_message_query.filter.return_value = mock_message_query
        mock_message_query.order_by.return_value = mock_message_query
        mock_message_query.all.return_value = []
        
        # Mock the query chain for MessageAgentThought
        mock_thought_query = MagicMock()
        mock_thought_query.filter.return_value = mock_thought_query
        mock_thought_query.count.return_value = 0
        
        def get_mock_query(model):
            if model == Message:
                return mock_message_query
            elif model == MessageAgentThought:
                return mock_thought_query
            return MagicMock()
        
        mock_session.query.side_effect = get_mock_query
        yield mock_session

@pytest.fixture
def base_runner(mock_app_config, mock_model_config, mock_prompt_tools, mock_db_session):
    """Fixture providing a basic configured CotCompletionAgentRunner instance."""
    with patch('core.agent.base_agent_runner.db.session', mock_db_session):
        runner = CotCompletionAgentRunner(
            tenant_id="test_tenant",
            application_generate_entity=MagicMock(
        invoke_from="test",
        files=[],
        model_conf=MagicMock(stop=[], provider="test_provider")
    ),
            conversation=MagicMock(),
            app_config=mock_app_config,
            model_config=mock_model_config,
            config=MagicMock(spec=AgentEntity),
            queue_manager=MagicMock(),
            message=MagicMock(spec=Message),
            user_id="test_user",
            memory=None,
            prompt_messages=[],
            variables_pool=None,
            db_variables=None,
            model_instance=MagicMock()
        )
    runner._prompt_messages_tools = mock_prompt_tools
    runner._instruction = "Help the user with their questions"
    runner._query = "What is 2+2?"
    runner._agent_scratchpad = []
    return runner


def test_organize_instruction_prompt_success(base_runner):
    """Test successful instruction prompt organization with all components."""
    result = base_runner._organize_instruction_prompt()
    
    # Verify instruction replacement
    assert "Help the user with their questions" in result
    
    # Verify tools json replacement
    tools_json = json.dumps(jsonable_encoder(base_runner._prompt_messages_tools))
    assert tools_json in result
    
    # Verify tool names replacement
    tool_names = ", ".join([tool.name for tool in base_runner._prompt_messages_tools])
    assert "calculator, weather" in result


def test_organize_instruction_prompt_no_agent_config(base_runner):
    """Test instruction prompt organization raises error when agent config is missing."""
    base_runner.app_config.agent = None
    with pytest.raises(ValueError, match="Agent configuration is not set"):
        base_runner._organize_instruction_prompt()


def test_organize_instruction_prompt_no_prompt_entity(base_runner):
    """Test instruction prompt organization raises error when prompt entity is missing."""
    base_runner.app_config.agent.prompt = None
    with pytest.raises(ValueError, match="prompt entity is not set"):
        base_runner._organize_instruction_prompt()


def test_organize_historic_prompt_empty(base_runner):
    """Test historic prompt organization with no messages."""
    result = base_runner._organize_historic_prompt()
    assert result == ""


def test_organize_historic_prompt_with_messages(base_runner):
    """Test historic prompt organization with various message types."""
    # Create sample historic messages
    # Create messages by instantiating and then setting attributes
    user_msg = UserPromptMessage(content="What's the weather?")
    
    assistant_msg1 = AssistantPromptMessage(content="Let me check that for you.")
    
    tool_msg = ToolPromptMessage(
        content="The weather is sunny.",
        name="weather",
        tool_call_id="123"
    )
    
    assistant_msg2 = AssistantPromptMessage(content="It's sunny today!")
    
    messages = [user_msg, assistant_msg1, tool_msg, assistant_msg2]
    
    # Set up the historic messages
    base_runner.history_prompt_messages = messages
    
    result = base_runner._organize_historic_prompt()
    
    # Verify message formatting
    assert "Question: What's the weather?" in result
    assert "Let me check that for you." in result
    assert "Tool Response: The weather is sunny." in result
    assert "It's sunny today!" in result


def test_organize_historic_prompt_with_non_text_content(base_runner):
    """Test historic prompt organization with non-text content types."""
    # Create a message with a list of content types
    user_msg = UserPromptMessage(content=[
        TextPromptMessageContent(data="What's the weather like?"),
        TextPromptMessageContent(data="And temperature too?")
    ])
    
    assistant_msg = AssistantPromptMessage(content="Let me check both for you.")
    
    base_runner.history_prompt_messages = [user_msg, assistant_msg]
    
    result = base_runner._organize_historic_prompt()
    assert "Question: What's the weather like? And temperature too?" in result
    assert "Let me check both for you." in result


def test_organize_prompt_messages_complete(base_runner):
    """Test full prompt message organization with all components."""
    # Set up historic messages
    user_msg = UserPromptMessage(content="What is 2+2?")
    base_runner._historic_prompt_messages = [user_msg]
    
    # Set up agent scratchpad
    base_runner._agent_scratchpad = [
        AgentScratchpadUnit(
            agent_response="Let me calculate that",
            thought="I should use the calculator",
            action_str='{"action": "calculator", "action_input": {"expression": "2+2"}}',
            action=AgentScratchpadUnit.Action(
                action_name="calculator",
                action_input={"expression": "2+2"}
            ),
            observation="4"
        )
    ]
    
    result = base_runner._organize_prompt_messages()
    
    # Verify message structure
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], UserPromptMessage)
    assert isinstance(result[0].content, list)
    assert len(result[0].content) == 1
    assert isinstance(result[0].content[0], TextPromptMessageContent)
    content = result[0].content[0].data
    assert isinstance(content, str)
    
    # Verify all components are present
    expected_components = [
        "System: You are a helpful AI assistant",
        "Help the user with their questions",
        "What is 2+2?",
        "Thought: I should use the calculator",
        "Action:",
        "Observation: 4"
    ]
    for component in expected_components:
        assert component in content


def test_organize_prompt_messages_minimal(base_runner):
    """Test prompt message organization with minimal components."""
    # Clear optional components
    base_runner._agent_scratchpad = []
    base_runner._historic_prompt_messages = []
    
    result = base_runner._organize_prompt_messages()
    
    # Verify message structure
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], UserPromptMessage)
    assert isinstance(result[0].content, list)
    assert len(result[0].content) == 1
    assert isinstance(result[0].content[0], TextPromptMessageContent)
    content = result[0].content[0].data
    assert isinstance(content, str)
    
    # Verify essential components
    expected_components = [
        "System: You are a helpful AI assistant",
        "Help the user with their questions",
        "What is 2+2?"
    ]
    for component in expected_components:
        assert component in content
    
    # Verify optional components are not present
    assert "Thought:" not in content
    assert "Action:" not in content
    assert "Observation:" not in content
