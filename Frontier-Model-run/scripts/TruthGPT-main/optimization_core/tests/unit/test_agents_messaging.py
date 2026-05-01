"""
Unit tests for the agents.messaging module.

Tests that all adapters are importable, instantiate correctly with mock
clients, and have the correct method signatures.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock


# ─── Import validation ────────────────────────────────────────────────────────

class TestMessagingImports:
    """Verify that every adapter can be imported from the unified __init__."""

    def test_import_base(self):
        from optimization_core.agents.messaging import BaseMessagingAdapter
        assert BaseMessagingAdapter is not None

    def test_import_telegram(self):
        from optimization_core.agents.messaging import TelegramAdapter
        assert TelegramAdapter is not None

    def test_import_whatsapp(self):
        from optimization_core.agents.messaging import WhatsAppAdapter
        assert WhatsAppAdapter is not None

    def test_import_discord(self):
        from optimization_core.agents.messaging import DiscordAdapter
        assert DiscordAdapter is not None

    def test_import_signal(self):
        from optimization_core.agents.messaging import SignalAdapter
        assert SignalAdapter is not None

    def test_import_slack(self):
        from optimization_core.agents.messaging import SlackAdapter
        assert SlackAdapter is not None

    def test_import_teams(self):
        from optimization_core.agents.messaging import TeamsAdapter
        assert TeamsAdapter is not None

    def test_import_email(self):
        from optimization_core.agents.messaging import EmailAdapter
        assert EmailAdapter is not None

    def test_import_router_factory(self):
        from optimization_core.agents.messaging import create_messaging_router
        assert callable(create_messaging_router)


# ─── Instantiation ────────────────────────────────────────────────────────────

class TestAdapterInstantiation:
    """Ensure adapters can be constructed with a mock agent client."""

    @pytest.fixture
    def mock_client(self):
        client = MagicMock()
        client.run = AsyncMock(return_value=MagicMock(
            content="ok", action_type="final_answer",
        ))
        return client

    def test_telegram_init(self, mock_client):
        from optimization_core.agents.messaging import TelegramAdapter
        adapter = TelegramAdapter(mock_client, bot_token="test-token")
        assert adapter.agent_client is mock_client

    def test_discord_init(self, mock_client):
        from optimization_core.agents.messaging import DiscordAdapter
        adapter = DiscordAdapter(mock_client, bot_token="test-token")
        assert adapter.agent_client is mock_client

    def test_signal_init(self, mock_client):
        from optimization_core.agents.messaging import SignalAdapter
        adapter = SignalAdapter(mock_client)
        assert adapter.agent_client is mock_client

    def test_slack_init(self, mock_client):
        from optimization_core.agents.messaging import SlackAdapter
        adapter = SlackAdapter(mock_client)
        assert adapter.agent_client is mock_client

    def test_teams_init(self, mock_client):
        from optimization_core.agents.messaging import TeamsAdapter
        adapter = TeamsAdapter(mock_client)
        assert adapter.agent_client is mock_client
        # Verify _token_expires_at is initialised
        assert hasattr(adapter, "_token_expires_at")
        assert adapter._token_expires_at == 0.0

    def test_email_init(self, mock_client):
        from optimization_core.agents.messaging import EmailAdapter
        adapter = EmailAdapter(mock_client)
        assert adapter.agent_client is mock_client

    def test_whatsapp_init(self, mock_client):
        from optimization_core.agents.messaging import WhatsAppAdapter
        adapter = WhatsAppAdapter(mock_client)
        assert adapter.agent_client is mock_client


# ─── Method signatures ─────────────────────────────────────────────────────────

class TestAdapterSignatures:
    """Verify that all concrete adapters implement the abstract interface."""

    ADAPTER_CLASSES = [
        "TelegramAdapter",
        "WhatsAppAdapter",
        "DiscordAdapter",
        "SignalAdapter",
        "SlackAdapter",
        "TeamsAdapter",
        "EmailAdapter",
    ]

    @pytest.mark.parametrize("cls_name", ADAPTER_CLASSES)
    def test_has_on_message(self, cls_name):
        import optimization_core.agents.messaging as msg
        cls = getattr(msg, cls_name)
        assert hasattr(cls, "on_message"), f"{cls_name} missing on_message"

    @pytest.mark.parametrize("cls_name", ADAPTER_CLASSES)
    def test_has_send_response(self, cls_name):
        import optimization_core.agents.messaging as msg
        cls = getattr(msg, cls_name)
        assert hasattr(cls, "send_response"), f"{cls_name} missing send_response"

    @pytest.mark.parametrize("cls_name", ADAPTER_CLASSES)
    def test_has_handle(self, cls_name):
        import optimization_core.agents.messaging as msg
        cls = getattr(msg, cls_name)
        assert hasattr(cls, "handle"), f"{cls_name} missing handle"


# ─── Models used by adapters ───────────────────────────────────────────────────

class TestAgentModels:
    """Verify Pydantic models are importable and valid."""

    def test_agent_response_creation(self):
        from optimization_core.agents.models import AgentResponse
        response = AgentResponse(
            content="Hello",
            action_type="final_answer",
        )
        assert response.content == "Hello"
        assert response.action_type == "final_answer"
        assert response.metadata == {}
        assert response.tool_calls == []

    def test_agent_action_creation(self):
        from optimization_core.agents.models import AgentAction
        action = AgentAction(
            thought="thinking...",
            tool="web_search",
            tool_input="query",
        )
        assert action.thought == "thinking..."
        assert action.tool == "web_search"

    def test_inference_result_creation(self):
        from optimization_core.agents.models import InferenceResult
        result = InferenceResult(text="generated text")
        assert result.text == "generated text"
        assert result.tokens_generated is None

    def test_agent_config_creation(self):
        from optimization_core.agents.models import AgentConfig
        config = AgentConfig(
            memory_db_path="test.db",
            use_swarm=True,
        )
        assert config.memory_db_path == "test.db"
        assert config.use_swarm is True
