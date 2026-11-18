import json
from importlib import import_module

import pytest


def load_mcp_client():
    try:
        return import_module("travel_rewards_multi_agent.mcp_client")
    except ModuleNotFoundError:
        pytest.fail("CrewAI MCP stdio connection is not implemented", pytrace=False)


def test_stdio_mcp_adapter_exposes_only_read_tools_and_reads_fixture_values():
    mcp_client = load_mcp_client()
    server_path = mcp_client.default_mcp_server_path()

    with mcp_client.open_mcp_tools("node", server_path) as tools:
        assert {tool.name for tool in tools} == {
            "get_household_snapshot",
            "search_benefit_evidence",
            "calculate_remaining_benefit",
        }

        household = json.loads(tools["get_household_snapshot"].run())
        remaining = json.loads(tools["calculate_remaining_benefit"].run())

    assert household == {
        "product": "Chase Sapphire Reserve",
        "issuer": "Chase",
        "annualFee": 795,
        "benefitName": "annual travel credit",
        "benefitAmount": 300,
        "usedAmount": 180,
        "anniversaryDate": "08-15",
    }
    assert remaining["remainingAmount"] == 120
    assert remaining["currency"] == "USD"
