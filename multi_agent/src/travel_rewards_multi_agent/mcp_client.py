from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from crewai.tools import BaseTool
from crewai_tools.adapters.mcp_adapter import MCPServerAdapter
from mcp.client.stdio import StdioServerParameters

from .agents import APPROVED_MCP_TOOLS


def default_mcp_server_path() -> Path:
    """Resolve the shared Node MCP server independently of the caller's cwd."""
    return Path(__file__).resolve().parents[3] / "src" / "mcp-server.mjs"


@contextmanager
def open_mcp_tools(node_executable: str, server_path: str | Path) -> Iterator[list[BaseTool]]:
    """Start the local Node MCP server and yield only its approved read tools."""
    path = Path(server_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Local MCP server entry point does not exist: {path}")
    if not node_executable.strip():
        raise ValueError("A Node.js executable is required to start the MCP server")

    parameters = StdioServerParameters(
        command=node_executable,
        args=[str(path)],
        cwd=str(path.parent.parent),
    )
    with MCPServerAdapter(parameters) as tools:
        names = {tool.name for tool in tools}
        if names != APPROVED_MCP_TOOLS:
            raise RuntimeError(
                "Local MCP server exposed an unexpected tool set: "
                f"expected {sorted(APPROVED_MCP_TOOLS)}, received {sorted(names)}"
            )
        yield tools
