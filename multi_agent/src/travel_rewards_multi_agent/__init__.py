"""Local multi-agent travel rewards capstone demo."""

from . import privacy as _privacy

# Must run before CrewAI, LangGraph, or LangChain are imported by any submodule.
_privacy.configure()

__version__ = "0.1.0"
