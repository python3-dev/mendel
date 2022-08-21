"""Controller model."""

from typing import Any

from models.genetic import Population


class Controller:
    """Controller model."""

    def __init__(self, input: dict[str, Any]) -> None:
        """Initialise Controller model."""
        self.input: dict[str, Any] = input
