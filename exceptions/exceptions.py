"""Custom Error definitions."""


class ZeroLengthBit(Exception):
    """ZeroLengthBit exception."""

    def __init__(self, *args: object) -> None:
        """Initialise ZeroLengthBit exception."""
        super().__init__(
            "Zero length bits are not allowed. Possibly insufficient bit_array length."
        )


class InsufficientBitArrayLength(Exception):
    """InsufficientBitArrayLength exception."""

    def __init__(self, *args: object) -> None:
        """Initialise InsufficientBitArrayLength exception."""
        super().__init__("self.bit_array length insufficient to create genes.")
