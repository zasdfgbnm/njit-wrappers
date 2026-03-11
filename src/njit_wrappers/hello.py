"""Hello world module."""


def greet(name: str = "World") -> str:
    """Return a greeting string.

    Args:
        name: The name to greet.

    Returns:
        A greeting string.
    """
    return f"Hello, {name}!"
