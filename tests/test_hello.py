"""Tests for the hello module."""

import pytest

from njit_wrappers.hello import greet


def test_greet_default() -> None:
    assert greet() == "Hello, World!"


def test_greet_with_name() -> None:
    assert greet("Alice") == "Hello, Alice!"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Bob", "Hello, Bob!"),
        ("Python", "Hello, Python!"),
        ("", "Hello, !"),
    ],
)
def test_greet_parametrize(name: str, expected: str) -> None:
    assert greet(name) == expected
