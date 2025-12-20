"""
Pytest configuration for PointillSim tests.

Run all tests:
    pytest tests/ -v

Run with visual output generation:
    pytest tests/ -v --visual

Run only visual tests:
    pytest tests/ -v --visual -m visual
"""

import pytest


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--visual",
        action="store_true",
        default=False,
        help="Run visual output tests that generate plots"
    )


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "visual: mark test as generating visual output for user verification"
    )


def pytest_collection_modifyitems(config, items):
    """Skip visual tests unless --visual flag is provided."""
    if config.getoption("--visual"):
        # Run all tests including visual
        return

    skip_visual = pytest.mark.skip(reason="need --visual option to run visual tests")
    for item in items:
        if "visual" in item.keywords:
            item.add_marker(skip_visual)
