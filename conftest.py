from __future__ import annotations


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run tests marked as slow (e.g. cluster convergence).",
    )
