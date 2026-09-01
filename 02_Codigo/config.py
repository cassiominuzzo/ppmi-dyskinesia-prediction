"""
Single place where the data root is configured.

PPMI is controlled-access and is not redistributed here. Point PROJECT_ROOT at a
local directory holding the raw PPMI tables, laid out as described in README.md.

    export LID_PROJECT_ROOT=/path/to/your/data
"""
import os

PROJECT_ROOT = os.environ.get("LID_PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))

if not os.path.isdir(PROJECT_ROOT):
    raise SystemExit(
        "Set LID_PROJECT_ROOT to the directory containing the raw PPMI tables. "
        "See README.md for the expected layout."
    )
