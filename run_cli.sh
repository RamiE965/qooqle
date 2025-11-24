#!/bin/bash
#
# Wrapper script for running setup_database.py with the correct PostgreSQL
# client library path. This is primarily needed on macOS (Apple Silicon /
# Homebrew installs) where psycopg2 may fail to load libpq unless DYLD_LIBRARY_PATH
# includes Homebrew’s PostgreSQL location.
#
# Usage:
#   ./run_setup.sh [args...]
#
# Any arguments you pass to this script are forwarded directly to
# setup_database.py.
#
# Notes:
#   - This avoids modifying global shell rc files just to run one script.
#   - DYLD_LIBRARY_PATH is prepended to preserve any existing user-defined value.
#   - On Linux systems this variable is typically unnecessary.
#

# Prepend Homebrew’s libpq path so psycopg2 can find libpq.dylib
export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libpq/lib:${DYLD_LIBRARY_PATH}"

# Delegate execution to the actual setup script
python setup_database.py "$@"
