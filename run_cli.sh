#!/bin/bash
#
# Wrapper script for running the CLI tool with the correct PostgreSQL
# client library path. This is mainly needed on macOS (especially on
# Apple Silicon) where psycopg2 may fail to locate libpq unless the
# Homebrew PostgreSQL path is included in DYLD_LIBRARY_PATH.
#
# Usage:
#   ./run_cli.sh [cli arguments...]
#
# All arguments passed to this wrapper are forwarded directly to cli.py.
#
# Why this exists:
#   - Avoids modifying shell dotfiles just to make psycopg2 work.
#   - Ensures consistent behavior across developers’ systems.
#   - Only affects the environment for this invocation.
#

# Add Homebrew's libpq directory so psycopg2 can find libpq.dylib
export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libpq/lib:${DYLD_LIBRARY_PATH}"

# Execute the CLI script with all forwarded arguments
python cli.py "$@"
