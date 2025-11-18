#!/bin/bash
# Wrapper script to run CLI with proper library paths for psycopg2

export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libpq/lib:$DYLD_LIBRARY_PATH"
python cli.py "$@"



