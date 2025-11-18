#!/bin/bash
# Wrapper script to run setup_database.py with proper library paths for psycopg2

export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libpq/lib:$DYLD_LIBRARY_PATH"
python setup_database.py "$@"

