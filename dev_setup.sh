#!/bin/bash
# Creates a virtual environment for CODEX

#############
# Utilities #
#############

YELLOW="\033[1;33m"
NC="\033[0m"

# Terminate immediately if the current command fails for any reason.
function check() {
    echo -e "${YELLOW}$@${NC}"
    if ! $@ ; then
        echo "Command failed: \"$@\""
        exit 1
    fi
}

ROOT_PATH="$(dirname $(realpath $0))"
PIP_NO_CACHE_DIR=1

check cd "${ROOT_PATH}"
check python3.8 -m venv .venv
check source .venv/bin/activate
check pip install --upgrade pip
check pip install -r requirements.txt
check pre-commit install
check deactivate
