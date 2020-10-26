#!/bin/bash

# Script is taken from https://circleci.com/developer/orbs/orb/roopakv/swissknife#commands-run_if_modified
# Usage: sh trigger_if_modified.sh <pattern>
# - for example: sh trigger_if_modified.sh "^(ignite|tests|examples|\.circleci).*"

if [ -z "$1" ]; then
    echo "Pattern should be provided. Usage: sh trigger_if_modified.sh <pattern>"
    exit 1
fi

base_branch=master
pattern=$1

if [ -z "$BASH" ]; then
  echo Bash not installed.
  exit 1
fi

git status >/dev/null 2>&1 || { echo >&2 "Not in a git directory or no git"; exit 1; }

circleci-agent >/dev/null 2>&1 || { echo >&2 "No Circle CI agent. These are in all Circle CI containers"; exit 1; }

FILES_MODIFIED=""

setcommit () {
  FILES_MODIFIED=$(git diff --name-only origin/${base_branch}..HEAD | grep -i -E ${pattern})
}

setcommit || true

if [ -z "$FILES_MODIFIED" ]; then
  echo "Files not modified. Halting job"
  circleci-agent step halt
else
  echo "Files modified, continuing steps"
fi