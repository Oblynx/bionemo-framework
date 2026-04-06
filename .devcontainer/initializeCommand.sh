#!/bin/bash
# Create the mounted config directories if they don't already exist

mkdir -p ~/.devcontainer_cache
mkdir -p ~/.ssh
mkdir -p ~/.cache/pre-commit-devcontainer
mkdir -p ~/.gnupg
mkdir -p ~/.config
mkdir -p ~/.cursor
# ~/.claude is built inside the container by fix-claude-paths.sh from ~/.claude_host
[ ! -f ~/.netrc ] && touch ~/.netrc
[ ! -f ~/.bash_history_devcontainer ] && touch ~/.bash_history_devcontainer

exit 0
