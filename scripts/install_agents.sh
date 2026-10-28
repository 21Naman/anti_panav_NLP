#!/usr/bin/env bash
# Copy the improvement-loop agents + /improve-loop command into .claude/ (Claude Code reads them there).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .claude/agents .claude/commands
cp agents/agents/*.md .claude/agents/
cp agents/commands/*.md .claude/commands/
echo "installed: $(ls agents/agents | tr '\n' ' ')+ /improve-loop"
