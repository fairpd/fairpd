#!/usr/bin/env bash
# Fetch FSW's released code (Park, Kim and Whang, KDD 2026) at the commit this
# work used, into third_party/FSW, and apply the one fix it needs to import
# (two blocks of algorithms/sensitive.py lost a level of indentation).
# The code carries no licence, so it is fetched rather than redistributed.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -d third_party/FSW ]]; then
  echo "third_party/FSW exists; remove it to fetch again" >&2
  exit 0
fi
git clone --quiet https://github.com/jyparkkr/FSW.git third_party/FSW
git -C third_party/FSW checkout --quiet 6b19d10d1ac0ccc33c3d4ab16d4d070ac9de5edf
git -C third_party/FSW apply ../../patches/fsw-sensitive-indentation.patch
echo "FSW at 6b19d10 in third_party/FSW, patched"
