#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Fetch the original Apache-2.0 Ravens implementation without vendoring it.
set -euo pipefail
cd "$(dirname "$0")/.."
COMMIT=4752ffab487ebd387781ad7890aa540ae2780970
REPOSITORY=https://github.com/google-research/ravens.git
if [ -e upstream ]; then
  test "$(git -C upstream rev-parse HEAD)" = "$COMMIT"
  test -z "$(git -C upstream status --porcelain)"
  echo "Existing clean upstream checkout verified: $COMMIT"
  exit 0
fi
# A failed fetch keeps a diagnostic directory; no existing files are deleted.
STAGING="$(mktemp -d .upstream-fetch-XXXXXX)"
git -C "$STAGING" init --quiet
git -C "$STAGING" remote add origin "$REPOSITORY"
if [ -n "${RAVENS_SOURCE_BUNDLE:-}" ]; then
  case "$RAVENS_SOURCE_BUNDLE" in
    /*) ;;
    *) echo 'RAVENS_SOURCE_BUNDLE must be an absolute path.' >&2; exit 1 ;;
  esac
  test -f "$RAVENS_SOURCE_BUNDLE"
  # The bundle must have been obtained from the official repository on a
  # trusted machine. Verification in the empty staging repository also rejects
  # incremental bundles with unavailable prerequisites. No TLS checks are disabled.
  git -C "$STAGING" bundle verify "$RAVENS_SOURCE_BUNDLE"
  git -C "$STAGING" fetch "$RAVENS_SOURCE_BUNDLE" "$COMMIT"
else
  git -C "$STAGING" fetch --depth=1 origin "$COMMIT"
fi
git -C "$STAGING" checkout --quiet --detach FETCH_HEAD
test "$(git -C "$STAGING" rev-parse HEAD)" = "$COMMIT"
test -z "$(git -C "$STAGING" status --porcelain)"
test -f "$STAGING/LICENSE"
test -f "$STAGING/ravens/test.py"
mv "$STAGING" upstream
echo "Official Ravens source ready at commit $COMMIT"
