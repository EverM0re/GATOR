#!/usr/bin/env bash
# Move data/ and store/ outside File_Router and symlink them back.
#
# The project folder is re-uploaded wholesale between runs, so anything inside
# it is destroyed on every update -- including corpora that take an hour to
# fetch and ingest.  This relocates the heavy directories to a sibling that
# updates never touch, and leaves symlinks so every path in the config, the
# scripts, and the docs keeps working unchanged.
#
#   bash scripts/link_external_data.sh
#   FR_DATA_ROOT=/mnt/big/fr bash scripts/link_external_data.sh
#
# Safe to re-run: an already-linked directory is left alone.
set -uo pipefail
cd "$(dirname "$0")/.."

ROOT="${FR_DATA_ROOT:-$(cd .. && pwd)/file_router_data}"
mkdir -p "$ROOT"
echo "[link] external root: $ROOT"

for name in data store; do
  target="$ROOT/$name"
  if [[ -L "$name" ]]; then
    echo "[link] $name -> $(readlink "$name") (already linked)"
    continue
  fi
  mkdir -p "$target"
  if [[ -d "$name" ]]; then
    if [[ -n "$(ls -A "$name" 2>/dev/null)" ]]; then
      # A manual upload replaces the symlink with the uploader's local copy,
      # which is usually a small smoke corpus.  Merging that over a populated
      # external root silently downgrades it -- an 8-QA corpus overwriting a
      # 300-QA one.  Keep whichever side is larger and say so.
      # Compare corpus size for data/ (QA lines) and byte size for store/,
      # which holds sqlite and vector indexes rather than qa.jsonl.
      if [[ "$name" == "data" ]]; then
        here=$(find "$name" -name 'qa.jsonl' -exec cat {} + 2>/dev/null | wc -l)
        there=$(find "$target" -name 'qa.jsonl' -exec cat {} + 2>/dev/null | wc -l)
        unit="QA"
      else
        here=$(du -sk "$name" 2>/dev/null | cut -f1)
        there=$(du -sk "$target" 2>/dev/null | cut -f1)
        unit="KB"
      fi
      if [[ "$there" -gt 0 && "$here" -lt "$there" ]]; then
        echo "[link] $name/ here holds $here $unit; external holds $there $unit."
        echo "[link] keeping the external copy and discarding the uploaded one."
        rm -rf "$name"
      else
        echo "[link] moving existing $name/ into $target/ ..."
        cp -a "$name/." "$target/" || { echo "[link] copy failed; leaving $name in place" >&2; continue; }
        rm -rf "$name"
      fi
    else
      rmdir "$name"
    fi
  fi
  ln -s "$target" "$name"
  echo "[link] $name -> $target"
done

cat <<TXT

[link] done. The heavy directories now live outside File_Router:
       $ROOT/data    corpora (raw + unified)
       $ROOT/store   ingested node stores and vector indexes

       Re-uploading File_Router no longer destroys them. Re-run this script
       after each upload to recreate the two symlinks (it will find the data
       already there and simply relink).
TXT
