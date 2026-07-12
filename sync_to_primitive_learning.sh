#!/usr/bin/env bash
#
# sync_to_primitive_learning.sh — one-way clean-release mirror of this repo's
# committed HEAD into DR-LfD-all/primitive_learning.
#
#   * Source of truth is `git archive HEAD`: only committed, git-tracked files
#     ship. Uncommitted working-tree edits are NOT synced until committed; the
#     source .git and untracked files never enter the snapshot.
#   * Dev cruft (the torch_compile diff dump and this script) is pruned via
#     EXCLUDE.
#   * Deletions are reconciled via a self-managed manifest: only files this
#     script created on a previous run are eligible for removal, so sibling
#     content in the DR-LfD-all repo is never touched.
#   * The destination is a subdirectory of a SEPARATE git repo (DR-LfD-all), so
#     the manifest is kept inside THIS repo's .git/ — never written into DST,
#     where it would pollute the downstream repository.
#
# Usage:
#   bash sync_to_primitive_learning.sh              # reconcile DST to HEAD
#   bash sync_to_primitive_learning.sh --dry-run    # preview adds/updates + deletions
#   bash sync_to_primitive_learning.sh --watch=30   # poll every 30s
#
set -euo pipefail

SRC="/home/user/yzchen_ws/docker_share_folder/difussion/equibot_abstract"
DST="/home/user/yzchen_ws/TAMP-ubuntu22/DR-LfD-all/primitive_learning"

# Manifest lives in this repo's git metadata, not in DST: the destination is
# nested in another git repo and must receive synced content only.
MANIFEST="$(git -C "$SRC" rev-parse --absolute-git-dir)/sync_to_primitive_learning.manifest"

# Excluded paths (anchored, relative to repo root): the saved torch_compile diff
# dump and this sync script itself (never ship the tooling, even once committed).
EXCLUDE='^(torch_compile\.txt|sync_to_primitive_learning\.sh)$'

DRYRUN=""
WATCH=""
for a in "$@"; do
  case "$a" in
    --dry-run) DRYRUN=1 ;;
    --watch=*)
      WATCH="${a#*=}"
      [[ "$WATCH" =~ ^[1-9][0-9]*$ ]] || { echo "--watch needs a positive integer of seconds" >&2; exit 2; }
      ;;
    *) echo "unknown arg: $a" >&2; exit 2 ;;
  esac
done

TMP=""
cleanup() { [[ -n "$TMP" ]] && rm -rf -- "$TMP"; TMP=""; }
trap cleanup EXIT INT TERM

sync_once() {
  local snapshot keep f
  local -a rsync_args=(-a --checksum)

  TMP="$(mktemp -d)"
  snapshot="$TMP/snapshot"
  keep="$TMP/keep"
  mkdir -p -- "$snapshot"

  # 1. Materialize a clean HEAD snapshot (committed content only).
  git -C "$SRC" archive --format=tar HEAD | tar -xf - -C "$snapshot"

  # 2. Build keep-list: snapshot files (and tracked symlinks) minus EXCLUDE.
  #    grep exit 1 (everything filtered) is fine; any other error aborts.
  (
    cd "$snapshot"
    find . \( -type f -o -type l \) | sed 's#^\./##' | { grep -vE "$EXCLUDE" || [[ $? -eq 1 ]]; }
  ) | sort -u > "$keep"

  # Refuse to proceed on an empty keep-list: it would delete every managed file
  # from DST in step 4. HEAD always has non-excluded content, so empty means a
  # snapshot/build failure, not an intended state.
  [[ -s "$keep" ]] || { echo "keep-list is empty — aborting before any deletion" >&2; return 1; }

  # 3. Copy/update kept files. --checksum prevents re-shipping unchanged binary
  #    weights merely because `git archive` restamps mtimes to the commit time.
  if [[ -n "$DRYRUN" ]]; then
    rsync_args+=(--dry-run --itemize-changes)
  else
    mkdir -p -- "$DST"
  fi
  rsync "${rsync_args[@]}" --files-from="$keep" "$snapshot"/ "$DST"/

  # 4. Reconcile deletions: remove only files we managed last time but no longer
  #    keep. Sibling files in DR-LfD-all (never in the manifest) are untouched.
  if [[ -f "$MANIFEST" ]]; then
    while IFS= read -r f; do
      [[ -n "$f" ]] || continue
      case "$f" in
        /*|..|../*|*/../*|*/..) echo "refusing unsafe manifest path: $f" >&2; return 1 ;;
      esac
      if [[ -n "$DRYRUN" ]]; then
        echo "would delete: $f"
      else
        rm -f -- "$DST/$f"
      fi
    done < <(comm -13 "$keep" <(sort -u "$MANIFEST"))
  fi

  # 5. Record the current managed set (skipped in dry-run).
  [[ -n "$DRYRUN" ]] || cp -- "$keep" "$MANIFEST"
  echo "synced $(wc -l < "$keep") managed files -> $DST"

  cleanup
}

if [[ -n "$WATCH" ]]; then
  echo "watch: re-syncing every ${WATCH}s (Ctrl-C to stop)"
  while true; do
    sync_once
    sleep "$WATCH"
  done
else
  sync_once
fi
