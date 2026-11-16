#!/usr/bin/env bash
# Download the SKU-110K archive with parallel range requests.
# The S3 bucket serves a single connection at a few hundred KiB/s; splitting
# the file into ranges is several times faster. Re-running resumes each part.
# Stalled connections (under 20 KB/s for 30 s) are dropped and reconnected.
set -euo pipefail

URL="http://trax-geometry.s3.amazonaws.com/cvpr_challenge/SKU110K_fixed.tar.gz"
OUT_DIR="${1:-data}"
PARTS="${PARTS:-16}"
OUT="$OUT_DIR/SKU110K_fixed.tar.gz"
TMP="$OUT_DIR/.sku110k-parts"

[ -f "$OUT" ] && { echo "already downloaded: $OUT"; exit 0; }
mkdir -p "$TMP"

size=$(curl -sI "$URL" | tr -d '\r' | awk 'tolower($1)=="content-length:"{print $2}')
chunk=$(( (size + PARTS - 1) / PARTS ))
echo "size: $size bytes, $PARTS parts of $chunk bytes"

fetch_part() {
  local i=$1 start=$(( $1 * chunk )) end=$(( ($1 + 1) * chunk - 1 ))
  (( end >= size )) && end=$(( size - 1 ))
  local want=$(( end - start + 1 )) f="$TMP/part-$(printf %03d "$i")"
  for _ in $(seq 1 50); do
    local have=0
    [ -f "$f" ] && have=$(stat -c %s "$f")
    (( have >= want )) && return 0
    curl -s --connect-timeout 20 -Y 20000 -y 30 -r "$(( start + have ))-$end" "$URL" >> "$f" || sleep 2
  done
  echo "part $i failed" >&2
  return 1
}

pids=()
for i in $(seq 0 $(( PARTS - 1 ))); do fetch_part "$i" & pids+=($!); done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
(( fail == 0 )) || { echo "some parts failed; re-run to resume" >&2; exit 1; }

cat "$TMP"/part-* > "$OUT.tmp"
[ "$(stat -c %s "$OUT.tmp")" = "$size" ] || { echo "size mismatch" >&2; exit 1; }
gzip -t "$OUT.tmp" || { echo "archive is corrupt; delete $TMP and $OUT.tmp, then re-run" >&2; exit 1; }
mv "$OUT.tmp" "$OUT"
rm -rf "$TMP"
echo "done: $OUT"
