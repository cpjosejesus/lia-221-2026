#!/bin/sh
# Build and run every ec2 program on the CPU emulation; print one line per program.
# Usage: ./run_all.sh   (from the ec2 folder; set CXX to choose the compiler, default c++)
CXX=${CXX:-c++}
BIN=$(mktemp -d)
fail=0
for f in */*.cu; do
  dir=$(dirname "$f"); name=$(basename "$f" .cu)
  if ! (cd "$dir" && $CXX -std=c++17 -O2 -pthread -Wall -Wextra -x c++ -I../common "$name.cu" -o "$BIN/$name"); then
    echo "BUILD FAIL  $f"; fail=1; continue
  fi
  if (cd "$dir" && "$BIN/$name" > "$BIN/$name.out" 2>&1); then
    printf 'PASS        %s\n' "$f"
  else
    printf 'FAIL        %s  (see %s)\n' "$f" "$BIN/$name.out"; fail=1
  fi
done
exit $fail
