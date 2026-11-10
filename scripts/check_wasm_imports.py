#!/usr/bin/env python3
"""Fail if a DuckDB-Wasm extension module imports any anofox_ml_* function.

The Rust FFI must be linked into the .wasm; an import means it was left out
and the module can't load (DataZooDE/anofox-ml#29).
"""
import sys


def leb(data, i):
    n = shift = 0
    while True:
        b = data[i]
        i += 1
        n |= (b & 0x7F) << shift
        shift += 7
        if b < 0x80:
            return n, i


def name(data, i):
    n, i = leb(data, i)
    return data[i:i + n].decode(errors="replace"), i + n


def imports(data):
    i = 8
    while i < len(data):
        section, i = data[i], i + 1
        size, i = leb(data, i)
        if section == 2:
            count, j = leb(data, i)
            for _ in range(count):
                _, j = name(data, j)
                field, j = name(data, j)
                kind, j = data[j], j + 1
                if kind == 0:  # function: type index
                    _, j = leb(data, j)
                elif kind == 1:  # table: element type, limits
                    j += 1
                    flags, j = data[j], j + 1
                    _, j = leb(data, j)
                    if flags & 1:
                        _, j = leb(data, j)
                elif kind == 2:  # memory: limits
                    flags, j = data[j], j + 1
                    _, j = leb(data, j)
                    if flags & 1:
                        _, j = leb(data, j)
                elif kind == 3:  # global: type, mutability
                    j += 2
                elif kind == 4:  # tag
                    j += 1
                    _, j = leb(data, j)
                yield field
        i += size


bad = 0
for path in sys.argv[1:]:
    with open(path, "rb") as f:
        missing = [f for f in imports(f.read()) if f.startswith("anofox_ml_")]
    if missing:
        bad = 1
        print(f"{path}: {len(missing)} anofox_ml_* functions are imports, not linked: {missing[:5]}")
    else:
        print(f"{path}: Rust FFI linked")
sys.exit(bad)
