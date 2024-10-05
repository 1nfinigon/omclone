#!/usr/bin/env python3

import struct
from pathlib import Path

for path in Path("test/tensorboard").glob("**/events.out.*"):
    print("trimming {}".format(path))
    data = b""
    old_size = None
    new_size = None
    with path.open('rb') as f:
        while True:
            length_chunk = f.read(8)
            if not length_chunk:
                break
            (length,) = struct.unpack("<Q", length_chunk)
            if length < 3000:
                chunk = f.read(length + 8)
                assert(len(chunk) == length + 8)
                data += length_chunk + chunk
            old_size = f.tell()
    with path.open('wb') as f:
        f.write(data)
        new_size = f.tell()
    print("old size: {}   new size: {}".format(old_size, new_size))
