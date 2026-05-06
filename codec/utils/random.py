from __future__ import annotations

import hashlib
import os
import time


def new_epoch_seed(base_seed: int = 137) -> int:
    raw = f"{base_seed}-{time.time_ns()}-{os.getpid()}-{os.urandom(16).hex()}".encode()
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "little") & 0xFFFFFFFF
