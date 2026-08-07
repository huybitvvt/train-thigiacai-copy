from __future__ import annotations

import hashlib

import numpy as np


def frame_fingerprint(frame: np.ndarray) -> str:
    """Identify one frozen camera frame without treating a repeated QR as a duplicate."""
    contiguous = np.ascontiguousarray(frame)
    digest = hashlib.blake2b(contiguous.data, digest_size=16).hexdigest()
    return f"{contiguous.shape}:{contiguous.dtype}:{digest}"
