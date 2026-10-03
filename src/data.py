"""Dataset loading. Downloaded once, cached as .npz, never re-fetched."""
import gzip
import os
import numpy as np

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def _idx(path):
    with gzip.open(path, "rb") as f:
        raw = f.read()
    ndim = raw[3]
    dims = [int.from_bytes(raw[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(raw, np.uint8, offset=4 + 4 * ndim).reshape(dims)


def load(name="mnist"):
    """Returns (xtr uint8 [N,784], ytr int32, xte, yte)."""
    if name == "mnist":
        d = np.load(os.path.join(DATA, "mnist.npz"))
        xtr, ytr, xte, yte = d["x_train"], d["y_train"], d["x_test"], d["y_test"]
    elif name == "fashion":
        cache = os.path.join(DATA, "fashion.npz")
        if not os.path.exists(cache):
            p = lambda f: os.path.join(DATA, f"fashion-{f}.gz")
            np.savez_compressed(
                cache,
                x_train=_idx(p("train-images-idx3-ubyte")),
                y_train=_idx(p("train-labels-idx1-ubyte")),
                x_test=_idx(p("t10k-images-idx3-ubyte")),
                y_test=_idx(p("t10k-labels-idx1-ubyte")))
        d = np.load(cache)
        xtr, ytr, xte, yte = d["x_train"], d["y_train"], d["x_test"], d["y_test"]
    else:
        raise ValueError(name)
    f = lambda x: x.reshape(len(x), -1).astype(np.uint8)
    return f(xtr), ytr.astype(np.int32), f(xte), yte.astype(np.int32)
