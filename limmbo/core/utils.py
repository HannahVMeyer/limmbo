import dask.array as da
import numpy as np


def get_chunks(G):
    if hasattr(G, "chunks"):
        return G.chunks[1]

    siz = G.shape[1] // 100
    sizl = G.shape[1] - siz * 100
    chunks = [siz] * 100
    if sizl > 0:
        chunks += [sizl]
    return tuple(chunks)

def _biallelic_dosage(geno):
    multiplier = da.arange(3, chunks=3, dtype=np.float64)
    dosage = np.sum(multiplier * geno, axis=2)
    return np.asarray(dosage)
