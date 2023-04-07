import numpy as np
import itertools as it

# create dictionaries to go back and forth between ACGT and one-hot vectors
nuc_enc = dict(zip("ACGT", np.identity(4, dtype=bool)))
nuc_complement = {char: set(nuc_enc.keys())-{char} for char in nuc_enc.keys()}
nuc_dec = {v.tobytes(): char for char, v in nuc_enc.items()}

# These two functions translate between one-hot vectors and ACGT
def seq2vec(seq: str) -> np.ndarray:
    return np.array([nuc_enc[char] for char in seq])

def vec2seq(vec: np.ndarray) -> str:
    if vec.ndim == 1:
        return nuc_dec[vec.tobytes()]
    return ''.join(nuc_dec[v.tobytes()] for v in vec)

def _seq_complements(dist=1):
    if dist==1:
        return nuc_complement
    return {''.join(cs): {''.join(css) for css in it.product(*[nuc_complement[cs[i]] for i in range(dist)])} for cs in it.product(nuc_complement.keys(), repeat=dist)}

def neighboring_seqs(seq: str, dist=1):
    neighbors = []
    complements = _seq_complements(dist)
    for poss in it.combinations(range(len(seq)), r=dist):
        for complement in complements[''.join([seq[i] for i in poss])]:
            iter_comp = iter(complement)
            neighbors.append(''.join([(next(iter_comp) if i in poss else c) for i, c in enumerate(seq)]))
    return neighbors
    