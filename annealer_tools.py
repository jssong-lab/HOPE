import math
import numpy as np

one_hots = {v.tobytes() for v in np.identity(4, dtype=bool)}
one_hot_complement = {v: [np.frombuffer(w, dtype=bool) for w in one_hots - {v}] for v in one_hots}

# generates a neighboring sequence in vector form
def neighbors(rng, vec, change):
    out_vec = vec.copy()
    poss = rng.choice(out_vec.shape[0], size=change, replace=False)
    if change == 1:
        out_vec[poss, :] = rng.choice(one_hot_complement[out_vec[poss, :].tobytes()])
        return out_vec
    
    for pos in poss:
        out_vec[pos, :] = rng.choice(one_hot_complement[out_vec[pos, :].tobytes()])           
    return out_vec

# provides acceptance chance
def accept(q_a, init_eval, new_eval, temp):
    delta = new_eval - init_eval
    if delta < 0:
        return 1.0
    if q_a == 1:
        return math.exp(-delta/temp)
    denum = 1 + (q_a-1)*delta/temp
    if (q_a < 1) & (denum < 0):
        return 0.0
    else:
        return denum**(-1/(q_a-1))
    
def temp(q_v, init_temp, iteration):
    if q_v == 1:
        return init_temp*math.log(2, 1+iteration)
    else:
        return init_temp*(2**(q_v - 1) - 1)/((1 + iteration)**(q_v - 1) - 1)