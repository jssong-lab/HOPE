import argparse
import os
import numpy as np
import pandas as pd
import keras

import nuc_translation as nt
import annealer_tools

from functools import partial


def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument('sequence', type=str, help='Initial sequence of nucleotids')
    parser.add_argument('indicator', type=str, help='A string of ones and zeros the same length as the initial sequnce')
    parser.add_argument('--qa', type=float, default=1.15, help='value greater than 1.0 to control acceptance rate')
    parser.add_argument('--qv', type=float, default=1.30, help='value greater than 1.0 to control variation rate of tempurature')
    parser.add_argument('-s', '--seed', type=int, help='Seed for random number generator')
    parser.add_argument('-j', '--jobID', type=str, help='Job ID for file name to resolve conflicts')
    parser.add_argument('-o', '--outdir', type=str, help='output directory')

    args = parser.parse_args()

    return args.sequence, args.indicator, args.qa, args.qv, args.seed, args.jobID, args.outdir


def get_evaluator(model_file, fixed_nucs, insert_loc, indicator):
    model = keras.models.load_model(filepath = model_file, compile = False)
    length = indicator.shape[0]
    
    # convert from (n, 4) to (n+2, 5) before getting model prediction
    def evaluator(annealing_sequence):
        full_sequence = np.insert(annealing_sequence, insert_loc, fixed_nucs, axis=0)
        network_input = np.hstack((full_sequence, indicator)).reshape((1, 1, length, 5))
        return model.predict(network_input).squeeze()
    
    return evaluator


if __name__ == '__main__':
    # get inputs
    # e.g. jobID = 98923_24, outdir = cutoff_30_test/PBSL13_RTTL15
    full_sequence, indicator, qa, qv, seed, jobID, outdir = setup()

    # set up random number generator
    rng = np.random.default_rng(seed=seed)

    # if jobID is none, assign it a random number to avoid potential conflicts
    if not jobID:
        jobID = str(rng.integers(9999)).zfill(4)

    init_temp = 5.0
    max_iter = 2_000_001

    # load model from hdf5 file e.g.
    model_file = "/home/groups/song/songlab2/dnewton2/FinalSim3/model/model.hdf5"

    # strip out the two fixed 'GG' since they shouldn't ever be changed
    # input sequence e.g. GGTCGGTATGGCCGTTACTGATAATGGTGGAGTACGCAATTCCCGTC
    fixed_nuc_loc = 25
    fixed_nuc_length = 2
    fixed_nuc_seq = full_sequence[fixed_nuc_loc: fixed_nuc_loc+fixed_nuc_length]
    fixed_nuc_vec = nt.seq2vec(fixed_nuc_seq)
    variable_sequence = full_sequence[:fixed_nuc_loc] + full_sequence[fixed_nuc_loc+fixed_nuc_length:]
    variable_vec = nt.seq2vec(variable_sequence)

    # indicator sequence e.g. 00000000111111111111111111111111111100000000000
    indicator_vec = np.array(list(indicator)).astype(int).reshape((len(indicator),1))

    # get a function to evaluate the sequences as they anneal
    evaluator = get_evaluator(model_file, fixed_nuc_vec, fixed_nuc_loc, indicator_vec)
    tempurature = partial(annealer_tools.temp, q_v=qv, init_temp=init_temp)
    accept_chance = partial(annealer_tools.accept, q_a=qa)
    neighbors = partial(annealer_tools.neighbors, rng=rng, change=1)

    # perform annealing and log the results
    previous_state, previous_eval = variable_vec, evaluator(variable_vec)
    data_log = {}
    data_log[0] = {'Tempurature': init_temp, 'Acceptance_Chance': 1.0, 'Sequence': full_sequence, 'Evaluation': previous_eval}
    for t in range(1, max_iter):
        new_state = neighbors(vec=previous_state)
        new_eval = evaluator(new_state)
        current_temp = tempurature(iteration=t)
        current_accept_chance = accept_chance(init_eval=previous_eval, new_eval=new_eval, temp=current_temp)
        if current_accept_chance <= rng.uniform():
            previous_state, previous_eval = new_state, new_eval
            full_sequence = nt.vec2seq(np.insert(new_state, fixed_nuc_loc, fixed_nuc_vec, axis=0))

        data_log[t] = {'Tempurature': current_temp, 'Acceptance_Chance': 1-current_accept_chance, 'Sequence': full_sequence, 'Evaluation': previous_eval}

    # save the results
    file_name = f'qa{qa:.2f}_qv{qv:.2f}_init{init_temp}_iter{int(max_iter/1_000)}k_{jobID}'.replace('.', '')
    if outdir is not None:
        file_path = f'{outdir}/{file_name}.csv'
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    else:
        file_path = f'{file_name}.csv'
    
    df_data_log = pd.DataFrame.from_dict(data_log, orient='index')
    df_data_log.to_csv(file_path)