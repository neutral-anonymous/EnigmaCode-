# -*- coding: utf-8 -*-
"""
Experiment Runner for Enigma Obfuscation
Handles dataset generation and parallel execution of obfuscation experiments.
"""

import os
import time
import random
from joblib import Parallel, delayed
from core_utils import generate_random_ising, save_pickle, load_pickle
from enigma import enigma


def run_single_experiment(graph_type, graph_param, trial_num, enigma_try, 
                          data_path, out_data_path, generate_ising_flag, generate_enigma_flag):
    """
    Run a single obfuscation experiment.
    
    Args:
        graph_type: type of graph ('er', 'ba', 'reg', 'sk')
        graph_param: graph-specific parameter
        trial_num: trial number
        enigma_try: enigma version number
        data_path: path to original data
        out_data_path: path to output data
        generate_ising_flag: whether to generate new Ising model
        generate_enigma_flag: whether to run Enigma obfuscation
    
    Returns:
        execution_time: time taken for obfuscation (if run), else None
    """
    _name = f'{graph_type.lower()}'
    if graph_type in ['ba', 'reg']:
        _name = f'{graph_type.lower()}{graph_param}'

    n = random.randint(500, 1000)
    if graph_type in ['reg'] and n % 2 == 1:
        n += 1

    ising_filename = f'{data_path}original/{_name}_{trial_num}.pkl'

    if generate_ising_flag:
        h, J = generate_random_ising(
            n=n,
            graph_type=graph_type,
            graph_param=graph_param,
            weight_mode='uniform'
        )
        ising_obj = {'h': h, 'J': J}
        save_pickle(ising_obj, ising_filename)

    if generate_enigma_flag:
        ising_obj = load_pickle(ising_filename)

        t0 = time.time()
        ising_list, info = enigma(ising_obj['h'], ising_obj['J'])
        execution_time = time.time() - t0

        enigma_filename = f'{out_data_path}enigma_{enigma_try}/{_name}_{trial_num}.pkl'
        enigma_obj = {'ising_list': ising_list, 'info': info, 'time': execution_time}
        save_pickle(enigma_obj, enigma_filename)

        return execution_time
    return None


def run_experiments_parallel(graph_type_list, T1, T2, enigma_try_list, 
                              data_path, expr_folder, n_jobs=15, 
                              generate_ising=False, generate_enigma=True):
    """
    Run experiments in parallel across multiple cores.
    
    Args:
        graph_type_list: list of (graph_type, graph_param) tuples
        T1, T2: trial range [T1, T2]
        enigma_try_list: list of enigma version numbers
        data_path: base data path
        expr_folder: experiment folder name
        n_jobs: number of parallel jobs
        generate_ising: whether to generate new Ising models
        generate_enigma: whether to run Enigma obfuscation
    
    Returns:
        time_list: list of execution times
        runtime: total runtime
    """
    out_data_path = f'{data_path}{expr_folder}'

    print(f'graph_type_list: ({len(graph_type_list)})', graph_type_list)
    start_time = time.time()

    # Build job list
    jobs = []
    for graph_type, graph_param in graph_type_list:
        print(graph_type)
        for trial_num in range(T1, T2 + 1):
            for enigma_try in enigma_try_list:
                jobs.append((trial_num, graph_type, graph_param, enigma_try))

    # Run in parallel
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
        delayed(run_single_experiment)(
            graph_type, graph_param, trial_num, enigma_try,
            data_path, out_data_path, generate_ising, generate_enigma
        )
        for trial_num, graph_type, graph_param, enigma_try in jobs
    )

    runtime = time.time() - start_time
    time_list = [t for t in results if t is not None]

    print()
    print('================')
    print('Runtime: ', runtime)
    print(f'Avg. ({len(time_list)}): {sum(time_list) / len(time_list) if time_list else 0}')

    return time_list, runtime


def run_experiments_sequential(graph_type_list, T1, T2, enigma_try_list,
                                data_path, expr_folder,
                                generate_ising=False, generate_enigma=True):
    """
    Run experiments sequentially (single-threaded).
    
    Args:
        graph_type_list: list of (graph_type, graph_param) tuples
        T1, T2: trial range [T1, T2]
        enigma_try_list: list of enigma version numbers
        data_path: base data path
        expr_folder: experiment folder name
        generate_ising: whether to generate new Ising models
        generate_enigma: whether to run Enigma obfuscation
    
    Returns:
        time_list: list of execution times
    """
    out_data_path = f'{data_path}{expr_folder}'
    time_list = []

    for graph_type, graph_param in graph_type_list:
        _name = f'{graph_type.lower()}'
        if graph_type in ['ba', 'reg']:
            _name = f'{graph_type.lower()}{graph_param}'
        print(_name)

        for trial_num in range(T1, T2 + 1):
            if trial_num % 50 == 0:
                print('    Instance: ', trial_num)

            execution_time = run_single_experiment(
                graph_type, graph_param, trial_num, enigma_try_list[0],
                data_path, out_data_path, generate_ising, generate_enigma
            )
            if execution_time is not None:
                time_list.append(execution_time)

    print()
    print('================')
    print(f'Avg. ({len(time_list)}): {sum(time_list) / len(time_list) if time_list else 0}')

    return time_list
