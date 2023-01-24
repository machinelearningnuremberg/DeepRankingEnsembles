import gc
import time
import pickle
import gpytorch
import numpy as np


def store_object(obj, obj_name):
    with open(obj_name, "wb") as fp:
        pickle.dump(obj, fp)


def load_object(obj_name):
    with open(obj_name, "rb") as fp:
        return pickle.load(fp)


def get_all_combinations(hpob_hdlr, n_trials):
    # Total combinations -
    #   430 combinations for HPO-B test dataset
    #   456 combinations for the HPO-B validation dataset.
    seed_list = ["test0", "test1", "test2", "test3", "test4"]
    evaluation_list = []
    for search_space in sorted(hpob_hdlr.get_search_spaces()):
        for dataset in sorted(hpob_hdlr.get_datasets(search_space)):
            for seed in seed_list:
                evaluation_list += [(search_space, dataset, seed, n_trials)]

    return evaluation_list


def get_input_dim(meta_data):
    dataset_key = list(meta_data.keys())[0]
    dim = np.array(meta_data[dataset_key]["X"]).shape[1]
    return dim


def convert_meta_data_to_np_dictionary(meta_data):
    temp_meta_data = {}
    for k in meta_data.keys():
        X = np.array(meta_data[k]["X"], dtype=np.float32)
        y = np.array(meta_data[k]["y"], dtype=np.float32)
        temp_meta_data[k] = {"X": X, "y": y}

    return temp_meta_data


# Created as a stub for parallel evaluations.
def evaluation_worker(hpob_hdlr, method, args):
    search_space, dataset, seed, n_trials = args
    print(search_space, dataset, seed, n_trials)
    res = []
    try:
        t_start = time.time()
        res = hpob_hdlr.evaluate(method,
                                  search_space_id=search_space,
                                  dataset_id=dataset,
                                  seed=seed,
                                  n_trials=n_trials)
        t_end = time.time()
        print(search_space, dataset, seed, n_trials, "Completed in", t_end - t_start, "s")
    # This exception needs to be ignored due to issues with Gaussian Processes fitting the HPO-B data.
    except gpytorch.utils.errors.NotPSDError:
        print("NotPSDError (Not Positive Semi Definite Error) encountered while evaluating. Not recording this as a valid evaluation combination.")
        res = []
    return (search_space, dataset, seed, n_trials), res


def evaluate_combinations(hpob_hdlr, method, keys_to_evaluate):
    print("Evaluating for", method)

    evaluation_list = []
    for key in keys_to_evaluate:
        search_space, dataset, seed, n_trials = key
        evaluation_list += [(search_space, dataset, seed, n_trials)]

    performance = []
    run_i = 0
    for eval_instance in evaluation_list:
        result = evaluation_worker(hpob_hdlr, method, eval_instance)
        performance.append(result)
        run_i = run_i + 1
        print("Completed Running", run_i, end="\n")
        gc.collect()

    return performance

