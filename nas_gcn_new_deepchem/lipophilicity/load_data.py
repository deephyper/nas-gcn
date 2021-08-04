from deepchem.molnet import load_lipo
from nas_gcn.data_utils import load_molnet_data


def load_data(test=False, seed=2020):
    """Load Lipophilicity dataset

    Args:
        test (bool): when training DeepHyper, set to False.
        seed (int): the random seed used to split the data.

    Returns:
        If test is True, return training, validation and testing data
        and labels, task name and transformer.
        If test is False, return training and validation data and labels.
    """
    # FIXED PARAMETERS
    MAX_ATOM = 116
    MAX_EDGE = 237
    N_FEAT = 75
    E_FEAT = 14

    FUNC = load_lipo
    FEATURIZER = 'Weave'
    SPLIT = 'random'

    X_train, y_train, X_valid, y_valid, X_test, y_test, \
    tasks, transformers = load_molnet_data(func=FUNC,
                                           featurizer=FEATURIZER,
                                           split=SPLIT,
                                           seed=seed,
                                           MAX_ATOM=MAX_ATOM,
                                           MAX_EDGE=MAX_EDGE,
                                           N_FEAT=N_FEAT,
                                           E_FEAT=E_FEAT)
    if test:
        return (X_train, y_train), (X_valid, y_valid), (X_test, y_test), \
               tasks, transformers

    else:
        # Shrink the data to its half
        X_train = [X_train[i][::2, ...] for i in range(len(X_train))]
        X_valid = [X_valid[i][::2, ...] for i in range(len(X_valid))]
        y_train = y_train[::2]
        y_valid = y_valid[::2]
        return (X_train, y_train), (X_valid, y_valid)


if __name__ == '__main__':
    load_data(test=False)
