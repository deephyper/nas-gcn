from nas_gcn.search_space_utils import create_search_space


def new_search_space():
    """Create a search space for Lipophilicity dataset.
    Returns:
        A search space containing multiple Keras architectures.
    """
    INPUT_SHAPE = [(116, 75), (353, 75), (353, 14), (353,), (51,)]
    OUTPUT_SHAPE = (1,)
    NUM_MPNN_CELLS = 3
    NUM_DENSE_LAYERS = 2
    arch = create_search_space(input_shape=INPUT_SHAPE,
                               output_shape=OUTPUT_SHAPE,
                               num_mpnn_cells=NUM_MPNN_CELLS,
                               num_dense_layers=NUM_DENSE_LAYERS)
    return arch


if __name__ == '__main__':
    arch = new_search_space()
