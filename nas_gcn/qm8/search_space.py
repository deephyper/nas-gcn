from nas_gcn.search_space_utils import create_search_space


def new_search_space():
    """Create a search space for QM8 dataset.
    Returns:
        A search space containing multiple Keras architectures.
    """
    INPUT_SHAPE = [(10, 75), (25, 75), (25, 14), (25,), (10,)]
    OUTPUT_SHAPE = (16,)
    NUM_MPNN_CELLS = 3
    NUM_DENSE_LAYERS = 2
    arch = create_search_space(input_shape=INPUT_SHAPE,
                               output_shape=OUTPUT_SHAPE,
                               num_mpnn_cells=NUM_MPNN_CELLS,
                               num_dense_layers=NUM_DENSE_LAYERS)
    return arch


if __name__ == '__main__':
    arch = new_search_space()
