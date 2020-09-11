def convert_data(X, E, MAX_ATOM, MAX_EDGE, N_FEAT, E_FEAT):
    """Convert an X(node feature), E(edge feature) to desired input format for GNN.

    Args:
        X (np.array): node features.
        E (np.array): edge features.
        MAX_ATOM (int): the maximum number of atoms zero-padding to.
        MAX_EDGE (int): the maximum number of edges zero-padding to.
        N_FEAT (int): the number of node features.
        E_FEAT (int): the number of edge features.

    Returns:
        Xo (np.array): the node features (batch * MAX_ATOM * N_FEAT).
        Ao (np.array): the edge pairs (batch * MAX_EDGE * 2).
        Eo (np.array): the edge features (batch * MAX_EDGE * E_FEAT).
        Mo (np.array): the mask of actual atoms (batch * MAX_ATOM).
        No (np.array): the inverse sqrt of node degrees for GCN attention 1/sqrt(N(i)*N(j)) (batch * MAX_EDGE).
    """
    import numpy as np
    import scipy.sparse as sp

    # The adjacency matrix A, the first 6 elements of E are bond information.
    A = E[..., :6].sum(axis=-1) != 0
    A = A.astype(np.float32)

    # The node feature Xo
    Xo = np.zeros(shape=(MAX_ATOM, N_FEAT))
    Xo[:X.shape[0], :X.shape[1]] = X

    # Convert A to edge pair format (if I use A_0 = np.zeros(...), the 0 to 0 pair will be emphasized a lot)
    # So I set all to the max_atom, then the max_atom atom has no node features.
    # And I mask all calculation for existing atoms.
    Ao = np.ones(shape=(MAX_EDGE + MAX_ATOM, 2)) * (MAX_ATOM - 1)
    A = sp.coo_matrix(A)
    n_edge = len(A.row)
    Ao[:n_edge, 0] = A.row
    Ao[:n_edge, 1] = A.col

    # The edge feature Eo
    Eo = np.zeros(shape=(MAX_EDGE + MAX_ATOM, E_FEAT))
    Eo[:n_edge, :] = [e[a.row, a.col] for e, a in zip([E], [A])][0]

    # Fill the zeros in Ao with self loop
    Ao[MAX_EDGE:, 0] = np.array([i for i in range(MAX_ATOM)])
    Ao[MAX_EDGE:, 1] = np.array([i for i in range(MAX_ATOM)])

    # The mask for existing nodes
    Mo = np.zeros(shape=(MAX_ATOM,))
    Mo[:X.shape[0]] = 1

    # The inverse of sqrt of node degrees
    outputa = np.unique(Ao[:, 0], return_counts=True, return_inverse=True)
    outputb = np.unique(Ao[:, 1], return_counts=True, return_inverse=True)
    n_a = []
    for element in outputa[1]:
        n_a.append(outputa[2][element])
    n_b = []
    for element in outputb[1]:
        n_b.append(outputb[2][element])
    n_a = np.array(n_a)
    n_b = np.array(n_b)
    no = np.multiply(n_a, n_b)
    No = 1 / np.sqrt(no)

    return Xo, Ao, Eo, Mo, No
