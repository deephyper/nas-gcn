.. _create-new-nas-problem:

Neural Architecture Search for Graph Neural Networks
***********************************************

In this tutorial, we will recreate results from out recent paper on `Graph Neural Network Architecture Search for Molecular Property Prediction <https://ieeexplore.ieee.org/abstract/document/9378060>`_.

Install from Github
=======================
You can directly download the code and install the package using ``pip``

.. code-block:: bash

    git pull https://github.com/deephyper/nas-gcn.git
    cd nas-gcn
    python -m pip install -e .

.. warning::
    The paper was based on deepchem==2.4.0rc1.dev20200819015415. The GNN code using the newest deepchem can be found `here <https://github.com/deephyper/nas-gcn/tree/master/nas_gcn_new_deepchem>`_. Using the newest deepchem package may lead to result difference。

Setting up the problem
=======================

Let’s start by creating a new DeepHyper project workspace. This is a directory where you will create search problem instances that are automatically installed and importable across your Python environment.

.. code-block:: console

    deephyper start-project dh_project

A new dh_project directory is created, containing the following files::

    dh_project/
        dh_project/
            __init__.py
        setup.py


We can now define our neural architecture search problem inside this directory. Let’s set up a NAS problem called ``lstm_search`` as follows:

.. code-block:: console

    cd dh_project/dh_project/
    deephyper new-problem nas gnn_search

A new NAS problem subdirectory should be in place. This is a Python subpackage containing sample code in the files ``__init__.py``, ``load_data.py``, ``search_space.py``, and ``problem.py``. Overall, your project directory should look like::

    dh_project/
        dh_project/
            __init__.py
            gnn_search/
                __init__.py
                load_data.py
                search_space.py
                problem.py
        setup.py

Load the data
===================

Fist, we will look at the ``load_data.py`` file that loads and returns the
training and validation data. The example ``load_data`` function generates the QM9 dataset. You can explore `here <https://github.com/deephyper/nas-gcn/tree/master/nas_gcn>`_ for other datasets, including QM7, QM8, ESOL, FreeSolv and Lipophilicity. Some of the dataset are shrunk to a smaller size because of memory limit. But you can use the full dataset.

.. code-block:: python

    from deepchem.molnet import load_qm9
    from nas_gcn.data_utils import load_molnet_data


    def load_data(test=False, seed=2020):
        """Load QM9 dataset

        Args:
            test (bool): when training DeepHyper, set to False.
            seed (int): the random seed used to split the data.

        Returns:
            If test is True, return training, validation and testing data
            and labels, task name and transformer.
            If test is False, return training and validation data and labels.
        """
        # FIXED PARAMETERS
        MAX_ATOM = 10
        MAX_EDGE = 17
        N_FEAT = 75
        E_FEAT = 14

        FUNC = load_qm9
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
            # Shrink the data to its one-tenth
            X_train = [X_train[i][::10, ...] for i in range(len(X_train))]
            X_valid = [X_valid[i][::10, ...] for i in range(len(X_valid))]
            y_train = y_train[::10]
            y_valid = y_valid[::10]
            return (X_train, y_train), (X_valid, y_valid)


    if __name__ == '__main__':
        load_data(test=False)




Define a neural architecture search space
======================
The MPNN cell is the core of search space. You can vary the state dimension ``state_dims``, number of message passing ``T``, attention methods ``attn_methods``, number of attention heads ``attn_heads``, aggregation methods ``aggr_methods``, update methods ``update_methods`` and activation functions ``activations``.

.. code-block:: python

    def mpnn_cell(node):
        """Create a variable node of MPNN cell.
        Args:
            node: A DeepHyper variable node object.
        Returns:
            A variable node of MPNN cell.
        """
        state_dims = [4, 8, 16, 32]
        Ts = [1, 2, 3, 4]
        attn_methods = ['const', 'gcn', 'gat', 'sym-gat', 'linear', 'gen-linear', 'cos']
        attn_heads = [1, 2, 4, 6]
        aggr_methods = ['max', 'mean', 'sum']
        update_methods = ['gru', 'mlp']
        activations = [tf.keras.activations.sigmoid,
                       tf.keras.activations.tanh,
                       tf.keras.activations.relu,
                       tf.keras.activations.linear,
                       tf.keras.activations.elu,
                       tf.keras.activations.softplus,
                       tf.nn.leaky_relu,
                       tf.nn.relu6]

        for state_dim in state_dims:
            for T in Ts:
                for attn_method in attn_methods:
                    for attn_head in attn_heads:
                        for aggr_method in aggr_methods:
                            for update_method in update_methods:
                                for activation in activations:
                                    node.add_op(SPARSE_MPNN(state_dim=state_dim,
                                                            T=T,
                                                            attn_method=attn_method,
                                                            attn_head=attn_head,
                                                            aggr_method=aggr_method,
                                                            update_method=update_method,
                                                            activation=activation))
        return


The gather cell is the readout layer of the network. The functions include global sum pooling, global max pooling, global average pooling, where you can pool with respect to node space or feature space. You can ask choose global attention pooling and global attention sum pooling.

.. code-block:: python

    def gather_cell(node):
        """Create a variable node of Gather cell.
        Args:
            node: A DeepHyper variable node object.
        Returns:
            A variable node of Gather cell.
        """
        for functions in [GlobalSumPool, GlobalMaxPool, GlobalAvgPool]:
            for axis in [-1, -2]:  # Pool in terms of nodes or features
                node.add_op(functions(axis=axis))
        node.add_op(Flatten())
        for state_dim in [16, 32, 64]:
            node.add_op(GlobalAttentionPool(state_dim=state_dim))
        node.add_op(GlobalAttentionSumPool())
        return


The final search space also includes skip-connection. You can select your own dataset and vary the ``input_shape`` and ``output_shape``.

.. code-block:: python

    def create_search_space(input_shape=None,
                            output_shape=None,
                            num_mpnn_cells=3,
                            num_dense_layers=2,
                            **kwargs):
        """Create a search space containing multiple Keras architectures
        Args:
            input_shape (list): the input shapes, e.g. [(3, 4), (5, 2)].
            output_shape (tuple): the output shape, e.g. (12, ).
            num_mpnn_cells (int): the number of MPNN cells.
            num_dense_layers (int): the number of Dense layers.
        Returns:
            A search space containing multiple Keras architectures
        """
        data = kwargs['data']
        if data == 'qm7':
            input_shape = [(8+1, 75), (8+1+10+1, 2), (8+1+10+1, 14), (8+1, ), (8+1+10+1, )]
            output_shape = (1, )
        elif data == 'qm8':
            input_shape = [(9+1, 75), (9+1+14+1, 2), (9+1+14+1, 14), (9+1, ), (9+1+14+1, )]
            output_shape = (16, )
        elif data == 'qm9':
            input_shape = [(9+1, 75), (9+1+16+1, 2), (9+1+16+1, 14), (9+1, ), (9+1+16+1, )]
            output_shape = (12, )
        elif data == 'freesolv':
            input_shape = [(24+1, 75), (24+1+25+1, 2), (24+1+25+1, 14), (24+1, ), (24+1+25+1, )]
            output_shape = (1, )
        elif data == 'esol':
            input_shape = [(55+1, 75), (55+1+68+1, 2), (55+1+68+1, 14), (55+1, ), (55+1+68+1, )]
            output_shape = (1, )
        elif data == 'lipo':
            input_shape = [(115+1, 75), (115+1+236+1, 2), (115+1+236+1, 14), (115+1, ), (115+1+236+1, )]
            output_shape = (1, )
        arch = KSearchSpace(input_shape, output_shape, regression=True)
        source = prev_input = arch.input_nodes[0]
        prev_input1 = arch.input_nodes[1]
        prev_input2 = arch.input_nodes[2]
        prev_input3 = arch.input_nodes[3]
        prev_input4 = arch.input_nodes[4]

        # look over skip connections within a range of the 3 previous nodes
        anchor_points = collections.deque([source], maxlen=3)

        count_gcn_layers = 0
        count_dense_layers = 0
        for _ in range(num_mpnn_cells):
            graph_attn_cell = VariableNode()
            mpnn_cell(graph_attn_cell)  #
            arch.connect(prev_input, graph_attn_cell)
            arch.connect(prev_input1, graph_attn_cell)
            arch.connect(prev_input2, graph_attn_cell)
            arch.connect(prev_input3, graph_attn_cell)
            arch.connect(prev_input4, graph_attn_cell)

            cell_output = graph_attn_cell
            cmerge = ConstantNode()
            cmerge.set_op(AddByProjecting(arch, [cell_output], activation="relu"))

            for anchor in anchor_points:
                skipco = VariableNode()
                skipco.add_op(Tensor([]))
                skipco.add_op(Connect(arch, anchor))
                arch.connect(skipco, cmerge)

            prev_input = cmerge
            anchor_points.append(prev_input)
            count_gcn_layers += 1

        global_pooling_node = VariableNode()
        gather_cell(global_pooling_node)
        arch.connect(prev_input, global_pooling_node)
        prev_input = global_pooling_node

        flatten_node = ConstantNode()
        flatten_node.set_op(Flatten())
        arch.connect(prev_input, flatten_node)
        prev_input = flatten_node

        for _ in range(num_dense_layers):
            dense_node = ConstantNode()
            dense_node.set_op(Dense(32, activation='relu'))
            arch.connect(prev_input, dense_node)
            prev_input = dense_node
            count_dense_layers += 1

        output_node = ConstantNode()
        output_node.set_op(Dense(output_shape[0], activation='linear'))
        arch.connect(prev_input, output_node)

        return arch


Create a problem instance
==================

Now, we will take a look at ``problem.py`` which contains the code for the
problem definition. ``negmae`` is the negative mean absolute error. You can also defined your own ``tf.keras`` metrics.

.. code-block:: python

    from deephyper.problem import NaProblem
    from nas_gcn.qm8.load_data import load_data
    from nas_gcn.search_space_utils import create_search_space

    Problem = NaProblem(seed=2020)
    Problem.load_data(load_data)
    Problem.search_space(create_search_space, data='qm8')
    Problem.hyperparameters(
        batch_size=128,
        learning_rate=1e-3,
        optimizer='adam',
        num_epochs=50)
    Problem.loss("mae")
    Problem.metrics(['mae', 'mse', 'r2', 'negmae'])
    Problem.objective('val_negmae__max')


    if __name__ == '__main__':
        print(Problem)


Running the search on LCRC Bebop
==============================================

Now let's run the search on LCRC Bebop

.. code-block:: bash

    srun -n 30 python -m tuster.system.bebop.run 'python -m deephyper.search.nas.regevo --evaluator ray --redis-address {redis_address} --problem nas_gcn.qm9.problem.Problem'


MPNN related functions
==============================================
You can find all the following codes `here <https://github.com/deephyper/nas-gcn/blob/master/nas_gcn/search/stack_mpnn.py>`_.

``SPARSE_MPNN`` is the main message passing cell. Any input node features are first map to ``state_dim`` with ``X = self.embed(X)``. Then we run the message passing ``T`` times.

.. code-block:: python

    class SPARSE_MPNN(tf.keras.layers.Layer):
        r"""Message passing cell.
        Args:
            state_dim (int): number of output channels.
            T (int): number of message passing repetition.
            attn_heads (int): number of attention heads.
            attn_method (str): type of attention methods.
            aggr_method (str): type of aggregation methods.
            activation (str): type of activation functions.
            update_method (str): type of update functions.
        """
        def __init__(self,
                     state_dim,
                     T,
                     aggr_method,
                     attn_method,
                     update_method,
                     attn_head,
                     activation):
            super(SPARSE_MPNN, self).__init__(self)
            self.state_dim = state_dim
            self.T = T
            self.activation = activations.get(activation)
            self.aggr_method = aggr_method
            self.attn_method = attn_method
            self.attn_head = attn_head
            self.update_method = update_method

        def build(self, input_shape):
            self.embed = tf.keras.layers.Dense(self.state_dim, activation=self.activation)
            self.MP = MP_layer(self.state_dim, self.aggr_method, self.activation,
                               self.attn_method, self.attn_head, self.update_method)
            self.built = True

        def call(self, inputs, **kwargs):
            """
            Args:
                inputs (list):
                    X (tensor): node feature tensor
                    A (tensor): edge pair tensor
                    E (tensor): edge feature tensor
                    mask (tensor): node mask tensor to mask out non-existent nodes
                    degree (tensor): node degree tensor for GCN attention
            Returns:
                X (tensor): results after several repetitions of edge network, attention, aggregation and update function
            """
            X, A, E, mask, degree = inputs
            A = tf.cast(A, tf.int32)
            X = self.embed(X)
            for _ in range(self.T):
                X = self.MP([X, A, E, mask, degree])
            return X


``MP_layer`` is the message passing layer. Here you perform the message passing ``agg_m = self.message_passer([X, A, E, degree])``, aggregation ``agg_m = tf.multiply(agg_m, mask)`` and update operations ``updated_nodes = self.update_functions([X, agg_m])``. We also include masking operations to filter any non-existent nodes.

.. code-block:: python

    class MP_layer(tf.keras.layers.Layer):
        r"""Message passing layer.
        Args:
            state_dim (int): number of output channels.
            attn_heads (int): number of attention heads.
            attn_method (str): type of attention methods.
            aggr_method (str): type of aggregation methods.
            activation (str): type of activation functions.
            update_method (str): type of update functions.
        """
        def __init__(self, state_dim, aggr_method, activation, attn_method, attn_head, update_method):
            super(MP_layer, self).__init__(self)
            self.state_dim = state_dim
            self.aggr_method = aggr_method
            self.activation = activation
            self.attn_method = attn_method
            self.attn_head = attn_head
            self.update_method = update_method

        def build(self, input_shape):
            self.message_passer = Message_Passer_NNM(self.state_dim, self.attn_head, self.attn_method,
                                                     self.aggr_method, self.activation)
            if self.update_method == 'gru':
                self.update_functions = Update_Func_GRU(self.state_dim)
            elif self.update_method == 'mlp':
                self.update_functions = Update_Func_MLP(self.state_dim, self.activation)

            self.built = True

        def call(self, inputs, **kwargs):
            """
            Args:
                inputs (list):
                    X (tensor): node feature tensor
                    A (tensor): edge pair tensor
                    E (tensor): edge feature tensor
                    mask (tensor): node mask tensor to mask out non-existent nodes
                    degree (tensor): node degree tensor for GCN attention
            Returns:
                updated_nodes (tensor): results after edge network, attention, aggregation and update function
            """
            X, A, E, mask, degree = inputs
            agg_m = self.message_passer([X, A, E, degree])
            mask = tf.tile(mask[..., None], [1, 1, self.state_dim])
            agg_m = tf.multiply(agg_m, mask)
            updated_nodes = self.update_functions([X, agg_m])
            updated_nodes = tf.multiply(updated_nodes, mask)
            return updated_nodes

``Message_Passer_NNM`` is the message passing kernel. Here you perform the attention, aggregate operations. Since we use edge pairs instead of adjacency matrices. The ``source index`` and ``sink index`` of an edge pair facilitate the message passing operation using ``tf.gather`` and aggregation operation such as ``tf.math.unsorted_segment_max``.

.. note::

    The edge network ``W = self.nn(E)`` only contains a single dense layer. In our recent study, we found by building a more sophisticated edge network, the prediction error is smaller.

.. code-block:: python

    class Message_Passer_NNM(tf.keras.layers.Layer):
        r"""Message passing kernel.
        Args:
            state_dim (int): number of output channels.
            attn_heads (int): number of attention heads.
            attn_method (str): type of attention methods.
            aggr_method (str): type of aggregation methods.
            activation (str): type of activation functions.
        """
        def __init__(self, state_dim, attn_heads, attn_method, aggr_method, activation):
            super(Message_Passer_NNM, self).__init__()
            self.state_dim = state_dim
            self.attn_heads = attn_heads
            self.attn_method = attn_method
            self.aggr_method = aggr_method
            self.activation = activation

        def build(self, input_shape):
            self.nn = tf.keras.layers.Dense(units=self.state_dim * self.state_dim * self.attn_heads,
                                            activation=self.activation)

            if self.attn_method == 'gat':
                self.attn_func = Attention_GAT(self.state_dim, self.attn_heads)
            elif self.attn_method == 'sym-gat':
                self.attn_func = Attention_SYM_GAT(self.state_dim, self.attn_heads)
            elif self.attn_method == 'cos':
                self.attn_func = Attention_COS(self.state_dim, self.attn_heads)
            elif self.attn_method == 'linear':
                self.attn_func = Attention_Linear(self.state_dim, self.attn_heads)
            elif self.attn_method == 'gen-linear':
                self.attn_func = Attention_Gen_Linear(self.state_dim, self.attn_heads)
            elif self.attn_method == 'const':
                self.attn_func = Attention_Const(self.state_dim, self.attn_heads)
            elif self.attn_method == 'gcn':
                self.attn_func = Attention_GCN(self.state_dim, self.attn_heads)

            self.bias = self.add_weight(name='attn_bias', shape=[self.state_dim], initializer='zeros')
            self.built = True

        def call(self, inputs, **kwargs):
            """
            Args:
                inputs (list):
                    X (tensor): node feature tensor
                    A (tensor): edge pair tensor
                    E (tensor): edge feature tensor
                    degree (tensor): node degree tensor for GCN attention
            Returns:
                output (tensor): results after edge network, attention and aggregation
            """
            # Edge network to transform edge information to message weight
            X, A, E, degree = inputs
            N = K.int_shape(X)[1]
            targets, sources = A[..., -2], A[..., -1]
            W = self.nn(E)
            W = tf.reshape(W, [-1, tf.shape(E)[1], self.attn_heads, self.state_dim, self.state_dim])
            X = tf.tile(X[..., None], [1, 1, 1, self.attn_heads])
            X = tf.transpose(X, [0, 1, 3, 2])

            # Attention added to the message weight
            attn_coef = self.attn_func([X, N, targets, sources, degree])
            messages = tf.gather(X, sources, batch_dims=1)
            messages = messages[..., None]
            messages = tf.matmul(W, messages)
            messages = messages[..., 0]
            output = attn_coef * messages
            num_rows = tf.shape(targets)[0]
            rows_idx = tf.range(num_rows)
            segment_ids_per_row = targets + N * tf.expand_dims(rows_idx, axis=1)

            # Aggregation to summarize neighboring node messages
            if self.aggr_method == 'max':
                output = tf.math.unsorted_segment_max(output, segment_ids_per_row, N * num_rows)
            elif self.aggr_method == 'mean':
                output = tf.math.unsorted_segment_mean(output, segment_ids_per_row, N * num_rows)
            elif self.aggr_method == 'sum':
                output = tf.math.unsorted_segment_sum(output, segment_ids_per_row, N * num_rows)

            # Output the mean of all attention heads
            output = tf.reshape(output, [-1, N, self.attn_heads, self.state_dim])
            output = tf.reduce_mean(output, axis=-2)
            output = K.bias_add(output, self.bias)
            return output


We also have two update functions [`details <https://ieeexplore.ieee.org/abstract/document/9378060>`_]:
    - Gated recurrent unit update function ``Update_Func_GRU``
    - Multi-layer perceptron update function ``Update_Func_MLP``.

We have Seven attention functions [`details <https://ieeexplore.ieee.org/abstract/document/9378060>`_]:
    - GAT attention ``Attention_GAT``
    - GAT symmetry attention ``Attention_SYM_GAT``
    - COS attention ``Attention_COS``
    - Linear attention ``Attention_Linear``,
    - Generalized linear attention ``Attention_Gen_Linear``
    - GCN attention``Attention_GCN``
    - Constant attention ``Attention_Const``.