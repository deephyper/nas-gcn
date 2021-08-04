from deephyper.problem import NaProblem
from nas_gcn.qm7.load_data import load_data
from nas_gcn.search_space_utils import create_search_space

Problem = NaProblem(seed=2020)
Problem.load_data(load_data)
Problem.search_space(create_search_space, data='qm7')
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