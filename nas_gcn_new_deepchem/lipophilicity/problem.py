from deephyper.problem import NaProblem
from nas_gcn.lipophilicity.load_data import load_data
from nas_gcn.search_space_utils import create_search_space

Problem = NaProblem(seed=2020)
Problem.load_data(load_data)
Problem.search_space(create_search_space, data='lipo')
Problem.hyperparameters(
    batch_size=128,
    learning_rate=1e-3,
    optimizer='adam',
    num_epochs=20)
Problem.loss("mse")
Problem.metrics(['mae', 'mse', 'r2', 'negmse'])
Problem.objective('val_negmse__max')

if __name__ == '__main__':
    print(Problem)
