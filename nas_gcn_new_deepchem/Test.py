from nas_gcn.qm7.load_data import load_data
(X_train, y_train), (X_valid, y_valid) = load_data(test=False, seed=2020)
print('nas_gcn.qm7.load_data', f"{y_train.shape}")

from nas_gcn.qm8.load_data import load_data
(X_train, y_train), (X_valid, y_valid) = load_data(test=False, seed=2020)
print('nas_gcn.qm8.load_data', f"{y_train.shape}")

from nas_gcn.qm9.load_data import load_data
(X_train, y_train), (X_valid, y_valid) = load_data(test=False, seed=2020)
print('nas_gcn.qm9.load_data', f"{y_train.shape}")

from nas_gcn.esol.load_data import load_data
(X_train, y_train), (X_valid, y_valid) = load_data(test=False, seed=2020)
print('nas_gcn.esol.load_data', f"{y_train.shape}")

from nas_gcn.freesolv.load_data import load_data
(X_train, y_train), (X_valid, y_valid) = load_data(test=False, seed=2020)
print('nas_gcn.freesolv.load_data', f"{y_train.shape}")

from nas_gcn.lipophilicity.load_data import load_data
(X_train, y_train), (X_valid, y_valid) = load_data(test=False, seed=2020)
print('nas_gcn.lipophilicity.load_data', f"{y_train.shape}")