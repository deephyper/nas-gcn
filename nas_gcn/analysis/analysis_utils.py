import json
import pickle
import glob
import numpy as np
import pandas as pd
from tabulate import tabulate
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


def load_json(path):
    """Load json file.
    Args:
        path (str): file location

    Returns:
        data (dict)
    """
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def to_sec(ts):
    """Format time string to seconds.

    Args:
        ts (string): time string.

    Returns:
        time (float): second format
    """
    try:
        return datetime.strptime(ts, '%Y-%m-%d %H:%M:%S').timestamp()
    except:
        return datetime.strptime(ts, '%Y-%m-%d %H:%M:%S.%f').timestamp()


def three_random_split(DATA_DIR, multi_class=False):
    """Combine results from three random seed trainings.

    Args:
        DATA_DIR (str): data pickle file location.
        multi_class (bool): if the regression has multi-class.

    Returns:
        train_true (np.array): training data true labels.
        train_pred (np.array): training data predicted labels.
        valid_true (np.array): validation data true labels.
        valid_pred (np.array): validation data predicted labels.
        test_true (np.array): testing data true labels.
        test_pred (np.array): testing data predicted labels.
    """
    y_true = []
    y_pred = []
    files = sorted(glob.glob(DATA_DIR + 'best_archs_result_0_*.pickle'))
    for file in files:
        with open(file, 'rb') as f:
            _ = pickle.load(f)
            for _ in range(3):
                if multi_class:
                    y_true.append(pickle.load(f)[np.newaxis, ...])
                    y_pred.append(pickle.load(f).squeeze()[np.newaxis, ...])
                else:
                    y_true.append(pickle.load(f).ravel())
                    y_pred.append(pickle.load(f).ravel().squeeze())
    train_true = np.vstack([y_true[i] for i in [0, 3, 6]])
    train_pred = np.vstack([y_pred[i] for i in [0, 3, 6]])
    valid_true = np.vstack([y_true[i] for i in [1, 4, 7]])
    valid_pred = np.vstack([y_pred[i] for i in [1, 4, 7]])
    test_true = np.vstack([y_true[i] for i in [2, 5, 8]])
    test_pred = np.vstack([y_pred[i] for i in [2, 5, 8]])
    return train_true, train_pred, valid_true, valid_pred, test_true, test_pred


def three_random_mean_std(DATA_DIR, multi_class=False):
    """Calculate the mean and standard deviation of three random seed trainings.

    Args:
        DATA_DIR (str): data pickle file location.
        multi_class (bool): if the regression has multi-class.

    Returns:
        m (float): mean value.
        s (float): standard deviation value.
    """
    output = three_random_split(DATA_DIR, multi_class=multi_class)
    funcs = [mean_absolute_error, mean_squared_error, r2_score]

    if not multi_class:
        result = []
        for func in funcs:
            for i in range(3):
                result.append([func(output[i * 2][j], output[i * 2 + 1][j]) for j in range(len(output[0]))])
        result = np.array(result)
        m = result.mean(axis=1)
        s = result.std(axis=1)
        print(tabulate(
            [['Train', f'{m[0]:0.4f}+/-{s[0]:0.4f}', f'{m[3]:0.4f}+/-{s[3]:0.4f}', f'{m[6]:0.4f}+/-{s[6]:0.4f}'],
             ['Valid', f'{m[1]:0.4f}+/-{s[1]:0.4f}', f'{m[4]:0.4f}+/-{s[4]:0.4f}', f'{m[7]:0.4f}+/-{s[7]:0.4f}'],
             ['Test', f'{m[2]:0.4f}+/-{s[2]:0.4f}', f'{m[5]:0.4f}+/-{s[5]:0.4f}', f'{m[8]:0.4f}+/-{s[8]:0.4f}']],
            headers=['', 'MAE', 'MSE', 'R2']))
    else:
        for c in range(output[0].shape[-1]):
            result = []
            for func in funcs:
                for i in range(3):
                    result.append(
                        [func(output[i * 2][j, :, c], output[i * 2 + 1][j, :, c]) for j in range(len(output[0]))])
            result = np.array(result)
            m = result.mean(axis=1)
            s = result.std(axis=1)
            print(tabulate(
                [['Train', f'{m[0]:0.4f}+/-{s[0]:0.4f}', f'{m[3]:0.4f}+/-{s[3]:0.4f}', f'{m[6]:0.4f}+/-{s[6]:0.4f}'],
                 ['Valid', f'{m[1]:0.4f}+/-{s[1]:0.4f}', f'{m[4]:0.4f}+/-{s[4]:0.4f}', f'{m[7]:0.4f}+/-{s[7]:0.4f}'],
                 ['Test', f'{m[2]:0.4f}+/-{s[2]:0.4f}', f'{m[5]:0.4f}+/-{s[5]:0.4f}', f'{m[8]:0.4f}+/-{s[8]:0.4f}']],
                headers=['', 'MAE', 'MSE', 'R2']))
    return m, s


def create_csv(DATA_DIR, data):
    """Create a csv file of the architecture components.

    Args:
        DATA_DIR (str): data file location.
        data (dict): the dictionary file containing the operations for each architecture.

    """
    # Task specific
    state_dims = ['dim(4)', 'dim(8)', 'dim(16)', 'dim(32)']
    Ts = ['repeat(1)', 'repeat(2)', 'repeat(3)', 'repeat(4)']
    attn_methods = ['attn(const)', 'attn(gcn)', 'attn(gat)', 'attn(sym-gat)', 'attn(linear)', 'attn(gen-linear)',
                    'attn(cos)']
    attn_heads = ['head(1)', 'head(2)', 'head(4)', 'head(6)']
    aggr_methods = ['aggr(max)', 'aggr(mean)', 'aggr(sum)']
    update_methods = ['update(gru)', 'update(mlp)']
    activations = ['act(sigmoid)', 'act(tanh)', 'act(relu)', 'act(linear)', 'act(elu)', 'act(softplus)',
                   'act(leaky_relu)',
                   'act(relu6)']

    out = []
    for state_dim in state_dims:
        for T in Ts:
            for attn_method in attn_methods:
                for attn_head in attn_heads:
                    for aggr_method in aggr_methods:
                        for update_method in update_methods:
                            for activation in activations:
                                out.append(
                                    [state_dim, T, attn_method, attn_head, aggr_method, update_method, activation])

    out_pool = []
    for functions in ['GlobalSumPool', 'GlobalMaxPool', 'GlobalAvgPool']:
        for axis in ['(feature)', '(node)']:  # Pool in terms of nodes or features
            out_pool.append(functions + axis)
    out_pool.append('flatten')
    for state_dim in [16, 32, 64]:
        out_pool.append(f'AttentionPool({state_dim})')
    out_pool.append('AttentionSumPool')

    out_connect = ['skip', 'connect']

    def get_gat(index):
        return out[index]

    def get_pool(index):
        return out_pool[index]

    def get_connect(index):
        return out_connect[index]

    archs = np.array(data['arch_seq'])
    rewards = np.array(data['raw_rewards'])
    a = np.empty((len(archs), 0), dtype=np.object)
    a = np.append(a, archs, axis=-1)
    a = np.append(a, rewards[..., np.newaxis], axis=-1)
    b = np.empty((0, 29), dtype=np.object)
    for i in range(len(a)):
        temp = a[i, :]
        b0 = [get_gat(temp[0])[i] + '[cell1]' for i in range(len(get_gat(temp[0])))]
        b1 = [get_connect(temp[1]) + '[link1]']
        b2 = [get_gat(temp[2])[i] + '[cell2]' for i in range(len(get_gat(temp[2])))]
        b3 = [get_connect(temp[3]) + '[link2]']
        b4 = [get_connect(temp[4]) + '[link3]']
        b5 = [get_gat(temp[5])[i] + '[cell3]' for i in range(len(get_gat(temp[5])))]
        b6 = [get_connect(temp[6]) + '[link4]']
        b7 = [get_connect(temp[7]) + '[link5]']
        b8 = [get_connect(temp[8]) + '[link6]']
        b9 = [get_pool(temp[9])]
        bout = b0 + b1 + b2 + b3 + b4 + b5 + b6 + b7 + b8 + b9 + [temp[10]]
        bout = np.array(bout, dtype=object)
        b = np.append(b, bout[np.newaxis, ...], axis=0)
    table = pd.DataFrame(data=b)
    table.to_csv(DATA_DIR + 'nas_result.csv', encoding='utf-8', index=False, header=False)


def moving_average(time_list, data_list, window_size=100):
    """Calculate the moving average.

    Args:
        time_list (list): a list of timestamps.
        data_list (list): a list of data points.
        window_size (int): the window size.

    Returns:
        time array and data array
    """
    res_list = []
    times_list = []
    for i in range(len(data_list) - window_size):
        times_list.append(sum(time_list[i:i + window_size]) / window_size)
        res_list.append(sum(data_list[i:i + window_size]) / window_size)
    return np.array(times_list), np.array(res_list)


def plot_reward_vs_time(data, PLOT_DIR, ylim=None, time=True, plot=False, metric='MAE'):
    """Generate plot of search trajectory.

    Args:
        data (dict): the data dictionary.
        PLOT_DIR (str): the location to store the figure.
        ylim (float): the minimum value of the y axis.
        time (bool): True if want time as x axis, else want instance number.
        plot (bool): if want to create a plot.
        metric (str): the type of metric on y axis.

    """
    start_infos = data['start_infos'][0]
    try:
        start_time = to_sec(data['workload']['times'][0])
    except:
        start_time = to_sec(start_infos['timestamp'])
    times = [to_sec(ts) - start_time for ts in data['timestamps']]
    x = times
    y = data['raw_rewards']
    plt.figure(figsize=(5, 4))
    if time:
        plt.plot(np.array(x) / 60, y, 'o', markersize=3)
        plt.xlabel('Time (min)')
    else:
        plt.plot(y, 'o', markersize=3)
        plt.xlabel('Iterations')
    plt.ylabel(f'Reward (-{metric})')

    plt.xlim(left=0)
    if ylim is not None:
        plt.ylim(ylim)
    plt.locator_params(axis='y', nbins=4)
    plt.savefig(PLOT_DIR + 'reward.png', dpi=300, bbox_inches='tight')
    plt.savefig(PLOT_DIR+'reward.svg', bbox_inches='tight')
    if not plot:
        plt.close();


def three_random_parity_plot(DATA_DIR, PLOT_DIR, multi_class=False, limits=None, plot=False, ticks=None):
    """Generate parity plots from three random seed trainings.

    Args:
        DATA_DIR (str): the location of the data file.
        PLOT_DIR (str): the location to store the figure.
        multi_class (bool): if it is multi-class regression.
        limits (list): the y limits you want to set.
        plot (bool): if want to create a plot.
        ticks (list): the x axis ticks.

    """
    _, _, _, _, y_true_raw, y_pred_raw = three_random_split(DATA_DIR, multi_class=multi_class)
    if not multi_class:
        y_true = y_true_raw.ravel()
        y_pred = y_pred_raw.ravel()
        scaler = StandardScaler()
        y_true = scaler.fit_transform(y_true[..., np.newaxis]).squeeze()
        y_pred = scaler.fit_transform(y_pred[..., np.newaxis]).squeeze()
        fig, ax = plt.subplots(figsize=(4, 4))
        min_value = np.min([y_true.min(), y_pred.min()])
        max_value = np.max([y_true.max(), y_pred.max()])
        dist = max_value - min_value
        min_value -= 0.03 * dist
        max_value += 0.03 * dist
        if limits is not None:
            min_value, max_value = limits
        ax.plot(np.linspace(min_value, max_value, 100), np.linspace(min_value, max_value, 100), 'k--', alpha=0.5)
        ax.scatter(y_true.ravel(), y_pred.ravel(), s=5, alpha=0.9)
        plt.xlim(min_value, max_value)
        plt.ylim(min_value, max_value)
        plt.xlabel("True")
        plt.ylabel("Predicted")
        print(min_value, max_value)
        from matplotlib import ticker
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1, 1))
        if ticks is not None:
            plt.xticks(ticks, ticks)
            plt.yticks(ticks, ticks)
        else:
            plt.locator_params(axis='x', nbins=5)
            plt.locator_params(axis='y', nbins=5)
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
        # plt.tight_layout()
        plt.savefig(PLOT_DIR + "parity_plot.png", bbox_inches='tight')
        plt.savefig(PLOT_DIR + "parity_plot.svg", bbox_inches='tight')
        if not plot:
            plt.close();
    else:
        for c in range(y_true_raw.shape[-1]):
            y_true = y_true_raw[..., c].ravel()
            y_pred = y_pred_raw[..., c].ravel()
            plt.figure(figsize=(4, 4))
            min_value = np.min([y_true.min(), y_pred.min()])
            max_value = np.max([y_true.max(), y_pred.max()])
            dist = max_value - min_value
            min_value -= 0.03 * dist
            max_value += 0.03 * dist
            if limits is not None:
                min_value, max_value = limits
            plt.plot(np.linspace(min_value, max_value, 100), np.linspace(min_value, max_value, 100), 'k--', alpha=0.5)
            plt.scatter(y_true.ravel(), y_pred.ravel(), s=5, alpha=0.9)
            plt.xlim(min_value, max_value)
            plt.ylim(min_value, max_value)
            plt.xlabel("True")
            plt.ylabel("Predicted")
            plt.locator_params(axis='x', nbins=5)
            plt.locator_params(axis='y', nbins=5)
            plt.savefig(PLOT_DIR + f"parity_plot_{c}.png", bbox_inches='tight')
            if not plot:
                plt.close();


def feature_importance(DATA_DIR, PLOT_DIR, plot=False):
    """Generate feature importance plots.

    Args:
        DATA_DIR (str): the location of the data file.
        PLOT_DIR (str): the location to store the figure.
        plot (bool): if want to create a plot.

    """
    train_data = pd.read_csv(DATA_DIR + 'nas_result.csv', header=None)
    df = train_data
    df_new = pd.DataFrame()
    for i in range(df.shape[1]):
        if df.dtypes[i] == 'object':
            vals = pd.get_dummies(df.iloc[:, i])
        else:
            vals = df.iloc[:, i]
        df_new = pd.concat([df_new.reset_index(drop=True), vals.reset_index(drop=True)], axis=1)
    X = df_new.iloc[:, :-1]
    y = df_new.iloc[:, -1]
    scaler = StandardScaler()
    y = scaler.fit_transform(y.values[..., np.newaxis]).squeeze()
    reg = RandomForestRegressor(n_estimators=100, random_state=0).fit(X.values, y)

    prediction, bias, contributions = ti.predict(reg, X.values)
    mask = np.copy(X.values)
    mask = mask.astype(float)
    mask[mask == 0] = -1
    importance = np.multiply(contributions, mask)
    importance = importance.mean(axis=0)
    importance = importance / np.max(np.abs(importance))
    indices = np.argsort(importance)[-5:]
    indices_neg = np.argsort(importance)[:5]
    plt.figure(figsize=(12, 4))
    plt.barh(range(5, 10), importance[indices], align='center')
    plt.barh(range(5), importance[indices_neg], align='center')
    plt.yticks(range(10), [X.columns[i] for i in indices_neg] + [X.columns[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig(PLOT_DIR + 'feature_importance.png', dpi=300, bbox_inches='tight')
    if not plot:
        plt.close();
