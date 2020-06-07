import os
import logging
import json
import numpy as np
import torch


def sort_dataset(data, labels, num_classes=10, stack=False):
    sorted_data = [[] for _ in range(num_classes)]
    for i, lbl in enumerate(labels):
        sorted_data[lbl].append(data[i])
    sorted_data = [np.stack(class_data) for class_data in sorted_data]
    sorted_labels = [np.repeat(i, (len(sorted_data[i]))) for i in range(num_classes)]
    if stack:
        sorted_data = np.vstack(sorted_data)
        sorted_labels = np.hstack(sorted_labels)
    return sorted_data, sorted_labels


def init_pipeline(model_dir, headers=None):
    """Initialize folder and csv logger. """
    # project folder
    os.makedirs(model_dir)
    os.makedirs(os.path.join(model_dir, 'checkpoints'))
    os.makedirs(os.path.join(model_dir, 'figures'))
    os.makedirs(os.path.join(model_dir, 'plabels'))
    if not headers:
        headers = ["epoch", "step", "loss", "discrimn_loss_e", "compress_loss_e", 
            "discrimn_loss_t",  "compress_loss_t"]
    create_csv(model_dir, 'losses.csv', headers)
    print("project dir: {}".format(model_dir))


def create_csv(model_dir, filename, headers):
    csv_path = os.path.join(model_dir, filename)
    if os.path.exists(csv_path):
        os.remove(csv_path)
    with open(csv_path, 'w+') as f:
        f.write(','.join(map(str, headers)))
    return csv_path


def save_params(model_dir, params):
    path = os.path.join(model_dir, 'params.json')
    with open(path, 'w') as f:
        json.dump(params, f, indent=2, sort_keys=True)


def update_params(model_dir, pretrain_dir):
    params = load_params(model_dir)
    old_params = load_params(pretrain_dir)
    params['arch'] = old_params["arch"]
    params['fd'] = old_params['fd']
    save_params(model_dir, params)


def load_params(model_dir):
    _path = os.path.join(model_dir, "params.json")
    with open(_path, 'r') as f:
        _dict = json.load(f)
    return _dict


def save_state(model_dir, *entries, filename='losses.csv'):
    csv_path = os.path.join(model_dir, filename)
    assert os.path.exists(csv_path), 'CSV file is missing in project directory.'
    with open(csv_path, 'a') as f:
        f.write('\n'+','.join(map(str, entries)))


def save_ckpt(model_dir, net, epoch):
    torch.save(net.state_dict(), os.path.join(model_dir, 'checkpoints', 
        'model-epoch{}.pt'.format(epoch)))


def save_labels(model_dir, labels, epoch):
    path = os.path.join(model_dir, 'plabels', f'epoch{epoch}.npy')
    np.save(path, labels)


def compute_accuracy(y_pred, y_true):
    assert y_pred.shape == y_true.shape
    return 1 - np.count_nonzero(y_pred - y_true) / y_true.size


def clustering_accuracy(labels_true, labels_pred):
    from sklearn.metrics.cluster import supervised
    from scipy.optimize import linear_sum_assignment
    labels_true, labels_pred = supervised.check_clusterings(labels_true, labels_pred)
    value = supervised.contingency_matrix(labels_true, labels_pred)
    [r, c] = linear_sum_assignment(-value)
    return value[r, c].sum() / len(labels_true)