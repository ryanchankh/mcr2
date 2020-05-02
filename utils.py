import os
import logging
import json
import numpy as np


def one_hot(targets, num_classes):
    """One-hot-ify labels.

    Args: 
        targets (np.ndarray): should have shape (num_samples)
        num_classes (int): number of classes for the labels

    Return:
        np.ndarray with shape (num_samples, num_classes)

    """
    if num_classes is None:
        num_classes = targets.max() + 1
    res = np.eye(num_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[num_classes]).astype(np.float32)


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


def init_pipeline(model_dir):
    """Initialize folder and csv logger. """
    # project folder
    os.makedirs(model_dir)
    print("project folder: {}".format(model_dir))

    # csv
    CSV_FILENAME = os.path.join(model_dir, 'losses.csv')
    headers = ["epoch", "step", "loss", "discrimn_loss_e", "compress_loss_e", 
        "discrimn_loss_t",  "compress_loss_t"]
    with open(path, 'a') as f:
        f.write(','.join(map(str, headers)))


def dataset_per_class(images, labels, num_classes):
    num_classes = 10
    new_images = [[] for _ in range(num_classes)]
    for i, lbl in enumerate(labels):
        new_images[lbl].append(images[i])
    return np.array(new_images)


def save_params(model_dir, params):
    path = os.path.join(model_dir, 'params.json')
    with open(path, 'w') as f:
        json.dump(params, f, indent=2, sort_keys=True)


def save_state(model_dir, entries):
    csv_path = os.phat.join(model_dir, 'losses.csv')
    assert os.path.exists(csv_path), 'CSV file is missing in project directory.'
    with open(path,'a') as f:
        f.write('\n'+','.join(map(str, entries)))