import logging
import numpy as np


def sample(dataset, max_samples_per_class=250, min_samples_per_class=10,
           class_identifier="category_id", seed=None):
    """
    Sample the dataset to keep at most max_samples_per_class samples in each class.
    We discard class with less than min_samples_per_class items.
    :param dataset: Dataframe, one sample per row.
    :param max_samples_per_class: Maximum number of samples in one class.
    :param min_samples_per_class: Minimum number of samples in one class.
    :param class_identifier: Name of the column containing the class of each sample.
    :param seed: Seed for random.
    :return: Sampled dataset and the sampling ratio of each class (as a dict class -> ratio).
    """
    if seed is not None:
        np.random.seed(seed)
    logging.info("Original dataset size: {}".format(len(dataset)))
    classes = np.array(dataset[class_identifier])
    class_ids, count_per_class = np.unique(classes, return_counts=True)
    selected_samples_idx = np.empty(classes.shape, dtype=np.int64)

    n_samples_processed = 0
    sampling_factor_per_class = dict()
    for i in range(class_ids.shape[0]):
        class_id = class_ids[i]
        n_samples_in_class = count_per_class[i]
        samples_idx = np.where(classes == class_id)[0]
        if n_samples_in_class > max_samples_per_class:
            random_samples_idx = np.random.choice(samples_idx, size=(max_samples_per_class, ), replace=False)
            selected_samples_idx[n_samples_processed: n_samples_processed + max_samples_per_class] = random_samples_idx
            n_samples_processed += max_samples_per_class
            sampling_factor_per_class[class_id] = max_samples_per_class/n_samples_in_class
        elif n_samples_in_class < min_samples_per_class:
            sampling_factor_per_class[class_id] = 0.0
        else:
            selected_samples_idx[n_samples_processed: n_samples_processed + n_samples_in_class] = samples_idx
            n_samples_processed += n_samples_in_class
            sampling_factor_per_class[class_id] = 1.0
    selected_samples_idx = selected_samples_idx[:n_samples_processed]

    sampled_dataset = dataset.iloc[np.sort(selected_samples_idx)]
    sampled_dataset.reset_index(drop=True, inplace=True)
    logging.info("Sampled dataset size: {}".format(len(sampled_dataset)))
    return sampled_dataset, sampling_factor_per_class


def stratified_split(dataset, ratio=0.70, class_identifier="category_id", seed=None):
    """
    Stratified split of the dataset, in two parts.
    :param dataset: Dataframe, one sample per row.
    :param ratio: Percentage of samples in the first split.
    :param class_identifier: Name of the column containing the class of each sample.
    :param seed: Seed for random.
    :return: train_dataset, val_dataset
    """
    if seed is not None:
        np.random.seed(seed)
    classes = np.array(dataset[class_identifier])
    class_ids, count_per_class = np.unique(classes, return_counts=True)
    train_samples_idx = np.empty(classes.shape, dtype=np.int64)
    val_samples_idx = np.empty(classes.shape, dtype=np.int64)

    n_train_samples = 0
    n_val_samples = 0
    for i in range(class_ids.shape[0]):
        class_id = class_ids[i]
        n_samples_in_class = count_per_class[i]

        n_train_in_class = max(1, int(round(ratio * n_samples_in_class)))
        n_val_in_class = n_samples_in_class - n_train_in_class

        samples_idx = np.where(classes == class_id)[0]
        perm_samples_idx = np.random.permutation(samples_idx)

        train_samples_idx[n_train_samples: n_train_samples + n_train_in_class] =\
            perm_samples_idx[:n_train_in_class]
        n_train_samples += n_train_in_class

        val_samples_idx[n_val_samples: n_val_samples + n_val_in_class] = \
            perm_samples_idx[n_train_in_class:]
        n_val_samples += n_val_in_class

    train_samples_idx = train_samples_idx[:n_train_samples]
    val_samples_idx = val_samples_idx[:n_val_samples]

    train_dataset = dataset.iloc[np.sort(train_samples_idx)]
    train_dataset.reset_index(drop=True, inplace=True)

    val_dataset = dataset.iloc[np.sort(val_samples_idx)]
    val_dataset.reset_index(drop=True, inplace=True)

    logging.info("Sampled datasets sizes: {} (train) and {} (val) ".format(len(train_dataset), len(val_dataset)))
    return train_dataset, val_dataset
