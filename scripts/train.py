"""
Train an image model.
The script take in input a configuration and a dataset name.
It will save all needed data to reproduce the experiment and to use the save model.
You can start Tensorboard to monitor the training.
"""
import argparse
import importlib
import json
import logging
import os
import multiprocessing

import numpy as np
import pandas as pd
from datetime import datetime
from collections import namedtuple
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from keras.metrics import categorical_crossentropy

from cdiscount.utils.TensorBoardCallBack import TensorBoardCallBack
from cdiscount.utils.image import ImageDataGenerator
from cdiscount.evaluation import metrics
from cdiscount.models.freeze import set_trainable_layers


TrainingState = namedtuple("TrainingState", ["initial_step", "initial_epoch", "need_compile"])


def get_callbacks(dst_dir):
    """
    Get training callbacks.
    :param dst_dir: Output path for callbacks which need to store files on disk
    :return: A list of keras.callbacks
    """
    return [
        TensorBoardCallBack(log_dir=dst_dir,
                            batch_freq=10),

        EarlyStopping(monitor='val_acc', min_delta=0.001,
                      patience=2, mode='max', verbose=1),

        ReduceLROnPlateau(monitor='val_acc', factor=0.1,
                          patience=1, verbose=1, mode='max',
                          epsilon=0.01),

        ModelCheckpoint('.'.join((dst_dir, "weights.{epoch:02d}-{val_acc:.2f}.hdf5")),
                        monitor='val_acc', mode='max', verbose=1)
    ]


def get_optimizer(conf_optimizer):
    """
    Get an optimizer defined by a dict.
    :param conf_optimizer: A dict, must contains keys type and params.
    """
    OptimizerClass = getattr(optimizers, conf_optimizer["type"])
    optimizer = OptimizerClass(**conf_optimizer["params"])
    return optimizer


def init_model(args, model_module, num_classes, model_params):
    """
    Load the model, and initial training state.
    If a checkpoint is provided, the model is loaded from it, and training will be resumed.
    Otherwise, the model is created and training will start at first step.
    :param args: Training command line args.
    :param model_module: Module containing the get_model function. Used only for a new model.
    :param num_classes: Number of output classes for the model. Used only for a new model.
    :return: model, initial training state
    """
    if os.path.exists(args.checkpoint):
        logging.info("Load model from checkpoint {}".format(args.checkpoint))
        model = load_model(args.checkpoint)
        initial_epoch = args.epoch
        initial_step = int(args.initial_step)
        need_compile = False
    else:
        logging.info("Get pretrained model")
        model = model_module.get_model(num_classes, **model_params)
        initial_step = 0
        initial_epoch = 0
        need_compile = True
    return model, TrainingState(initial_step, initial_epoch, need_compile)


def train_step(model, train_gen, val_gen, step, state, logdir):
    """
    Train one step.
    """
    logging.info("Start training step {}".format(step["name"]))
    optimizer = get_optimizer(step["optimizer"])

    if state.need_compile or state.initial_epoch == 0:
        set_trainable_layers(model, nb_layers=int(step["n_trainable_layers"]))
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy', metrics.top_k(3), metrics.top_k(5),
                               categorical_crossentropy])

    callbacks = get_callbacks(os.path.join(logdir, step["name"]))
    history = model.fit_generator(
        train_gen,
        steps_per_epoch=int(np.ceil(train_gen.samples / train_gen.batch_size)),
        epochs=step["n_epochs"],
        validation_data=val_gen,
        validation_steps=int(np.ceil(val_gen.samples / val_gen.batch_size)),
        callbacks=callbacks,
        max_queue_size=10,
        workers=1,
        initial_epoch=state.initial_epoch,
        verbose=1)
    logging.info("Finished step {}!".format(step['name']))


def save_category_mapping(train_gen, val_gen, log_dir):
    """
    Save the class id to category id mapping.
    :param train_gen: Iterator on train batches.
    :param val_gen: Iterator on val batches.
    :param log_dir: Path to save the file.
    :return:
    """
    train_class_id_to_category_id = {int(class_id): int(cat_id)
                                     for (cat_id, class_id)
                                     in train_gen.class_indices.items()}
    val_class_id_to_category_id = {int(class_id): int(cat_id)
                                   for (cat_id, class_id)
                                   in val_gen.class_indices.items()}
    if train_class_id_to_category_id != val_class_id_to_category_id:
        raise Exception("Error, the train and val mapping are different")

    mapping_filename = os.path.join(log_dir, "class_id_to_category_id.json")
    with open(mapping_filename, "w") as mfile:
        json.dump(train_class_id_to_category_id, mfile, indent=2)
    logging.info("Saved the mapping class id to category in {}"
                 .format(mapping_filename))


def train(learning_conf, args, log_dir):
    """
    Train a model on a given dataset.
    :param learning_conf: Learning configuration dict.
    :param args: Command line arguments.
    :param log_dir: Directory for all output files.
    """
    model_module = importlib.import_module("cdiscount.models." + learning_conf["model"])
    logging.info("Load dataset...")

    train_gen = get_image_iterator(args.train, args.img_dir,
                                   model_module.preprocess_input,
                                   learning_conf["generator"]["augmentation"],
                                   learning_conf["generator"]["flow"])
    val_gen = get_image_iterator(args.train, args.img_dir,
                                 model_module.preprocess_input,
                                 {},
                                 learning_conf["generator"]["flow_test"])
    save_category_mapping(train_gen, val_gen, log_dir)

    logging.info("Initialize model...")
    model, state = init_model(args, model_module, train_gen.num_class,
                              learning_conf.get("model_params", {}))

    for step in learning_conf["steps"][state.initial_step:]:
        train_step(model, train_gen, val_gen, step, state, log_dir)
        state = TrainingState(initial_step=state.initial_step + 1,
                              initial_epoch=0, need_compile=True)

    logging.info("Done!")


def create_timestamped_log_dir(dst_dir):
    now = datetime.now()
    time_dir = now.strftime("%Y-%m-%d.%H.%M.%S")
    timestamped_log_dir = os.path.join(dst_dir, time_dir)
    os.makedirs(timestamped_log_dir)
    return timestamped_log_dir


def get_image_iterator(dataset_fn, img_dir, preprocess, augmentation, flow):
    dataset = pd.read_feather(dataset_fn)
    pool = multiprocessing.Pool()
    imgen = ImageDataGenerator(preprocessing_function=preprocess,
                               pool=pool, **augmentation)
    iter_im = imgen.flow_from_dataset(dataset, img_dir, **flow)
    return iter_im


def load_configuration(conf_filename):
    with open(conf_filename, "r") as conf_file:
        conf = json.load(conf_file)
    return conf


def save_full_configuration(args, log_dir, conf):
    conf["train"] = args.train
    conf["val"] = args.val
    conf_filename = os.path.join(log_dir, "conf.json")
    with open(conf_filename, "w") as out_conf_file:
        json.dump(conf, out_conf_file, indent=2)
    logging.info("Saved the configuration in {}".format(conf_filename))


def setup_training(args):
    log_dir = create_timestamped_log_dir(args.logdir)
    conf = load_configuration(args.conf)

    save_full_configuration(args, log_dir, conf)
    return conf, log_dir


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description='Finetune Pre-trained model using Keras')
    parser.add_argument('--logdir', type=str,
                        help='Path to the directory containing logs and checkpoints.'
                             'The script will create a subdir with the current date.')
    parser.add_argument('--conf', type=str,
                        help="Path to the configuration file.")
    parser.add_argument('--imgdir', type=str,
                        help='Path to the directory containing images.')
    parser.add_argument('--train', type=str,
                        help="Path to the train dataset.")
    parser.add_argument('--val', type=str,
                        help="Path to the validation dataset.")
    parser.add_argument('--checkpoint', type=str, default="",
                        help="Checkpoint to use to resume training.")
    parser.add_argument('--epoch', type=int, default=0,
                        help="Resume training from this epoch.")
    parser.add_argument('--initial_step', type=int,
                        default="0",
                        help="Resume training from this step.")
    main_args = parser.parse_args()

    conf, log_dir = setup_training(main_args)

    train(conf, main_args, log_dir)
