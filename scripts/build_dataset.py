"""
Build a train/val split of the dataset.
We also threshold categories with too many items, and discard categories with 
too few items.
"""
import os
import logging
import argparse
import pandas as pd
from cdiscount.utils.sampling import stratified_split, sample


def img_name(x):
    cid, pid, iid = x["category_id"], x["product_id"], x["image_id"]
    im_name = "{}_{}.jpg".format(pid, iid)
    return os.path.join(str(cid), im_name)


def add_path(df):
    df["path"] = df.apply(img_name, axis=1)


def main(args):
    dataset = pd.read_feather(args.dataset)
    sampled, sampling_factor_per_class =\
        sample(dataset, min_samples_per_class=args.min_samples,
               max_samples_per_class=args.max_samples, seed=args.seed)
    train, val = stratified_split(sampled, ratio=args.train_ratio)
    add_path(train)
    add_path(val)
    train_filename = os.path.join(args.out_dir, "train_{}_{}_{:.2f}_{}.feather"
                                  .format(args.min_samples, args.max_samples,
                                          args.train_ratio, args.seed))
    val_filename = os.path.join(args.out_dir, "val_{}_{}_{:.2f}_{}.feather"
                                .format(args.min_samples, args.max_samples,
                                        1 - args.train_ratio, args.seed))

    os.makedirs(args.out_dir, exist_ok=True)
    train.to_feather(train_filename)
    val.to_feather(val_filename)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description='Create train and val datasets.')
    parser.add_argument('--dataset', type=str,
                        help="Path to the dataset.")
    parser.add_argument('--img_dir', type=str,
                        help="Path to the raw images directory.")
    parser.add_argument('--out_dir', type=str,
                        help="Path to the output dir.")
    parser.add_argument('--min_samples', type=int,
                        default=10,
                        help="Minimum number of samples in a class."
                             " If a class has less samples, it is discarded.")
    parser.add_argument('--max_samples', type=int,
                        default=250,
                        help="Maximum number of samples in a class. "
                             "If a class has more samples, "
                             "it will be uniformely subsampled.")
    parser.add_argument('--train_ratio', type=float,
                        default=0.7,
                        help="Proportion of the kept samples in the train set.")
    parser.add_argument('--seed', type=int,
                        default=42,
                        help="Seed for sampling")
    main_args = parser.parse_args()
    main(main_args)
