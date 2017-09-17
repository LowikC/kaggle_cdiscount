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


def create_item_symlink(x, img_dir, out_dir):
    pid, iid, cid = x["product_id"], x["image_id"], x["category_id"]
    cid_dir = os.path.join(img_dir, str(cid))
    if not os.path.isdir(cid_dir):
        os.makedirs(cid_dir)
    img_name = "{}_{}.jpg".format(pid, iid)
    src_name = os.path.join(img_dir, str(cid), img_name)
    dst_name = os.path.join(out_dir, str(cid), img_name)
    os.symlink(src_name, dst_name)


def create_symlinks(img_dir, out_dir, train, val):
    out_train_dir = os.path.join(out_dir, "train")
    out_val_dir = os.path.join(out_dir, "val")
    os.makedirs(out_train_dir, exist_ok=True)
    os.makedirs(out_val_dir, exist_ok=True)
    train.apply(create_item_symlink, axis=1, args=(img_dir, out_train_dir))
    val.apply(create_item_symlink, axis=1, args=(img_dir, out_val_dir))
    logging.info("Symlinks created in {} (train) and {} (val)"
                 .format(out_train_dir, out_val_dir))


def main(args):
    dataset = pd.read_feather(args.dataset)
    sampled, sampling_factor_per_class =\
        sample(dataset, min_samples_per_class=args.min_samples,
               max_samples_per_class=args.max_samples, seed=args.seed)
    train, val = stratified_split(sampled, ratio=args.train_ratio)

    train_filename = os.path.join(args.out_dir, "train_{}_{}_{:.2f}_{}.feather"
                                  .format(args.min_samples, args.max_samples,
                                          args.train_ratio, args.seed))
    val_filename = os.path.join(args.out_dir, "val_{}_{}_{:.2f}_{}.feather"
                                .format(args.min_samples, args.max_samples,
                                        1 - args.train_ratio, args.seed))

    os.makedirs(args.out_dir, exist_ok=True)
    train.to_feather(train_filename)
    val.to_feather(val_filename)

    if args.symlinks:
        create_symlinks(args.img_dir, args.out_dir, train, val)


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
    parser.add_argument('--symlinks', action='store_true',
                        help="Create symlink to all images in train and val.")
    parser.add_argument('--seed', type=int,
                        default=42,
                        help="Seed for sampling")
    main_args = parser.parse_args()
    main(main_args)
