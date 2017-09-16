"""
Convert the bson files to jpg images.
It will be easier to use existing Keras code with images.
+ bson may be slow to do random access in parallel.
"""
import os
import io
import bson
import logging
import argparse
import progressbar
import pandas as pd
import multiprocessing as mp
from PIL import Image
from functools import partial

IMAGES_DIR = "images_raw"


def create_directory_structure(data_dir, sub_dir):
    df_names = pd.read_csv(os.path.join(data_dir, "category_names.csv"))
    category_ids = set(df_names.category_id)
    output_dir = os.path.join(data_dir, sub_dir, IMAGES_DIR)
    os.makedirs(output_dir)
    for cid in category_ids:
        cdir = os.path.join(output_dir, str(cid))
        os.makedirs(cdir)


def process(item, output_dir):
    size = 0
    category_id = item['category_id']
    product_id = item['_id']
    images = item['imgs']
    for image_id, pic in enumerate(images):
        size += len(pic['picture'])
        image = Image.open(io.BytesIO(pic['picture']))
        image_name = "{}_{}.jpg".format(product_id, image_id)
        out_filename = os.path.join(output_dir, str(category_id), image_name)
        image.save(out_filename)

    return product_id, category_id, len(images), size


def build_metadata(pid_to_cid, pid_to_count):
    all_images = []
    for pid in pid_to_cid.keys():
        cid = pid_to_cid[pid]
        for iid in range(pid_to_count[pid]):
            all_images.append((pid, iid, cid))
    return pd.DataFrame(all_images,
                        columns=["product_id", "image_id", "category_id"])


def save_to_jpg(data_dir, sub_dir, bson_filename):
    pid_to_cid = dict()
    pid_to_count = dict()
    total_size_bytes = 0
    total_images = 0
    output_dir = os.path.join(data_dir, sub_dir, IMAGES_DIR)
    process_spec = partial(process, output_dir=output_dir)
    data = bson.decode_file_iter(open(os.path.join(data_dir, bson_filename), 'rb'))
    with mp.Pool() as pool, \
            progressbar.ProgressBar(0, progressbar.UnknownLength) as bar:
        for i, r in enumerate(pool.imap_unordered(process_spec, data)):
            product_id, category_id, n_images, size = r
            pid_to_cid[product_id] = category_id
            pid_to_count[product_id] = n_images
            total_size_bytes += size
            total_images += n_images
            bar.update(total_images)

    logging.info("Save {n} images (total size {s:.2f} GB"
                 .format(n=total_images, s=total_size_bytes))

    df = build_metadata(pid_to_cid, pid_to_count)
    df.to_feather(os.path.join(data_dir, sub_dir, data.feather))


def main(args):
    create_directory_structure(args.data_dir, "train")
    create_directory_structure(args.data_dir, "test")

    save_to_jpg(args.data_dir, "train", "train.bson")
    save_to_jpg(args.data_dir, "test", "test.bson")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description='Convert the bson files to jpg images and feather dataframe')
    parser.add_argument('--data_dir', type=str,
                        help="Path to the data.")
    main_args = parser.parse_args()
    main(main_args)
