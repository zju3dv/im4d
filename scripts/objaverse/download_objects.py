import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
sys.path.append('.')

import objaverse
import multiprocessing
import random
import trimesh


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--download_all', type=bool, default=False)
    args = parser.parse_args()
    return args

def main(args):
    uids = objaverse.load_uids()
    annotations = objaverse.load_annotations(uids[:10])
    processes = multiprocessing.cpu_count()
    random.seed(42)
    uids = objaverse.load_uids()
    random_object_uids = random.sample(uids, 10)
    objects = objaverse.load_objects(
        uids=random_object_uids,
        download_processes=processes
    )
    # objects = objaverse.load_objects(uids=random_object_uids)
    # trimesh.load(list(objects.values())[0]).show() 

if __name__ == '__main__':
    args = parse_args()
    main(args)
    # scripts usage:
    # python scripts/objaverse/download_objects.py
