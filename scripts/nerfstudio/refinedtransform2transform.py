import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
sys.path.append('.')

from lib.utils.data_utils import read_camera
import json
from copy import deepcopy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()
    return args

def main(args):
    # extri_path = join(args.input, 'extri.yml')
    # intri_path = join(args.input, 'intri.yml')
    # cams = read_camera(intri_path, extri_path)
    transforms = json.load(open(join(args.input, 'transforms.json'), 'r'))
    transforms_train = json.load(open(join(args.input, 'transforms_train.json'), 'r'))
    
    refined_transforms = deepcopy(transforms)
    
    file_paths = [item['file_path'] for item in transforms_train]
    
    def get_index(inference_file_path):
        for file_path in file_paths:
            if inference_file_path in file_path:
                return file_paths.index(file_path)
    
    for frame in refined_transforms['frames']:
        file_path = frame['file_path']
        idx = get_index(file_path)
        transform_matrix = np.eye(4)
        transform_matrix[:3, :4] = np.array(transforms_train[idx]['transform'])
        frame['transform_matrix'] = transform_matrix.tolist()
        
    with open(join(f'{args.input}', 'refined_transforms.json'), 'w', encoding='utf-8') as f:
        json.dump(refined_transforms, f, indent=4)
        
if __name__ == '__main__':
    args = parse_args()
    main(args)

