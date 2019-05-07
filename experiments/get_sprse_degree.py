import os
import sys
import numpy as np


def read_from_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    def convert_to_num_arr(line):
        return [float(x) for x in line.strip().split()]

    data = [convert_to_num_arr(l) for l in lines]
    data = np.array(data)
    return data


if __name__ == "__main__":
    dir_path = sys.argv[1]
    if not os.path.exists(dir_path):
        raise ValueError('cannot find dir: {}'.format(dir_path))

    files = os.listdir(dir_path)
    files = [x for x in files if x.find('weight') != -1 and x.endswith('txt')]

    print('=' * 30)
    print('weight files:')
    print('\t'.join(files))
    print('\n')

    weights = []
    for f in files:
        print('processing ifle: {}'.format(f))
        f = os.path.join(dir_path, f)
        data = read_from_file(f)
        print('weight size: {}'.format(data.shape))
        weights.append(data.reshape(-1))

    weights = np.concatenate(weights).reshape(-1)

    non_zero_cnt = np.count_nonzero(weights)
    total_cnt = weights.size

    print('non zero count: {}\ntotal count: {}'.format(non_zero_cnt, total_cnt))
    print('sparse rate: {}'.format(1 - non_zero_cnt * 1.0 / total_cnt))
