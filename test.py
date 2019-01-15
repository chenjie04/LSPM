import os
from argparse import ArgumentParser
from collections import defaultdict
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from tqdm import tqdm
import random

from mlperf_compliance import mlperf_log


MIN_RATINGS = 20


USER_COLUMN = 'user_id'
ITEM_COLUMN = 'item_id'


TRAIN_RATINGS_FILENAME = 'train_ratings.csv'
TEST_RATINGS_FILENAME = 'test_ratings.csv'
TEST_NEG_FILENAME = 'test_negative.csv'

PATH = 'data/taobao'
OUTPUT = 'data/taobao'
NEGATIVES = 99
HISTORY_SIZE = 9
RANDOM_SEED = 0

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--file', type=str, default=(os.path.join(PATH,'UserBehavior.csv')),
                        help='Path to reviews CSV file from MovieLens')
    parser.add_argument('--output', type=str, default=OUTPUT,
                        help='Output directory for train and test CSV files')
    parser.add_argument('-n', '--negatives', type=int, default=NEGATIVES,
                        help='Number of negative samples for each positive'
                             'test example')
    parser.add_argument('--history_size',type=int,default=HISTORY_SIZE,
                        help='The size of history')
    parser.add_argument('-s', '--seed', type=int, default=RANDOM_SEED,
                        help='Random seed to reproduce same negative samples')
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    print("Loading raw data from {}".format(args.file))
    df = pd.read_csv(args.file,sep=',',header=None)
    df.columns = ['user_id','item_id','category_id','behavior','timestamp']
    print(df)
    df = df.sample(frac=0.1) # we only use 10% data

    print("Filtering out users with less than {} ratings".format(MIN_RATINGS))
    grouped = df.groupby(USER_COLUMN)
    mlperf_log.ncf_print(key=mlperf_log.PREPROC_HP_MIN_RATINGS, value=MIN_RATINGS)
    df = grouped.filter(lambda x: len(x) >= MIN_RATINGS)

    print("Mapping original user and item IDs to new sequential IDs")
    original_users = df[USER_COLUMN].unique()
    original_items = df[ITEM_COLUMN].unique()

    nb_users = len(original_users)
    nb_items = len(original_items)
    interactions = len(df)

    user_map = {user: index for index, user in enumerate(original_users)}
    item_map = {item: index for index, item in enumerate(original_items)}

    df[USER_COLUMN] = df[USER_COLUMN].apply(lambda user: user_map[user])
    df[ITEM_COLUMN] = df[ITEM_COLUMN].apply(lambda item: item_map[item])

    print(df)


    assert df[USER_COLUMN].max() == len(original_users) - 1
    assert df[ITEM_COLUMN].max() == len(original_items) - 1

    print(interactions)






if __name__ == '__main__':
    main()
