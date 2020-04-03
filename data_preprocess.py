import os
from argparse import ArgumentParser
from collections import defaultdict
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from tqdm import tqdm
import random

from load import implicit_load

from mlperf_compliance import mlperf_log

MIN_RATINGS = 20


USER_COLUMN = 'user_id'
ITEM_COLUMN = 'item_id'


TRAIN_RATINGS_FILENAME = 'train_ratings.csv'
TEST_RATINGS_FILENAME = 'test_ratings.csv'
TEST_NEG_FILENAME = 'test_negative.csv'

PATH = 'data/ml-1m'
OUTPUT = 'data/ml-1m'
NEGATIVES = 99
HISTORY_SIZE = 9
RANDOM_SEED = 0

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--file', type=str, default=(os.path.join(PATH,'ratings.csv')),
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
    df = implicit_load(args.file, sort=False)
    # df = pd.read_csv(args.file,sep=',',header=None)
    # df.columns = ['user_id','item_id','rating','timestamp']
    # print(df)
    # df = df.sample(frac=0.1) # we only use 10% data

    print("Filtering out users with less than {} ratings".format(MIN_RATINGS))
    grouped = df.groupby(USER_COLUMN)
    mlperf_log.ncf_print(key=mlperf_log.PREPROC_HP_MIN_RATINGS, value=MIN_RATINGS)
    df = grouped.filter(lambda x: len(x) >= MIN_RATINGS)

    print("Mapping original user and item IDs to new sequential IDs")
    original_users = df[USER_COLUMN].unique()
    original_items = df[ITEM_COLUMN].unique()

    nb_users = len(original_users)
    nb_items = len(original_items)

    user_map = {user: index for index, user in enumerate(original_users)}
    item_map = {item: index for index, item in enumerate(original_items)}

    df[USER_COLUMN] = df[USER_COLUMN].apply(lambda user: user_map[user])
    df[ITEM_COLUMN] = df[ITEM_COLUMN].apply(lambda item: item_map[item])

    # print(df)


    assert df[USER_COLUMN].max() == len(original_users) - 1
    assert df[ITEM_COLUMN].max() == len(original_items) - 1

    print("Creating list of items for each user")
    # Need to sort before popping to get last item
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.sort_values(by='timestamp', inplace=True)
    all_ratings = set(zip(df[USER_COLUMN], df[ITEM_COLUMN]))
    user_to_items = defaultdict(list)
    for row in tqdm(df.itertuples(), desc='Ratings', total=len(df)):
        user_to_items[getattr(row, USER_COLUMN)].append(getattr(row, ITEM_COLUMN))  # noqa: E501

    print(len(user_to_items[0]))
    print(user_to_items[0])
    print(user_to_items[0][-args.history_size:])


    train_ratings = []
    test_ratings = []
    test_negs = []
    all_items = set(range(len(original_items)))

    print("Generating {} negative samples for each user and creating training set"
          .format(args.negatives))
    mlperf_log.ncf_print(key=mlperf_log.PREPROC_HP_NUM_EVAL, value=args.negatives)

    # The default of np.random.choice is replace=True
    mlperf_log.ncf_print(key=mlperf_log.PREPROC_HP_SAMPLE_EVAL_REPLACEMENT, value=True)

    #===========================================================================
    #== First random operation triggers the clock start. =======================
    #===========================================================================
    mlperf_log.ncf_print(key=mlperf_log.RUN_START)
    mlperf_log.ncf_print(key=mlperf_log.INPUT_STEP_EVAL_NEG_GEN)

    for user in tqdm(range(len(original_users)), desc='Users', total=len(original_users)):  # noqa: E501
        all_negs = all_items - set(user_to_items[user])
        all_negs = sorted(list(all_negs))  # determinism
        negs = random.sample(all_negs,args.negatives)

        test_item = user_to_items[user].pop()

        all_ratings.remove((user, test_item))

        tmp = []
        tmp.extend([user, test_item])
        tmp.extend(negs)
        test_negs.append(list(tmp))

        tmp = []
        tmp.extend([user, test_item])
        tmp.extend(user_to_items[user][-args.history_size:])
        test_ratings.append(list(tmp))

        while len(user_to_items[user]) > args.history_size:
            tgItem = user_to_items[user].pop()
            tmp = []
            tmp.extend([user,tgItem])
            tmp.extend(user_to_items[user][-args.history_size:])
            train_ratings.append(list(tmp))




    print("\nSaving train and test CSV files to {}".format(args.output))



    df_train_ratings = pd.DataFrame(list(train_ratings))

    print('Saving data description ...')
    f_writer = open(os.path.join(OUTPUT,'data_summary.txt'), 'w')
    f_writer.write('users = ' + str(nb_users) + ', items = ' + str(nb_items)+', history_size = '+str(HISTORY_SIZE) + ', train_entries = ' + str(len(df_train_ratings)))

    df_train_ratings['fake_rating'] = 1
    df_train_ratings.to_csv(os.path.join(args.output, TRAIN_RATINGS_FILENAME),
                            index=False, header=False, sep='\t')

    mlperf_log.ncf_print(key=mlperf_log.INPUT_SIZE, value=len(df_train_ratings))

    df_test_ratings = pd.DataFrame(test_ratings)
    df_test_ratings['fake_rating'] = 1
    df_test_ratings.to_csv(os.path.join(args.output, TEST_RATINGS_FILENAME),
                           index=False, header=False, sep='\t')

    df_test_negs = pd.DataFrame(test_negs)
    df_test_negs.to_csv(os.path.join(args.output, TEST_NEG_FILENAME),
                        index=False, header=False, sep='\t')




if __name__ == '__main__':
    main()
