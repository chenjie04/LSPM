import heapq
import math
import time
from functools import partial
from datetime import datetime
from collections import OrderedDict
from argparse import ArgumentParser

import random
import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import multiprocessing as mp

import utils
from CNCF import ContextCF
from dataset import CFTrainDataset, load_test_ratings, load_test_negs
from convert import (TEST_NEG_FILENAME, TEST_RATINGS_FILENAME,
                     TRAIN_RATINGS_FILENAME)

from mlperf_compliance import mlperf_log

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def parse_args():
    parser = ArgumentParser(description="Train a Nerual Collaborative"
                                        " Filtering model")
    parser.add_argument('--data', type=str, default='data/taobao',
                        help='path to test and training data files')
    parser.add_argument('-e', '--epochs', type=int, default=20,
                        help='number of epochs for training')
    parser.add_argument('-b', '--batch-size', type=int, default=2048,
                        help='number of examples for each iteration')
    parser.add_argument('-f', '--factors', type=int, default=64,
                        help='number of predictive factors')
    parser.add_argument('--layers', nargs='+', type=int,
                        default=[512,256,128,64],
                        help='size of hidden layers for MLP')
    parser.add_argument('-n', '--negative-samples', type=int, default=4,
                        help='number of negative examples per interaction')
    parser.add_argument('-l', '--learning-rate', type=float, default=0.001,
                        help='learning rate for optimizer')
    parser.add_argument('-k', '--topk', type=int, default=10,
                        help='rank for test examples to be considered a hit')
    parser.add_argument('--no-cuda', action='store_true',default=False,
                        help='use available GPUs')
    parser.add_argument('--seed', '-s', type=int,default=3,
                        help='manually set random seed for torch')
    parser.add_argument('--processes', '-p', type=int, default=1,
                        help='Number of processes for evaluating model')
    parser.add_argument('--workers', '-w', type=int, default=4,
                        help='Number of workers for training DataLoader')
    return parser.parse_args()


def predict(model, users, items, history, batch_size=1024, use_cuda=True):
    batches = [(users[i:i + batch_size], items[i:i + batch_size],history[i:i + batch_size])
               for i in range(0, len(users), batch_size)]
    preds = []
    for user, item, _history in batches:
        def proc(x):
            x = np.array(x,dtype=int)
            x = torch.from_numpy(x)
            if use_cuda:
                x = x.cuda(async=True)
            return torch.autograd.Variable(x)

        outp, _ = model(proc(user), proc(item), proc(_history), sigmoid=True)
        outp = outp.data.cpu().numpy()
        preds += list(outp.flatten())
    return preds


def _calculate_hit(ranked, test_item):
    return int(test_item in ranked)


def _calculate_ndcg(ranked, test_item):
    for i, item in enumerate(ranked):
        if item == test_item:
            return math.log(2) / math.log(i + 2)
    return 0.


def eval_one(rating, items, model, K, use_cuda=True):

    user = rating[0]
    test_item = rating[1]
    items.append(test_item)
    users = [user] * len(items)
    history = []
    _history = rating[2:11]
    for i in range(len(items)):
        history.append(_history)

    assert len(users) == len(items) == len(history)

    predictions = predict(model, users, items, history, use_cuda=use_cuda)

    map_item_score = {item: pred for item, pred in zip(items, predictions)}
    ranked = heapq.nlargest(K, map_item_score, key=map_item_score.get)

    hit = _calculate_hit(ranked, test_item)
    ndcg = _calculate_ndcg(ranked, test_item)
    return hit, ndcg, len(predictions)


def val_epoch(model, ratings, negs, K, use_cuda=True, output=None, epoch=None,
              processes=1):
    if epoch is None:
        print("Initial evaluation")
    else:
        print("Epoch {} evaluation".format(epoch))

    mlperf_log.ncf_print(key=mlperf_log.EVAL_START, value=epoch)
    start = datetime.now()
    model.eval()
    if processes > 1:
        context = mp.get_context('spawn')
        _eval_one = partial(eval_one, model=model, K=K, use_cuda=use_cuda)
        with context.Pool(processes=processes) as workers:
            hits_ndcg_numpred = workers.starmap(_eval_one, zip(ratings, negs))
        hits, ndcgs, num_preds = zip(*hits_ndcg_numpred)
    else:
        hits, ndcgs, num_preds = [], [], []
        for rating, items in zip(ratings, negs):
            hit, ndcg, num_pred = eval_one(rating, items, model, K, use_cuda=use_cuda)
            hits.append(hit)
            ndcgs.append(ndcg)
            num_preds.append(num_pred)

    hits = np.array(hits, dtype=np.float32)
    ndcgs = np.array(ndcgs, dtype=np.float32)

    assert len(set(num_preds)) == 1
    num_neg = num_preds[0] - 1  # one true positive, many negatives
    mlperf_log.ncf_print(key=mlperf_log.EVAL_SIZE, value={"epoch": epoch, "value": len(hits) * (1 + num_neg)})
    mlperf_log.ncf_print(key=mlperf_log.EVAL_HP_NUM_USERS, value=len(hits))
    mlperf_log.ncf_print(key=mlperf_log.EVAL_HP_NUM_NEG, value=num_neg)

    end = datetime.now()
    if output is not None:
        result = OrderedDict()
        result['timestamp'] = datetime.now()
        result['duration'] = end - start
        result['epoch'] = epoch
        result['K'] = K
        result['hit_rate'] = np.mean(hits)
        result['NDCG'] = np.mean(ndcgs)
        utils.save_result(result, output)

    return hits, ndcgs


def main():
    # Note: The run start is in convert.py

    args = parse_args()
    if args.seed is not None:
        print("Using seed = {}".format(args.seed))
        torch.manual_seed(args.seed)
        np.random.seed(seed=args.seed)

    # Save configuration to file
    config = {k: v for k, v in args.__dict__.items()}
    config['timestamp'] = "{:.0f}".format(datetime.utcnow().timestamp())
    config['local_timestamp'] = str(datetime.now())
    run_dir = "./run/ancf_taobao/{}".format(config['timestamp'])
    print("Saving config and results to {}".format(run_dir))
    if not os.path.exists(run_dir) and run_dir != '':
        os.makedirs(run_dir)
    utils.save_config(config, run_dir)

    # Check that GPUs are actually available
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    if use_cuda:
        print("Using cuda ...")
    else:
        print("Using CPU ...")

    t1 = time.time()
    # Load Data
    print('Loading data')
    print(os.path.join(args.data, TRAIN_RATINGS_FILENAME))
    train_dataset = CFTrainDataset(
        os.path.join(args.data, TRAIN_RATINGS_FILENAME), args.negative_samples)

    mlperf_log.ncf_print(key=mlperf_log.INPUT_BATCH_SIZE, value=args.batch_size)
    mlperf_log.ncf_print(key=mlperf_log.INPUT_ORDER)  # set shuffle=True in DataLoader
    train_dataloader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)
    test_ratings = load_test_ratings(os.path.join(args.data, TEST_RATINGS_FILENAME))  # noqa: E501
    test_negs = load_test_negs(os.path.join(args.data, TEST_NEG_FILENAME))
    nb_users, nb_items = train_dataset.nb_users, train_dataset.nb_items
    print('Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d'
          % (time.time()-t1, nb_users, nb_items, train_dataset.mat.nnz,
             len(test_ratings)))

    # Create model
    model = ContextCF(nb_users=nb_users, nb_items=nb_items,
                      embed_dim=64,mlp_layer_sizes=args.layers,mlp_layer_regs=[0. for i in args.layers])
    print(model)
    print("{} parameters".format(utils.count_parameters(model)))

    # Save model text description
    with open(os.path.join(run_dir, 'model.txt'), 'w') as file:
        file.write(str(model))

    # Add optimizer and loss to graph
    mlperf_log.ncf_print(key=mlperf_log.OPT_LR, value=args.learning_rate)
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    mlperf_log.ncf_print(key=mlperf_log.OPT_NAME, value="Adam")
    mlperf_log.ncf_print(key=mlperf_log.OPT_HP_ADAM_BETA1, value=beta1)
    mlperf_log.ncf_print(key=mlperf_log.OPT_HP_ADAM_BETA2, value=beta2)
    mlperf_log.ncf_print(key=mlperf_log.OPT_HP_ADAM_EPSILON, value=epsilon)
    optimizer = torch.optim.Adam(model.parameters(), betas=(beta1, beta2),
                                 lr=args.learning_rate, eps=epsilon)

    mlperf_log.ncf_print(key=mlperf_log.MODEL_HP_LOSS_FN, value=mlperf_log.BCE)
    criterion = nn.BCEWithLogitsLoss()


    if use_cuda:
        # Move model and loss to GPU
        model = model.cuda()
        criterion = criterion.cuda()

    # Create files for tracking training
    valid_results_file = os.path.join(run_dir, 'ancf_taobao_valid_results.csv')

    # ############################################################################################################
    #
    # #Attention visualization
    # instances = random.sample(test_ratings,10)
    # df_instances = pd.DataFrame(instances)
    # user = df_instances.iloc[:,[0]]
    # item = df_instances.iloc[:,[1]]
    # _history = df_instances.iloc[:,2:]
    #
    #
    #
    # def proc(x):
    #     x = np.array(x, dtype=int)
    #     x = torch.from_numpy(x)
    #     if use_cuda:
    #         x = x.contiguous().cuda(async=True).squeeze()
    #     return torch.autograd.Variable(x)
    #
    # _, weights = model(proc(user), proc(item), proc(_history), sigmoid=True)
    #
    # weights = weights.squeeze().detach()
    #
    # sns.set()
    # heatmap = sns.heatmap(weights,annot=True, fmt="f")
    # plt.show()
    #
    #
    #
    #
    # ############################################################################################################




    # Calculate initial Hit Ratio and NDCG

    hits, ndcgs = val_epoch(model, test_ratings, test_negs, args.topk,
                            use_cuda=use_cuda, processes=args.processes)
    print('Initial HR@{K} = {hit_rate:.4f}, NDCG@{K} = {ndcg:.4f}'
          .format(K=args.topk, hit_rate=np.mean(hits), ndcg=np.mean(ndcgs)))

    best_hit, best_ndcg = np.mean(hits), np.mean(ndcgs)
    mlperf_log.ncf_print(key=mlperf_log.TRAIN_LOOP)
    for epoch in range(args.epochs):
        mlperf_log.ncf_print(key=mlperf_log.TRAIN_EPOCH, value=epoch)
        model.train()
        losses = utils.AverageMeter()

        mlperf_log.ncf_print(key=mlperf_log.INPUT_HP_NUM_NEG, value=train_dataset.nb_neg)
        mlperf_log.ncf_print(key=mlperf_log.INPUT_STEP_TRAIN_NEG_GEN)
        begin = time.time()
        loader = tqdm.tqdm(train_dataloader)
        for batch_index, (user, item, history, label) in enumerate(loader):
            user = torch.autograd.Variable(user, requires_grad=False)
            item = torch.autograd.Variable(item, requires_grad=False)
            history = torch.autograd.Variable(history, requires_grad=False)
            label = torch.autograd.Variable(label, requires_grad=False)
            if use_cuda:
                user = user.cuda(async=True)
                item = item.cuda(async=True)
                history = history.cuda(async=True)
                label = label.cuda(async=True)

            outputs, _ = model(user, item,history)
            loss = criterion(outputs, label)
            losses.update(loss.data.item(), user.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Save stats to file
            description = ('Epoch {} Loss {loss.val:.4f} ({loss.avg:.4f})'
                           .format(epoch, loss=losses))
            loader.set_description(description)

        train_time = time.time() - begin
        begin = time.time()
        hits, ndcgs = val_epoch(model, test_ratings, test_negs, args.topk,
                                use_cuda=use_cuda, output=valid_results_file,
                                epoch=epoch, processes=args.processes)
        mlperf_log.ncf_print(key=mlperf_log.EVAL_ACCURACY, value={"epoch": epoch, "value": float(np.mean(hits))})
        mlperf_log.ncf_print(key=mlperf_log.EVAL_STOP)
        val_time = time.time() - begin
        print('Epoch {epoch}: HR@{K} = {hit_rate:.4f}, NDCG@{K} = {ndcg:.4f},'
              ' train_time = {train_time:.2f}, val_time = {val_time:.2f}'
              .format(epoch=epoch, K=args.topk, hit_rate=np.mean(hits),
                      ndcg=np.mean(ndcgs), train_time=train_time,
                      val_time=val_time))
        if np.mean(hits) >= best_hit:
            best_hit = np.mean(hits)
        if np.mean(ndcgs) >= best_ndcg:
            best_ndcg = np.mean(ndcgs)

    print("Best hit: ",best_hit)
    print("Best_ndcg: ", best_ndcg)

    mlperf_log.ncf_print(key=mlperf_log.RUN_STOP)
    mlperf_log.ncf_print(key=mlperf_log.RUN_FINAL)

    ############################################################################################################

    #Attention visualization
    instances = random.sample(test_ratings,10)
    df_instances = pd.DataFrame(instances)
    user = df_instances.iloc[:,[0]]
    item = df_instances.iloc[:,[1]]
    _history = df_instances.iloc[:,2:]



    def proc(x):
        x = np.array(x, dtype=int)
        x = torch.from_numpy(x)
        if use_cuda:
            x = x.contiguous().cuda(async=True).squeeze()
        return torch.autograd.Variable(x)

    _, weights = model(proc(user), proc(item), proc(_history), sigmoid=True)

    weights = weights.squeeze().detach()
    # weights = np.round(weights,decimals=1)
    print(weights)

    sns.set()
    heatmap = sns.heatmap(weights,annot=True,cmap='Wistia',fmt='f')
    plt.show()




    ############################################################################################################






if __name__ == '__main__':
    main()