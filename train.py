from __future__ import print_function

import numpy as np
import torch
from torch import nn
from model import highwayNet
from utils import ngsimDataset, maskedMSE
from torch.utils.data import DataLoader
import time
import math
import datetime
import warnings
from joblib import Parallel, delayed


def item_dist(item):
    # for j in range(arr.shape[3]):
    #     grap = arr[i, :, :, j]
    item = item[:, :, 0]  # 提取3*13的矩阵
    graph = np.zeros((3, 13))
    for m in range(3):
        for n in range(13):
            if m == 1 and n == 6:
                graph[m, n] = 2
            else:
                if item[m, n] == 1:
                    x = m - 1
                    y = n - 6
                    z = np.sqrt(x ** 2 + y ** 2)
                    w = 1 / z
                    # w = np.around(w, 3)
                    graph[m, n] = w
    return graph


def c_dist(arr):
    result = Parallel(n_jobs=2)(delayed(item_dist)(i) for i in arr)  # 并行代码
    graphs = np.array(result)
    return graphs


if __name__ == '__main__':
    args = {}
    args['use_cuda'] = True
    args['encoder_size'] = 64  # lstm encoder hidden state size, adjustable
    args['decoder_size'] = 128  # lstm decoder  hidden state size, adjustable
    args['in_length'] = 16
    args['out_length'] = 5
    args['grid_size'] = (13, 3)

    # init self- multi-attention
    args['dim_in'] = 64
    args['dim_k'] = 64
    args['dim_v'] = 64
    args['num_heads'] = 4

    args['input_embedding_size'] = 32  # input dimension for lstm encoder, adjustable

    args['train_flag'] = True

    start_time = datetime.datetime.now()

    # Initialize network
    net = highwayNet(args)
    if args['use_cuda']:
        net = net.cuda()

    ## Initialize optimizer
    trainEpochs = 15
    optimizer = torch.optim.Adam(net.parameters())  
    batch_size = 128
    crossEnt = torch.nn.BCELoss()  # binary cross entropy

    ## Initialize data loaders
    trSet = ngsimDataset('../data/trajectory/TrainSet.mat')
    valSet = ngsimDataset('../data/trajectory/ValSet.mat')
    trDataloader = DataLoader(trSet, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=trSet.collate_fn,
                              drop_last=True)
    valDataloader = DataLoader(valSet, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=valSet.collate_fn,
                               drop_last=True)

    ## Variables holding train and validation loss values:
    train_loss = []
    val_loss = []
    prev_val_loss = math.inf

    for epoch_num in range(trainEpochs):

        ## Train:_________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
        net.train_flag = True

        # Variables to track training performance:
        avg_tr_loss = 0
        avg_tr_time = 0
        avg_lat_acc = 0
        avg_lon_acc = 0

        for i, data in enumerate(trDataloader):

            st_time = time.time()
            hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask, vehid, t, ds = data
            mask = mask.bool()

            dist = mask.clone()
            dist = dist.numpy()
            graph = c_dist(dist)
            graph = torch.tensor(graph, dtype=torch.float)

            if args['use_cuda']:
                hist = hist.cuda()
                nbrs = nbrs.cuda()
                mask = mask.cuda()
                lat_enc = lat_enc.cuda()
                lon_enc = lon_enc.cuda()
                fut = fut.cuda()
                op_mask = op_mask.cuda()
                graph = graph.cuda()

            fut_pred, weight_ts_nbr, weight_ha = net(hist, nbrs, mask, lat_enc, lon_enc, graph)

            l = maskedMSE(fut_pred, fut, op_mask)  # maskedNLL(fut_pred, fut, op_mask)

            # Backprop and update weights
            optimizer.zero_grad()
            l.backward()
            a = torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
            optimizer.step()
            # optimizer = tf.train.RMSPropOptimizer(learning_rate, decay).minimize(cost)
            # Track average train loss and average train time:
            batch_time = time.time() - st_time
            avg_tr_loss += l.item()  # sum mse for 100 batches
            avg_tr_time += batch_time

            if i % 100 == 99:
                eta = avg_tr_time / 100 * (len(trSet) / batch_size - i)  # average time/batch * rest batches
                # len(trset) total length; i * batch_size / len(trSet)
                print("Epoch no:", epoch_num + 1, "| Epoch progress(%):",
                      format(i / (len(trSet) / batch_size) * 100, '0.2f'), "| Avg train loss:",
                      format(avg_tr_loss / 100, '0.4f'), "| Acc:", format(avg_lat_acc, '0.4f'),
                      format(avg_lon_acc, '0.4f'),
                      "| Validation loss prev epoch", format(prev_val_loss, '0.4f'), "| ETA(s):", int(eta))
                train_loss.append(avg_tr_loss / 100)
                avg_tr_loss = 0  # clear the result every 100 batches
                avg_lat_acc = 0
                avg_lon_acc = 0
                avg_tr_time = 0
        # _________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

        ## Validate:______________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
        net.train_flag = False

        print("Epoch", epoch_num + 1, 'complete. Calculating validation loss...')
        avg_val_loss = 0
        avg_val_lat_acc = 0
        avg_val_lon_acc = 0
        val_batch_count = 0
        total_points = 0

        for i, data in enumerate(valDataloader):
            st_time = time.time()
            hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask, vehid, t, ds = data
            mask = mask.bool()

            dist = mask.clone()
            dist = dist.numpy()
            graph = c_dist(dist)
            graph = torch.tensor(graph, dtype=torch.float)

            if args['use_cuda']:
                hist = hist.cuda()
                nbrs = nbrs.cuda()
                mask = mask.cuda()
                lat_enc = lat_enc.cuda()
                lon_enc = lon_enc.cuda()
                fut = fut.cuda()
                op_mask = op_mask.cuda()
                graph = graph.cuda()

            fut_pred, weight_ts_nbr, weight_ha = net(hist, nbrs, mask, lat_enc, lon_enc, graph)

            l = maskedMSE(fut_pred, fut, op_mask)

            avg_val_loss += l.item()
            val_batch_count += 1

        print(avg_val_loss / val_batch_count)

        # Print validation loss and update display variables
        print('Validation loss :', format(avg_val_loss / val_batch_count, '0.4f'), "| Val Acc:",
              format(avg_val_lat_acc / val_batch_count * 100, '0.4f'),
              format(avg_val_lon_acc / val_batch_count * 100, '0.4f'))
        val_loss.append(avg_val_loss / val_batch_count)
        prev_val_loss = avg_val_loss / val_batch_count

    end_time = datetime.datetime.now()

    print('Total training time: ', end_time - start_time)
    # __________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

    torch.save(net.state_dict(), 'trained_models/pg_stff_04172023.tar')
