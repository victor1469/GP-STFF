from __future__ import print_function

import numpy as np
import torch
from torch import nn
from model import highwayNet
from utils import ngsimDataset, maskedMSETest
from torch.utils.data import DataLoader
import time
import math
import datetime
import warnings
from joblib import Parallel, delayed
import pandas as pd
import numpy as np


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

    args['train_flag'] = False

    # Evaluation metric:

    metric = 'rmse'

    start_time = datetime.datetime.now()

    # Initialize network
    net = highwayNet(args)
    net.load_state_dict(torch.load('trained_models/pg_stff_04172023.tar'))
    if args['use_cuda']:
        net = net.cuda()

    ## Initialize data loaders
    valSet = ngsimDataset('../data/trajectory/ValSet.mat')
    valDataloader = DataLoader(valSet, batch_size=128, shuffle=True, num_workers=4, collate_fn=valSet.collate_fn,
                               drop_last=True)

    ## Variables holding train and validation loss values:
    lossVals = torch.zeros(5).cuda()
    counts = torch.zeros(5).cuda()
    lossVal = 0  # revised by Lei
    count = 0

    vehid = []
    pred_x = []
    pred_y = []
    T = []
    dsID = []
    ts_cen = []
    ts_nbr = []
    wt_ha = []


    for i, data in enumerate(valDataloader):
        st_time = time.time()
        hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask, vehid, t, ds = data
        mask = mask.bool()

        dist = mask.clone()
        dist = dist.numpy()
        graph = c_dist(dist)
        graph = torch.tensor(graph, dtype=torch.float)

        if not isinstance(hist, list):  # nbrs are not zeros
            vehid.append(vehid)  # current vehicle to predict

            T.append(t)  # current time
            dsID.append(ds)

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

            l, c = maskedMSETest(fut_pred, fut, op_mask)

            fut_pred_x = fut_pred[:, :, 0].detach()
            fut_pred_x = fut_pred_x.cpu().numpy()

            fut_pred_y = fut_pred[:, :, 1].detach()
            fut_pred_y = fut_pred_y.cpu().numpy()
            pred_x.append(fut_pred_x)
            pred_y.append(fut_pred_y)

            lossVal += l.detach()  # revised by Lei
            count += c.detach()

    print('lossVal is:', lossVal)

    print(torch.pow(lossVal / count, 0.5) * 0.3048)  # Calculate RMSE and convert from feet to meters

    end_time = datetime.datetime.now()

    print('Total training time: ', end_time - start_time)
   

---

### Step-by-Step Workflow

#### 1. Historical Trajectory Data Collection
First, historical trajectory data of the ego vehicle (target vehicle) are extracted across the highway segment over the predefined observation horizon ($T_{obs}$). 

<div align="center">
  <img src="assets/fig1.png" width="80%" alt="Fig 1: Target Vehicle Historical Trajectory Collection">
  <p><i>Figure 1: Extraction of historical trajectory points for the target vehicle on the road section.</i></p>
</div>

---

#### 2. Surroundings Perception & Multi-Vehicle Radar Sensing
On-board sensors (e.g., radar, LiDAR) collect spatial context and historical motion trajectories from surrounding traffic agents within the local grid interaction area.

<div align="center">
  <img src="assets/fig2.png" width="80%" alt="Fig 2: Surroundings Radar Sensing">
  <p><i>Figure 2: Perception field capturing surrounding vehicles' historical movement patterns via vehicle radar.</i></p>
</div>

---

#### 3. Spatial-Temporal Feature Fusion & Model Inference
The collected spatial-temporal sequences are processed using the trained **GP-STFF** architecture[cite: 1]. The positional graph ($G_{GP}$) combines topological graph distances with learnable position encodings ($PE$), while the selected Spatial-Temporal Feature Fusion (selected-STFF) module extracts cross-modal interactions to forecast future states over the prediction horizon ($T_{pred}$)[cite: 1].

<div align="center">
  <img src="assets/fig3.png" width="90%" alt="Fig 3: GP-STFF Model Architecture">
  <p><i>Figure 3: Structural diagram of the trained GP-STFF prediction model.</i></p>
</div>

---

#### 4. Future Trajectory Forecasting & Visualization
The decoder outputs sequence-level future trajectory coordinates ($\hat{Y}$), which are rendered onto the spatial domain to demonstrate lane-keeping, lane-changing, and speed variation behaviors.

<div align="center">
  <img src="assets/fig4.png" width="80%" alt="Fig 4: Visualized Trajectory Prediction Result">
  <p><i>Figure 4: Visualized trajectory prediction outputs for target and surrounding traffic agents.</i></p>
</div>

---

#### 5. Density Estimation & Probability Distribution Analysis
To quantify prediction uncertainty and multi-modal trajectory distributions, fine-grained spatio-temporal features are analyzed using spatial heatmaps and positional probability density point maps.

<div align="center">
  <p align="center">
    <img src="assets/fig5.png" width="48%" alt="Fig 5: Spatial Trajectory Heatmap">
    <img src="assets/fig6.png" width="48%" alt="Fig 6: Probability Density Distribution">
  </p>
  <p><i>Figure 5 & Figure 6: Spatial trajectory uncertainty heatmaps (left) and predicted trajectory probability density distribution maps (right).</i></p>
</div>