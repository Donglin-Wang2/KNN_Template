from utils import *

import os

import torch
import numpy as np
import open3d as o3d
import torchmetrics
from tqdm import tqdm
from tabulate import tabulate
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import knn_points, knn_gather

data_root = './data/shapenet_part/processed/'
points = read_pkl(os.path.join(data_root, 'points_by_cat.pkl'))
point_labs = read_pkl(os.path.join(data_root, 'point_labels_by_cat.pkl'))
temp_idx = read_pkl(os.path.join(data_root, 'temp_index.pkl'))

cat_num_part = {'Motorbike': 6, 'Guitar': 3, 'Rocket': 3, 'Cap': 2, 'Bag': 2, 'Airplane': 4, 'Lamp': 4, 'Car': 4, 'Skateboard': 3, 'Table': 3, 'Mug': 2, 'Knife': 2, 'Chair': 4, 'Laptop': 2, 'Pistol': 3, 'Earphone': 3}
color_map = np.array([[235, 87, 87], [242, 153, 74], [242, 201, 76],
    [33, 150, 83], [47, 128, 237], [155, 81, 224]])

# 1. Select K templates at random for the template pool
# 2. For points not in template pool, find the closest template
# 3. For each point in the shape, find the closest point on its corresponding template and inherit the pred. 

n_temps = [3, 5, 7, 10]
K = 3
result = {n_temp:[] for n_temp in n_temps}

# n_K = [1, 3, 5, 7]
# n_temp = 10
# result = {K:[] for K in n_K}

result['Category'] = [key for key in points.keys()]

DEVICE = torch.device('cuda')

for n_temp in tqdm(n_temps):
# for K in tqdm(n_K):
    for cat in tqdm(points.keys(), leave=False):
        n_points = len(points[cat])
        if len(temp_idx[cat]) > n_temp:
            chosen_temp_idx = np.random.choice(temp_idx[cat], n_temp, replace=False)
        else:
            faux_temp_idx = np.arange(len(points[cat]))
            faux_temp_idx = np.delete(faux_temp_idx, temp_idx[cat])
            faux_temp_idx = np.random.choice(faux_temp_idx, n_temp - len(temp_idx[cat]), replace=False)
            chosen_temp_idx = np.concatenate((temp_idx[cat], faux_temp_idx))
            
        non_temp_idx = np.ones(n_points, dtype=bool)
        non_temp_idx[chosen_temp_idx] = False
        temp_pool = torch.Tensor(points[cat][chosen_temp_idx]).to(DEVICE)
        temp_lab_pool = torch.Tensor(point_labs[cat][chosen_temp_idx]).long().to(DEVICE)
        non_temp = torch.Tensor(points[cat][non_temp_idx]).to(DEVICE)

        chosen_temp = []
        chosen_temp_lab = []

        for shape in tqdm(non_temp, leave=False):
            shape = shape.unsqueeze(0).repeat(n_temp, 1, 1)
            dists, _ = chamfer_distance(shape, temp_pool, batch_reduction=None)
            min_dix = torch.argmin(dists)
            chosen_temp.append(temp_pool[min_dix])
            chosen_temp_lab.append(temp_lab_pool[min_dix])

        chosen_temp = torch.stack(chosen_temp).to(DEVICE)
        chosen_temp_lab = torch.stack(chosen_temp_lab).unsqueeze(-1).to(DEVICE)
        dists, idx, nn = knn_points(non_temp, chosen_temp, K=K)
        pred = knn_gather(chosen_temp_lab, idx).squeeze(-1)
        pred, _ = torch.mode(pred)
        gt = torch.Tensor(point_labs[cat][non_temp_idx]).long().to(DEVICE)
        avg_iou = 0.0
        for obj_pred, obj_gt in zip(pred, gt):
            avg_iou += iou(obj_pred, obj_gt)
        avg_iou = avg_iou / len(gt)
        result[n_temp].append(avg_iou.item())
        

        # n_sample = 5

        # for i in np.random.choice(np.arange(len(non_temp)), n_sample, replace=False):
        #     sample_labels = pred[i]
        #     sample_points = o3d.utility.Vector3dVector(non_temp[i])
        #     colors = o3d.utility.Vector3dVector([color_map[j] for j in sample_labels.numpy()])
        #     pcd = o3d.geometry.PointCloud()
        #     pcd.points = sample_points
        #     pcd.colors = colors
        #     o3d.visualization.draw_geometries([pcd])

# print(tabulate(result, tablefmt="pipe", headers=['Category'] + [f'{n_temp} templates' for n_temp in n_temps]))
# print(tabulate(result, tablefmt="pipe", headers=['Category'] + [f'{K} nearest' for K in n_K]))
write_pkl(result, './logs/knn_n_temps.pkl')

