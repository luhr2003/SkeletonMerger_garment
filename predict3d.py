# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 14:48:13 2020

@author: eliphat
"""
import os
import open3d as o3d

import torch
import merger.merger_net as merger_net
import json
import tqdm
import numpy as np
import argparse

arg_parser = argparse.ArgumentParser(description="Predictor for Skeleton Merger on KeypointNet dataset. Outputs a npz file with two arrays: kpcd - (N, k, 3) xyz coordinates of keypoints detected; nfact - (N, 2) normalization factor, or max and min coordinate values in a point cloud.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
arg_parser.add_argument('-i', '--pcd-path', type=str, default='/home/luhr/correspondence/softgym_cloth/garmentgym/dress',
                        help='Point cloud file folder path from KeypointNet dataset.')
arg_parser.add_argument('-m', '--checkpoint-path', '--model-path', type=str, default='dress.pth',
                        help='Model checkpoint file path to load.')
arg_parser.add_argument('-d', '--device', type=str, default='cuda',
                        help='Pytorch device for predicting.')
arg_parser.add_argument('-k', '--n-keypoint', type=int, default=50,
                        help='Requested number of keypoints to detect.')
arg_parser.add_argument('-b', '--batch', type=int, default=4,
                        help='Batch size.')
arg_parser.add_argument('--max-points', type=int, default=10000,
                        help='Indicates maximum points in each input point cloud.')
ns = arg_parser.parse_args()

def load_cloth_mesh(path):
    """Load .obj of cloth mesh. Only quad-mesh is acceptable!
    Return:
        - vertices: ndarray, (N, 3)
        - triangle_faces: ndarray, (S, 3)
        - stretch_edges: ndarray, (M1, 2)
        - bend_edges: ndarray, (M2, 2)
        - shear_edges: ndarray, (M3, 2)
    This function was written by Zhenjia Xu
    email: xuzhenjia [at] cs (dot) columbia (dot) edu
    website: https://www.zhenjiaxu.com/
    """
    vertices, faces = [], []
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        # 3D vertex
        if line.startswith('v '):
            vertices.append([float(n)
                             for n in line.replace('v ', '').split(' ')])
        # Face
        elif line.startswith('f '):
            idx = [n.split('/') for n in line.replace('f ', '').split(' ')]
            face = [int(n[0]) - 1 for n in idx]
            assert(len(face) == 4)
            faces.append(face)

    triangle_faces = []
    for face in faces:
        triangle_faces.append([face[0], face[1], face[2]])
        triangle_faces.append([face[0], face[2], face[3]])

    stretch_edges, shear_edges, bend_edges = set(), set(), set()

    # Stretch & Shear
    for face in faces:
        stretch_edges.add(tuple(sorted([face[0], face[1]])))
        stretch_edges.add(tuple(sorted([face[1], face[2]])))
        stretch_edges.add(tuple(sorted([face[2], face[3]])))
        stretch_edges.add(tuple(sorted([face[3], face[0]])))

        shear_edges.add(tuple(sorted([face[0], face[2]])))
        shear_edges.add(tuple(sorted([face[1], face[3]])))

    # Bend
    neighbours = dict()
    for vid in range(len(vertices)):
        neighbours[vid] = set()
    for edge in stretch_edges:
        neighbours[edge[0]].add(edge[1])
        neighbours[edge[1]].add(edge[0])
    for vid in range(len(vertices)):
        neighbour_list = list(neighbours[vid])
        N = len(neighbour_list)
        for i in range(N - 1):
            for j in range(i+1, N):
                bend_edge = tuple(
                    sorted([neighbour_list[i], neighbour_list[j]]))
                if bend_edge not in shear_edges:
                    bend_edges.add(bend_edge)
    return np.array(vertices), np.array(triangle_faces),\
        np.array(list(stretch_edges)), np.array(
            list(bend_edges)), np.array(list(shear_edges))




def prepare_data(path):
    data=[]
    paths=[]
    ori_data=[]
    for root,dirs,files in os.walk(path):
        for file in files:
            if file.endswith('.obj'):
                file_path=os.path.join(root,file)
                vertices, trangle_faces, stretch_edges, bend_edges, shear_edges = load_cloth_mesh(os.path.join(root, file))
                points=np.array(vertices)
                ori_data.append(points)
                points=points[np.random.choice(len(points),ns.max_points,replace=True)]
                data.append(points)
                paths.append(file_path)
    return ori_data,data,paths

def find_nearest_point(points,kp):
    kp=np.array(kp)
    points=np.array(points)
    dist=np.linalg.norm(kp[np.newaxis,:,:]-points[:,np.newaxis,:],axis=2)
    return np.argmin(dist,axis=0)

def find_nearest_point_group(points,kp):
    kp=np.array(kp)
    points=np.array(points)
    dist=np.linalg.norm(kp[:,np.newaxis,:]-points[np.newaxis,:,:],axis=2)
    return np.argsort(dist,axis=1)[:,:20]

def visualize_pointcloud(pc,kp):
    kp=kp.reshape(-1)
    pcd=o3d.geometry.PointCloud()
    pcd.points=o3d.utility.Vector3dVector(pc)
    colors= np.zeros_like(pc)
    colors[kp]=np.array([1,0,0])
    pcd.colors=o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])

def get_id_visualize(pc,kp):
    for i in range(len(kp)):
        visualize_pointcloud(pc,kp[i])
        print(i)


if __name__=='__main__':

    net = merger_net.Net(ns.max_points, ns.n_keypoint).to(ns.device)
    net.load_state_dict(torch.load(ns.checkpoint_path, map_location=torch.device(ns.device))['model_state_dict'])
    net.eval()

    ori_data,kpn_ds,paths=prepare_data(ns.pcd_path)

    out_kpcd=[]
    for i in tqdm.tqdm(range(0, len(kpn_ds), ns.batch), unit_scale=ns.batch):
        Q = []
        for j in range(ns.batch):
            if i + j >= len(kpn_ds):
                continue
            pc = kpn_ds[i + j]
            Q.append(pc)
        if len(Q) == 1:
            Q.append(Q[-1])
        with torch.no_grad():
            recon, key_points, kpa, emb, null_activation = net(torch.Tensor(np.array(Q)).to(ns.device))
        for kp in key_points:
            out_kpcd.append(kp)
    for i in range(len(out_kpcd)):
        out_kpcd[i] = out_kpcd[i].cpu().numpy()
    for i in range(len(out_kpcd)):
        kp_id=find_nearest_point(ori_data[i],out_kpcd[i])
        # visualize_pointcloud(ori_data[i],kp_id)
        # get_id_visualize(ori_data[i],kp_id)
        print("save keypoints to "+paths[i].replace('.obj','keypoints.npz'+str(ns.n_keypoint)))
        np.savez(paths[i].replace('.obj','keypoints.npz'+str(50)),keypoints=out_kpcd[i],keypoint_id=kp_id,pointcloud=ori_data[i])
