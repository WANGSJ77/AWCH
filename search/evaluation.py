# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 14:25:44 2019

@author: 71773
"""
import numpy as np
import os
from typing import Dict, List
import scipy.io as scio
import torch
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# KMP_DUPLICATE_LIB_OK = TRUE
# from config import config

def L2_normalization(X):
    """
    —对输入数据（X逐行）进行L2标准化
    """
    X_norm = X.copy()
    n = X.shape[0]
    for i in range(n):
        # print(i)
        c_norm = np.linalg.norm(X[i])  # 求出每行向量的模长
        if c_norm == 0:  # 特殊情况：全0向量
            c_norm = 1
        X_norm[i] = X[i] / c_norm
    return X_norm


def Euclidean_distance(X, Y):
    """
    —欧氏距离
    输入：
    X n*p数组, p为特征维度
    Y m*p数组

    输出：
    D n*m距离矩阵
    """
    n = X.shape[0]
    m = Y.shape[0]
    X2 = np.sum(X ** 2, axis=1)
    Y2 = np.sum(Y ** 2, axis=1)
    D = np.tile(X2.reshape(n, 1), (1, m)) + (np.tile(Y2.reshape(m, 1), (1, n))).T - 2 * np.dot(X, Y.T)
    D[D < 0] = 0
    D = np.sqrt(D)
    return D

def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH


def evaluation(feature_path, lable_path, distance_func):
    """
    features_path 图像特征库路径
    label_path 图像标签库路径
    distance_func 距离度量函数
    """
    # ----------------------预处理----------------------------
    # -----输入为.npy文件路径
    # img_all = np.load(feature_all_path, allow_pickle=True).item()
    # img_query = np.load(feature_query_path, allow_pickle=True).item()
    img_all_features = feature_path
    img_all_labels = lable_path
    img_query_features = feature_path
    img_query_labels = lable_path

    # -----输入为.mat文件路径
    # img_all_features = scio.loadmat(feature_all_path)['features']
    # img_all_labels = scio.loadmat(feature_all_path)['labels']
    # img_query_features = scio.loadmat(feature_query_path)['features']
    # img_query_labels = scio.loadmat(feature_query_path)['labels']

    # 检索结果存放文件夹
    # FE_identity = os.path.splitext(os.path.split(feature_query_path)[1])[0]
    # result_folder = results_root + FE_identity
    # os.makedirs(result_folder, exist_ok=True)
    # 存储评价结果
    results_array = {}

    maxdepth = img_all_labels.size  # 一次查询最大的返回图像数
    nQueries = img_query_labels.size  # 查询次数， 即查询图像数目

    unique_labels = np.unique(img_all_labels)  # 图像库类别数
    rel_docs = np.zeros(nQueries, float)  # 存储查询图像中每幅图像在图像库中的同类图像数目
    rel_classes = []  # 存储查询图像中每一个类别的图像数
    for i in unique_labels:
        res = np.where(img_all_labels == i)[0]  # 找出label为i的所有图像位置(下标)
        rel_docs[np.where(img_query_labels == i)[0]] = res.size
        rel_classes.append((np.where(img_query_labels == i)[0]).size)

    Kq = 2 * rel_docs
    rep_labels = np.tile(img_all_labels.reshape(1, -1), (nQueries, 1))  # 每行存储图像库中图像的label
    gt_labels = np.tile(img_query_labels, (1, maxdepth))  # 每一列存储查询图像的label

    distance = []
    imgFea_all_norm = L2_normalization(img_all_features)
    imgFea_query_norm = L2_normalization(img_query_features)

    # print(imgFea_query_norm.shape)
    if distance_func == 'euclidean_distance':
        distance = Euclidean_distance(imgFea_query_norm, imgFea_all_norm)  # 距离矩阵,每行是一幅查询图像的检索结果
        # print(distance)
    elif distance_func == 'hamming_distance':
        distance = CalcHammingDist(imgFea_query_norm, imgFea_all_norm)  # 距离矩阵,每行是一幅查询图像的检索结果
        # np.save('../results/distance2.npy', {'Distance': distance})  # ——存入.npy文件
        # scio.savemat(result_folder+'/distance.mat', {'Distance': distance})   #——存入.mat文件
    image_idxs = np.argsort(distance, axis=1)  # 对distance进行按行排序，返回结果为排序后的索引
    # print(image_idxs)
    # ------------------------计算指标_AP--------------------------
    ##performance evaluation
    result_labels = np.zeros((nQueries, maxdepth))  # 存储nQuieries次查询排序后图像的标签
    for i in range(nQueries):
        current_labels = rep_labels[i, :]  # 第i次查询图像库中图像的label
        temp_idxs = image_idxs[i, :]  # 第i次查询排序结果
        result_labels[i, :] = current_labels[temp_idxs]  # 第i次查询排序后标签

    results_maxdepth = (result_labels == gt_labels)  # nQueries次查询排序后的图像标签与真实标签比较
    precision = np.zeros((nQueries, maxdepth), float)  # 存储nQueries次查询，返回图像从1:maxdepth时的查准率
    recall = np.zeros((nQueries, maxdepth), float)
    avg_precision = np.zeros((nQueries, 1), float)
    results = []
    for pr in range(maxdepth):  # 计算average precision
        results = (result_labels[:, 0:pr + 1] == gt_labels[:, 0:pr + 1])
        num_tp = np.sum(results, axis=1)  # 返回影像数为pr时的相似图像数
        precision_k = num_tp / (pr + 1)
        recall_k = num_tp / rel_docs
        precision[:, pr] = precision_k
        recall[:, pr] = recall_k
        avg_precision = avg_precision + (precision_k * results_maxdepth[:, pr]).reshape(-1, 1)
    avg_precision = avg_precision / rel_docs

    # ------------------------计算指标_PVR曲线-----------------------
    global_precision = np.mean(precision, axis=0)  # 逐列求均值
    global_recall = np.mean(recall, axis=0)
    # np.save(result_folder + '/PR.npy',
    #         {'global_precision': global_precision, 'global_recall': global_recall})  # ——存入.npy文件
    # scio.savemat(result_folder+'/PR.mat', {'global_precision': global_precision, 'global_recall':global_recall}) #——存入.mat文件

    interpolated_precision = []
    interpolated_recall = np.linspace(0, 1, 11)
    for recall in interpolated_recall:
        recall_idx = (global_recall >= recall)
        selected_precision = global_precision[recall_idx]
        pr = max(selected_precision)
        interpolated_precision.append(pr)
    # ——存入.npy文件
    # np.save(result_folder + '/interpolated_PR.npy',
    #         {'interpolated_precision': interpolated_precision, 'interpolated_recall': interpolated_recall})
    # ——存入.mat文件
    # scio.savemat(result_folder+'/interpolated_PR.mat', {'interpolated_precision': interpolated_precision, 'interpolated_recall':interpolated_recall})

    # ------------------------计算指标_mAP_AVR_NMRR_ANMRR--------------------
    MAP = np.mean(avg_precision)
    AVR = np.zeros((nQueries, 1), float)
    NMRR = np.zeros((nQueries, 1), float)  # nQueries次查询的NMRR

    for q in range(nQueries):
        for n in range(maxdepth):
            if ((n + 1) <= Kq[q]):
                AVR[q] = AVR[q] + (n + 1) * results[q, n]
            else:
                AVR[q] = AVR[q] + (1.25 * Kq[q]) * results[q, n]
        AVR[q] = AVR[q] / rel_docs[q]
        NMRR[q] = (AVR[q] - 0.5 * (1 + rel_docs[q])) / (1.25 * Kq[q] - 0.5 * (1 + rel_docs[q]))
    ANMRR = np.mean(NMRR)
    results_array['ANMRR'] = ANMRR
    results_array['MAP'] = MAP * 100

    precision_steps = [5, 10, 20, 50, 100]  # 返回图像数，计算P@5,10,20,50,100
    print('ANMRR为:{:.3f} MAP为:{:.2f}%'.format(ANMRR, MAP * 100))
    for pr in precision_steps:
        prr = np.mean(precision[:, pr - 1]) * 100
        print('P@{}:{:.2f}%'.format(pr, prr), end=' ')
        results_array['P@{}'.format(pr)] = round(prr, 2)

    # np.save(result_folder + '/results.npy', results_array)  # ——存入.npy文件
    # scio.savemat(result_folder+'/results.mat', {'results': results_array}) #——存入.mat文件

    # with open(result_folder + '/results.txt', "w") as f:
    #     f.write(str(results_array))

if __name__ == "__main__":
    # results_root = "./Resnet50/results/"
    distance_func = "euclidean_distance"
    # feature_folder = "./Resnet50/features/"
    # for feature in os.listdir(feature_folder):
    #     feature_path = feature_folder + feature
    #     evaluation(feature_path, feature_path, distance_func, results_root)

    # npy形式数据

    loadData_feature_low = np.load('../特征和标签/feature_low.npy')
    loadData_feature_low = torch.Tensor(loadData_feature_low)
    loadData_feature_low = torch.squeeze(loadData_feature_low)
    loadData_feature_low = np.array(loadData_feature_low)
    #
    loadData_feature_high = np.load('../特征和标签/feature_high.npy')
    loadData_feature_high = torch.Tensor(loadData_feature_high)
    loadData_feature_high = torch.squeeze(loadData_feature_high)
    loadData_feature_high = np.array(loadData_feature_high)

    distance_func2 = "hamming_distance"
    loadData_hashcode = np.load('../特征和标签/hashcode.npy')
    loadData_hashcode = torch.Tensor(loadData_hashcode)
    loadData_hashcode = torch.squeeze(loadData_hashcode)
    loadData_hashcode = np.array(loadData_hashcode)

    loadData_lable = np.load('../特征和标签/lable.npy')
    # loadData_feature = loadData_feature.reshape(420,512)
    print("\nlow-feature------------------------")
    evaluation(loadData_feature_low, loadData_lable, distance_func)
    print("\nhigh-feature------------------------")
    evaluation(loadData_feature_high, loadData_lable, distance_func)
    # print("\npre-feature------------------------")
    # evaluation(pre_feature, loadData_lable, distance_func)
    print("\nhashcode------------------------")
    evaluation(loadData_hashcode, loadData_lable, distance_func2)
    #
    # pre_feature = np.load('../特征和标签/pre_feature.npy')
    # pre_feature = torch.Tensor(pre_feature)
    # pre_feature = torch.squeeze(pre_feature)
    # pre_feature = np.array(pre_feature)

 # mat数据
# if __name__ == '__main__':
    # distance_func = "hamming_distance"
    # load_data = scio.loadmat('../特征和标签/hashcode_test.mat')
    # loadData_hashcode = load_data['test']
    # loadData_hashcode = torch.Tensor(loadData_hashcode)
    # loadData_hashcode = torch.squeeze(loadData_hashcode)
    # loadData_hashcode = np.array(loadData_hashcode)
    # loadData_lable = load_data['label']
    # loadData_lable = loadData_lable.reshape([420,1])
    # loadData_hashcode = np.array(loadData_hashcode)
    # evaluation(loadData_hashcode, loadData_lable, distance_func)

    # loadData_hashcode = np.load('../特征和标签/hashcode.npy')
    # loadData_hashcode = torch.Tensor(loadData_hashcode)
    # loadData_hashcode = torch.squeeze(loadData_hashcode)
    # loadData_hashcode = np.array(loadData_hashcode)
    # loadData_lable = np.load('../特征和标签/lable.npy')
    # print(loadData_hashcode)
    # print(loadData_lable)
    # evaluation(loadData_hashcode, loadData_lable, distance_func)