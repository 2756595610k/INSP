import numpy as np
import copy
import random
from numpy import array
from pycm import *

def load_data(data, social_network, missing_label):  # data = data/twitter_data.npy
    # social_network = data/twitter_network.pkl
    input_data_list = []  # missing_label = -1
    input_time_list = []
    input_mask_list = []
    input_interval_list = []
    input_length_list = []

    output_mask_all_list = []
    origin_u = []

    max_length = data.shape[1]
    channel = data.shape[2]
    upper = np.max(data)  # 0.9999983310699463

    # find min value
    data_tmp = copy.deepcopy(data)
    data_tmp[data_tmp == missing_label] = upper
    lower = np.min(data_tmp)
    del data_tmp

    for u in data:
        idx_u = np.arange(len(u))  # idx_u=[0,1,2,..124]
        # print idx_u.shape
        u[u != missing_label] = (u[u != missing_label] - lower) / (upper - lower)  # 单个用户的所有数据（包括缺失）

        valid_data = u[u[:, 0] != missing_label]  # 每个用户的有效数据(去掉缺失的位置)(索引从0开始)
        valid_real_idx = idx_u[u[:, 0] != missing_label]  # 有效数据在原始数据的索引(125)

        input_idx = valid_real_idx
        input_data = np.zeros((max_length, channel))
        input_mask = np.zeros(max_length)
        input_length = np.zeros(max_length)
        input_interval = np.zeros(max_length) + 1
        input_time = np.zeros(max_length)
        output_mask_all = np.zeros(max_length)

        input_data[:len(input_idx)] = valid_data  # 125位，前面填充完整有效数据，后面填充缺失数据（0占位）
        input_time[:len(input_idx)] = valid_real_idx  # 每个数据对应的时间
        input_mask[:len(input_idx)] = 1  # 有效数据用1，无效数据用0
        input_length[min(len(input_idx) - 1, max_length - 1)] = 1

        input_real_idx = valid_real_idx
        output_mask_all[input_real_idx.astype(dtype=int)] = 1  # 原始数据对应位置，有效数据的位置用1，无效用0

        input_interval[0] = 1
        input_interval[1:len(input_idx)] = input_real_idx[1:] - input_real_idx[:-1]  # 有效数据的时间间隔

        input_data_list.append(input_data)
        input_time_list.append(input_time)
        input_mask_list.append(input_mask)
        input_interval_list.append(input_interval)
        input_length_list.append(input_length)

        origin_u.append(list(reversed(u)))  # 原始数据反转
        output_mask_all_list.append(list(reversed(output_mask_all)))

    input_data_list = np.array(input_data_list).astype(dtype=np.float32)
    input_time_list = np.array(input_time_list).astype(dtype=np.float32)
    input_mask_list = np.array(input_mask_list).astype(dtype=np.float32)
    input_interval_list = np.array(input_interval_list).astype(dtype=np.float32)
    input_length_list = np.array(input_length_list).astype(dtype=np.float32)
    output_mask_all_list = np.array(output_mask_all_list).astype(dtype=np.float32)
    origin_u = np.array(origin_u).astype(dtype=np.float32)

    max_num = 8
    # neighbor_data.shape:(3494, 8, 125, 6)
    neighbor_length = np.zeros((input_length_list.shape[0], max_num, input_length_list.shape[1])).astype(
        dtype=np.float32)
    neighbor_interval = np.zeros((input_interval_list.shape[0], max_num, input_interval_list.shape[1])).astype(
        dtype=np.float32)
    neighbor_time = np.zeros((input_interval_list.shape[0], max_num, input_interval_list.shape[1])).astype(
        dtype=np.float32)
    neighbor_data = np.zeros(
        (input_data_list.shape[0], max_num, input_data_list.shape[1], input_data_list.shape[2])).astype(
        dtype=np.float32)

    for i, neighbors in enumerate(social_network):  # social_network本身长度：3494;其本身存放的是一个用户对应的邻居
        for j in range(max_num):  # 只考虑8个邻居
            m = random.randint(0, len(neighbors))  # 随机返回邻居中的其中一个
            m = (neighbors + [i])[m]  # neighbors是个list  ,如果没有邻居，取自己作为邻居，否则随机选取一个
            neighbor_length[i][j] = input_length_list[m]
            neighbor_interval[i][j] = input_interval_list[m]
            neighbor_time[i][j] = input_time_list[m]
            neighbor_data[i][j] = input_data_list[m]
    # print(neighbor_length)
    return input_data_list, input_time_list, input_mask_list, input_interval_list, input_length_list, \
           output_mask_all_list, origin_u, neighbor_length, neighbor_interval, neighbor_time, neighbor_data, lower, upper


def li(p):
    s = [0, 0, 0, 0, 0, 0]
    for i in p:
        s[int(i)] += 1


""""
预测数据
"""
def process():
    # twitter 0:Anger 1:Disgust 2 Fear 3 Happiness 4 Sadness 5 Surprise
   # predict = np.load("data/twitter_label_data.npy")
   # f = open('data/true_label_twitter.txt', 'r')
    predict = np.load("input_label_data/twitter_label_data.npy")
    f = open('input_label_data/true_label_twitter.txt', 'r')
    a = f.read()
    tr = eval(a)
    f.close()

    pred_list = []
    label_list = []
    for i in tr:
        label_list.append(np.argmax(tr[i][1]))  # np.argmax(tr[i][1]):返回的是tr中元素最大值所对应的索引值
        pred_list.append(np.argmax(predict[i][tr[i][0]]))

    # print()

    li(pred_list)
    # cal_precision_and_recall(label_list, pred_list)
    # print(classification_report(label_list, pred_list, digits=5))

    cm = ConfusionMatrix(actual_vector=label_list, predict_vector=pred_list)
    print("Average Accuracy:", cm.overall_stat['ACC Macro'])
    cm.F_beta(1)
    print("ACC(Accuracy):", cm.class_stat["ACC"])
    print(cm)
    # print(type(cm.print_matrix()))
    # print(cm.print_matrix())
