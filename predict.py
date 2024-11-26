# -*- coding: utf-8 -*-
import os
import sys
import torch
import numpy as np
import argparse
#import torchvision
#import torchvision.transforms as transforms
import torch.nn.functional as F
import dataprocess
from model import *
import time
import pickle
import logging


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--data_file', type=str, default='dataServer/twitter_data.npy', help='path of input file')
parser.add_argument('-n', '--social_network', type=str, default='dataServer/twitter_network.pkl', help='path of network file')
parser.add_argument('-o', '--output_file', type=str, default='dataServer/twitter_label_data.npy', help='path of output file')
parser.add_argument('-m', '--missing_marker', type=float, default=-1, help='marker of missing elements, default value is -1')
parser.add_argument('-b', '--batch_size', type=int, default=64, help='the number of samples in each batch, default value is 256')
parser.add_argument('-e', '--num_epoch', type=str, default=280, help='number of epoch, default value is 200')
parser.add_argument('-s', '--hidden_size', type=int, default=32, help='size of hidden feature in LSTM, default value is 32')
parser.add_argument('-k', '--dim_memory', type=int, default=32, help='dimension of memory matrix, default value is 32')
parser.add_argument('-l', '--learning_rate', type=float, default=0.0002)
parser.add_argument('-d', '--dropout', type=float, default=0.96, help='the dropout rate of output layers, default value is 0.8')
parser.add_argument('-r', '--decoder_learning_ratio', type=float, default=5, help='ratio between the learning rate of decoder and encoder, default value is 5')
parser.add_argument('-w', '--weight_decay', type=float, default=0)
parser.add_argument('--log', action='store_true', help='print log information, you can see the train loss in each epoch')
parser.add_argument('--mode', type=str, default='train', help='')
#parser.add_argument('--save_path', type=str, default='modelParameter/logs_modelParameter/', #help='')
parser.add_argument('--save_path', type=str, default='modelParameter/twitter_modelParameter/', help='')
parser.add_argument('--save_step_interval', type=str, default='500', help='')
parser.add_argument('--log_step_interval', type=str, default='20', help='')

args = parser.parse_args()
logging.basicConfig(
    level=logging.WARN,
    stream=sys.stdout,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)

# Device configuration
if torch.cuda.is_available():
    print("cuda is available")
    torch.cuda.set_device(0)

# 设置pytorch中默认的浮点类型
torch.set_default_tensor_type('torch.FloatTensor')

# Hyper-parameters
hidden_size = args.hidden_size
batch_size = args.batch_size
K = args.dim_memory   # dimension of memory matrix, default value is 32
num_epochs = args.num_epoch

learning_rate = args.learning_rate
weight_decay = args.weight_decay
dropout = args.dropout
decoder_learning_ratio = args.decoder_learning_ratio

input_data = np.load(args.data_file)  # data/twitter_data.npy  input_data.shape: (3494,125,6)
input_size = input_data.shape[2]      # input_size == 6
S = input_data.shape[1]               # S == 125 (sequence length)
# K == 32
# L：位置编码计算公式
L = [[1. * s * k / S / K + (1 - 1. * k / K) * (1 - 1. * s / S) for k in range(1, K + 1)] for s in range(1, S + 1)]
#print("L:",L)
L = th.from_numpy(np.array(L))
#print("L:",L)
"""  L: tensor([[0.9612, 0.9305, 0.8998,  ..., 0.0695, 0.0387, 0.0080],
        [0.9537, 0.9235, 0.8932,  ..., 0.0765, 0.0462, 0.0160],
        [0.9463, 0.9165, 0.8867,  ..., 0.0835, 0.0537, 0.0240],
        ...,
        [0.0463, 0.0765, 0.1068,  ..., 0.9235, 0.9538, 0.9840],
        [0.0388, 0.0695, 0.1003,  ..., 0.9305, 0.9612, 0.9920],
        [0.0312, 0.0625, 0.0938,  ..., 0.9375, 0.9688, 1.0000]],
       dtype=torch.float64)     torch.Size([125, 32])
       """

class DataSet(torch.utils.data.Dataset):
    def __init__(self):
        super(DataSet, self).__init__()

        self.input_data, self.input_time, self.input_mask, self.input_interval, self.input_length, \
           self.output_mask_all, self.origin_u, self.neighbor_length, self.neighbor_interval, self.neighbor_time, self.neighbor_data, self.lower, self.upper\
            = dataprocess.load_data(np.load(args.data_file), pickle.load(open(args.social_network, 'rb')), args.missing_marker)
        self.input_interval = np.expand_dims(self.input_interval, axis=2)
        self.neighbor_interval = np.expand_dims(self.neighbor_interval, axis=3)
        self.mask_in = (self.output_mask_all == 1).astype(dtype=np.float32)
        self.mask_out = (self.output_mask_all == 2).astype(dtype=np.float32)
        self.mask_all = (self.output_mask_all != 0).astype(dtype=np.float32)

    def __getitem__(self, index):
        return self.input_data[index], self.input_mask[index], self.input_interval[index], self.input_length[index],\
               self.origin_u[index], self.mask_in[index], self.mask_out[index], self.mask_all[index], self.neighbor_data[index],\
               self.neighbor_interval[index], self.neighbor_length[index]

    def __len__(self):
        return len(self.input_data)


# print(input_data.shape[1])
encoder = MemoryEncoder(input_size, hidden_size, K)
neighbor_encoder = MemoryEncoder(input_size, hidden_size, K)
decoder = DecoderRNN(input_size, hidden_size, 6, K)

if th.cuda.is_available():
    encoder = encoder.cuda()
    decoder = decoder.cuda()

# optimizer
encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
neighbor_encoder_optimizer = torch.optim.Adam(neighbor_encoder.parameters(), lr=learning_rate)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)

if th.cuda.is_available():
    encoder = encoder.cuda()
    neighbor_encoder = neighbor_encoder.cuda()
    decoder = decoder.cuda()

# Train the model
if args.mode == "train":
    # 加载训练数据
    train_dataset = DataSet()
    print('load Training data successfully')
    beginTime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    print("start training time: [{}]".format(beginTime))
    start_epoch = 0
    start_step = 0
    #resume = args.save_path + "epoch_295.pt"
    resume = ""
    if resume != "":
            #  加载之前训过的模型的参数文件
            logging.warning(f"loading from {resume}")
            checkpoint = torch.load(resume, map_location=torch.device('cuda')) #可以是cpu,cuda,cuda:index
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
            neighbor_encoder.load_state_dict(checkpoint['neighbor_encoder_state_dict'])
            decoder.load_state_dict(checkpoint['decoder_state_dict'])

            encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer_state_dict'])
            neighbor_encoder_optimizer.load_state_dict(checkpoint['neighbor_encoder_optimizer_state_dict'])
            decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            # start_step = checkpoint['step']
            # 模型拷贝
            encoder.cuda()
            neighbor_encoder.cuda()
            decoder.cuda()
    curve = []
    curve_train = []
    best_performance = [10000, 10000]
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
    for epoch in range(start_epoch, num_epochs):
        num_batches = len(train_loader)
        loss_all, num_all = 0, 0
        for batch_index, (i_data, i_mask, i_interval, i_length, u_all, m_in, m_out, m_all, n_input, n_inter, n_len) in enumerate(train_loader):
            input = i_data
            # print("i_data.shape:",i_data.shape)  [256, 125, 6],最后一个是:[166, 125, 6]  n_inpt: (256,8,125,6)
            m_all = m_all.unsqueeze(2)
            m_in = m_in.unsqueeze(2)
            # print u_all.size()
            if th.cuda.is_available():
                input = input.cuda()
                i_length = i_length.cuda()
                i_interval = i_interval.cuda()
                i_mask = i_mask.cuda()
                u_all = u_all.cuda()
                m_in = m_in.cuda()
                n_input = n_input.cuda()
                n_inter = n_inter.cuda()
                n_len = n_len.cuda()
                #n_data = n_data.cuda()
                #n_mask = n_mask.cuda()

            step = num_batches * epoch + batch_index + 1
            loss, num = train_batch(input, i_length, i_interval,i_mask, u_all, m_in, n_input, n_inter, n_len,
                    encoder.train(), neighbor_encoder.train(), decoder.train(), encoder_optimizer, neighbor_encoder_optimizer, decoder_optimizer)
            loss_all += loss
            num_all += num
            log_step_interval = int(args.log_step_interval)
            #if step % log_step_interval == 0:
            if num_batches == batch_index+1:  # 一个epoch打印一次
                print("epoch:{}".format(epoch), "batch_index:{}".format(batch_index), 'ema_loss:',
                          loss_all * (train_dataset.upper - train_dataset.lower) * 1. / num_all / train_dataset.input_data.shape[2])
                curve_train.append(loss_all)
            # if step % args.log_step_interval == 0:
            #     logging.warning(f"epoch_index: {epoch}, batch_index: {batch_index}, ema_loss: {epoch}")
            #save_step_interval = int(args.save_step_interval)
            #if epoch % save_step_interval == 0:
        if (epoch+1) >= 5 and (epoch+1) % 5 == 0:  # epoch从0开始计算， 5个epoch保存一次
            # print("epoch:{}".format(epoch), "batch_index:{}".format(batch_index), 'ema_loss:',
            #       loss_all * (train_dataset.upper - train_dataset.lower) * 1. / num_all /
            #       train_dataset.input_data.shape[2])
            # curve_train.append(loss_all)
            os.makedirs(args.save_path, exist_ok=True)
            save_file = os.path.join(args.save_path, f"epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                #'step': step,
                'encoder_state_dict': encoder.state_dict(),
                'neighbor_encoder_state_dict': neighbor_encoder.state_dict(),
                'decoder_state_dict':  decoder.state_dict(),
                'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
                'neighbor_encoder_optimizer_state_dict': neighbor_encoder_optimizer.state_dict(),
                'decoder_optimizer_state_dict': decoder_optimizer.state_dict(),
                'loss': loss_all,
            }, save_file)
            #resume = save_file
            logging.warning(f"checkpoint has been saved in {save_file}")

    EndTime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    print(" End training time: [{}]".format(EndTime))
    #if args.log:
     # print('train epoch {} mse:'.format(epoch), loss_all * (train_dataset.upper - train_dataset.lower) *1./num_all/train_dataset.input_data.shape[2])
    #curve_train.append(loss_all)


#if args.mode == "test":
# test_dataset =DataSet()
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
# print("Start test")
# predict_data_all = []
# for i, (i_data, i_mask, i_interval, i_length, u_all, m_in, m_out, m_all, n_input, n_inter, n_len) in enumerate(test_loader):
#     input = i_data
#     m_out = m_out.unsqueeze(2)
#     m_in = m_in.unsqueeze(2)
#     m_all = m_all.unsqueeze(2)
#     if th.cuda.is_available():
#         input = input.cuda()
#         i_length = i_length.cuda()
#         i_interval = i_interval.cuda()
#         i_mask = i_mask.cuda()
#         m_out = m_out.cuda()
#         m_in = m_in.cuda()
#         m_all = m_all.cuda()
#         u_all = u_all.cuda()
#         n_input=n_input.cuda()
#         n_inter=n_inter.cuda()
#         n_len=n_len.cuda()
#
#     #predictd_data = predict_batch(i_data, i_length, i_interval, i_mask, u_all, m_all, n_input, n_inter, n_len,
#     predictd_data = predict_batch(input, i_length, i_interval, i_mask, u_all, m_all, n_input, n_inter, n_len,
#             encoder.eval(), neighbor_encoder.eval(), decoder.eval())
#
#     predict_data_all.append(predictd_data)
#
# #print(predict_data_all[0], type(predict_data_all[0]))
# predict_data_all = np.concatenate(predict_data_all, axis=0)
# predict_data_all = predict_data_all * (test_dataset.upper - test_dataset.lower) + test_dataset.lower
# np.save(args.output_file, predict_data_all[:,::-1])
# print("test finish")
dataprocess.process()

# if __name__ == "__main__":
#     if torch.cuda.is_available():
#         logging.warning("Cuda is available!")
#         os.environ['CUDA_VISIBLE_DEVICES'] = 0
#     else:
#         logging.warning("Cuda is not available! Exit!")
#         #return
#     #print("模型总参数:", sum(p.numel() for p in model.parameters()))
#     resume = ""
#     if resume != "":
#         #  加载之前训过的模型的参数文件
#         logging.warning(f"loading from {resume}")
#         checkpoint = torch.load(resume,map_location=torch.device('cpu')) #可以是cpu,cuda,cuda:index
#         #$model.load_state_dict(checkpoint['model_state_dict'])
#         #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         start_epoch = checkpoint['epoch']
#         start_step = checkpoint['step']
