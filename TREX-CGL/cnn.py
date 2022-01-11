import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from utils import get_gaze_heatmap


class Net(nn.Module):
    def __init__(self, gaze_loss_type):
        super().__init__()

        self.conv1 = nn.Conv2d(4, 16, 7, stride=3)  # 26x26
        self.conv2 = nn.Conv2d(16, 16, 5, stride=2)  # 11x11
        self.conv3 = nn.Conv2d(16, 16, 3, stride=1)  # 9x9
        self.conv4 = nn.Conv2d(16, 16, 3, stride=1)  # 7x7
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 1)

        self.gaze_loss_type = gaze_loss_type

    def cum_return(self, traj, use_gaze=False):
        '''calculate cumulative return of trajectory'''
        sum_rewards = 0
        sum_abs_rewards = 0
        x = traj.permute(0, 3, 1, 2)  # get into NCHW format
        # compute forward pass of reward network (we parallelize across frames so batch size is length of partial trajectory)
        x1 = F.leaky_relu(self.conv1(x))
        x2 = F.leaky_relu(self.conv2(x1))
        x3 = F.leaky_relu(self.conv3(x2))
        x4 = F.leaky_relu(self.conv4(x3))
        x = x4.view(-1, 784)
        x = F.leaky_relu(self.fc1(x))
        r = self.fc2(x)
        sum_rewards += torch.sum(r)
        sum_abs_rewards += torch.sum(torch.abs(r))

        # prepare conv map to be returned for gaze loss
        conv_map_traj = []
        conv_map_stacked = torch.tensor([[]]) # 26,11,9,7 (size of conv layers)

        if self.gaze_loss_type is not None and use_gaze:
            # sum over all dimensions of the conv map
            gaze_conv = x4
            conv_map = gaze_conv

            # 1x1 convolution followed by softmax to get collapsed and normalized conv output
            norm_operator = nn.Conv2d(16, 1, kernel_size=1, stride=1)
            if torch.cuda.is_available():
                #print("Initializing Cuda Nets...")
                norm_operator.cuda()
            attn_map = norm_operator(torch.squeeze(conv_map))


            conv_map_traj.append(attn_map)
            conv_map_stacked = torch.stack(conv_map_traj)

        return sum_rewards, sum_abs_rewards, conv_map_stacked

    def forward(self, traj_i, traj_j, gaze_conv_layer=0):
        '''compute cumulative return for each trajectory and return logits'''
        cum_r_i, abs_r_i, conv_map_i = self.cum_return(traj_i, gaze_conv_layer)
        cum_r_j, abs_r_j, conv_map_j = self.cum_return(traj_j, gaze_conv_layer)
        return torch.cat((cum_r_i.unsqueeze(0), cum_r_j.unsqueeze(0)), 0), abs_r_i + abs_r_j, conv_map_i, conv_map_j

    def single_forward(self, batch):
        '''compute cumulative return for each trajectory and return logits'''
        x = batch.permute(0, 3, 1, 2)  # get into NCHW format
        # compute forward pass of reward network (we parallelize across frames)
        x1 = F.leaky_relu(self.conv1(x))
        x2 = F.leaky_relu(self.conv2(x1))
        x3 = F.leaky_relu(self.conv3(x2))
        x4 = F.leaky_relu(self.conv4(x3))
        x = x4.view(-1, 784)
        x = F.leaky_relu(self.fc1(x))
        r = self.fc2(x)
        return r