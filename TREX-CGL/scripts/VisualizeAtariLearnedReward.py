
# coding: utf-8

import pickle
import gym
import time
import numpy as np
import random
import torch
from run_test import *
import matplotlib.pylab as plt
import argparse
from cnn import Net

parser = argparse.ArgumentParser(description=None)
parser.add_argument('--env_name', default='', help='Select the environment name to run, i.e. pong')
parser.add_argument('--reward_net_path', default='', help="name and location for learned model params")
parser.add_argument('--seed', default=0, help="random seed for experiments")
parser.add_argument('--save_fig_dir', help ="where to save visualizations")
parser.add_argument('--data_dir', default='../../atari-head/', help='path to data directory with demonstrations')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args = parser.parse_args()
env_name = args.env_name
save_fig_dir = args.save_fig_dir

if env_name == "spaceinvaders":
    env_id = "SpaceInvadersNoFrameskip-v4"
elif env_name == "mspacman":
    env_id = "MsPacmanNoFrameskip-v4"
elif env_name == "videopinball":
    env_id = "VideoPinballNoFrameskip-v4"
elif env_name == "beamrider":
    env_id = "BeamRiderNoFrameskip-v4"
else:
    env_id = env_name[0].upper() + env_name[1:] + "NoFrameskip-v4"
env_type = "atari"

seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)

print(env_id)

stochastic = True

reward_net_path = args.reward_net_path


env = make_vec_env(env_id, 'atari', 1, 0,
                   wrapper_kwargs={
                       'clip_rewards':False,
                       'episode_life':False,
                   })


env = VecFrameStack(env, 4)
agent = PPO2Agent(env, env_type, stochastic)

import torch
import torch.nn as nn
import torch.nn.functional as F


if torch.cuda.is_available():
    map_location=lambda storage, loc: storage.cuda()
else:
    map_location='cpu'
print(map_location, device)
reward = Net(gaze_loss_type=None)
reward.to(device)
reward.load_state_dict(torch.load(reward_net_path,map_location=map_location))


from gaze import human_utils
from gaze import atari_head_dataset as ahd
# get real human demos

dataset = ahd.AtariHeadDataset(args.env_name, args.data_dir)
demonstrations, learning_returns, learning_rewards = human_utils.get_preprocessed_trajectories(env_name, dataset, args.data_dir)



#plot extrapolation curves
def convert_range(x,minimum, maximum,a,b):
    return (x - minimum)/(maximum - minimum) * (b - a) + a


#search for min and max predicted reward observations
min_reward = 100000
max_reward = -100000
cnt = 0
with torch.no_grad():
    for d in demonstrations:
        # print(cnt)
        cnt += 1

        for i,s in enumerate(d[2:-1]):
            r = reward.cum_return(torch.from_numpy(s[0]).float())[0].float().to(device).item()

            if r < min_reward:
                min_reward = r
                min_frame = s[0]
                min_frame_i = i+2
            elif r > max_reward:
                max_reward = r
                max_frame = s[0]
                max_frame_i = i+2


def mask_coord(i,j,frames, mask_size, channel):
    #takes in i,j pixel and stacked frames to mask
    masked = frames.copy()
    masked[i:i+mask_size,j:j+mask_size,channel] = 0
    return masked

def gen_attention_maps(frames, mask_size):

    orig_frame = frames
    batch,height,width,channels = orig_frame.shape

    #find reward without any masking once
    r_before = reward.cum_return(torch.from_numpy(orig_frame).float())[0].float().to(device).item()
    heat_maps = []
    for c in range(4): #four stacked frame channels
        delta_heat = np.zeros((height, width))
        for i in range(height-mask_size):
            for j in range(width - mask_size):
                #get masked frames
                masked_ij = mask_coord(i,j,orig_frame, mask_size, c)
                r_after = r = reward.cum_return(torch.from_numpy(masked_ij).float())[0].float().to(device).item()
                r_delta = abs(r_after - r_before)
                #save to heatmap
                delta_heat[i:i+mask_size, j:j+mask_size] += r_delta
        heat_maps.append(delta_heat)
    return heat_maps



#plot heatmap
mask_size = 3
delta_heat_max = gen_attention_maps(max_frame, mask_size)
delta_heat_min = gen_attention_maps(min_frame, mask_size)



plt.figure(5)
for cnt in range(4):
    plt.subplot(1,4,cnt+1)
    plt.imshow(delta_heat_max[cnt],cmap='seismic', interpolation='nearest')
    plt.axis('off')
plt.tight_layout()
plt.savefig(save_fig_dir + "/" + env_name + "max_attention.png", bbox_inches='tight')


plt.figure(6)
#print(max_frame_i)
#print(max_reward)
for cnt in range(4):
    plt.subplot(1,4,cnt+1)
    print(max_frame[0].shape)
    plt.imshow(max_frame[:,:,cnt])
    plt.axis('off')
plt.tight_layout()
plt.savefig(save_fig_dir + "/" + env_name + "max_frames.png", bbox_inches='tight')


plt.figure(7)
for cnt in range(4):
    plt.subplot(1,4,cnt+1)
    plt.imshow(delta_heat_min[cnt],cmap='seismic', interpolation='nearest')
    plt.axis('off')
plt.tight_layout()
plt.savefig(save_fig_dir + "/" + env_name + "min_attention.png", bbox_inches='tight')

# print(min_frame_i)
# print(min_reward)
plt.figure(8)
for cnt in range(4):
    plt.subplot(1,4,cnt+1)
    plt.imshow(min_frame[:,:,cnt])
    plt.axis('off')
plt.tight_layout()
plt.savefig(save_fig_dir + "/" + env_name + "min_frames.png", bbox_inches='tight')


#random frame heatmap
d_rand = np.random.randint(len(demonstrations))
f_rand = np.random.randint(len(demonstrations[d_rand]))
rand_frames = demonstrations[d_rand][f_rand][0]
print(rand_frames)

plt.figure(9)
for cnt in range(4):
    plt.subplot(1,4,cnt+1)
    plt.imshow(rand_frames[:,:,cnt])
    plt.axis('off')
plt.tight_layout()
plt.savefig(save_fig_dir + "/" + env_name + "random_frames.png", bbox_inches='tight')


delta_heat_rand = gen_attention_maps(rand_frames, mask_size)
plt.figure(10)
for cnt in range(4):
    plt.subplot(1,4,cnt+1)
    plt.imshow(delta_heat_rand[cnt],cmap='seismic', interpolation='nearest')
    plt.axis('off')
plt.tight_layout()
plt.savefig(save_fig_dir + "/" + env_name + "random_attention.png", bbox_inches='tight')