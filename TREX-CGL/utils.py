import numpy as np
import cv2
import csv
import os
import torch
from os import path, listdir
from gaze import gaze_heatmap as gh
import time

from baselines.common.trex_utils import preprocess

cv2.ocl.setUseOpenCL(False)


def normalize_state(obs):
    return obs / 255.0


def normalize(obs, max_val):
    if(max_val != 0):
        norm_map = obs/float(max_val)
    else:
        norm_map = obs
    return norm_map


# need to grayscale and warp to 84x84
def GrayScaleWarpImage(image):
    """Warp frames to 84x84 as done in the Nature paper and later work."""
    width = 84
    height = 84
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return frame


def MaxSkipAndWarpFrames(trajectory_dir, img_dirs, frames):
    """take a trajectory file of frames and max over every 3rd and 4th observation"""
    num_frames = len(frames)
    skip = 4

    sample_pic = np.random.choice(
        listdir(path.join(trajectory_dir, img_dirs[0])))
    image_path = path.join(trajectory_dir, img_dirs[0], sample_pic)
    pic = cv2.imread(image_path)
    obs_buffer = np.zeros((2,)+pic.shape, dtype=np.uint8)
    max_frames = []
    for i in range(num_frames):
        img_name = frames[i] + ".png"
        img_dir = img_dirs[i]

        if i % skip == skip - 2:
            obs = cv2.imread(path.join(trajectory_dir, img_dir, img_name))

            obs_buffer[0] = obs
        if i % skip == skip - 1:
            obs = cv2.imread(path.join(trajectory_dir, img_dir, img_name))
            obs_buffer[1] = obs

            # warp max to 80x80 grayscale
            image = obs_buffer.max(axis=0)
            warped = GrayScaleWarpImage(image)
            max_frames.append(warped)
    return max_frames


def StackFrames(frames):
    import copy
    """stack every four frames to make an observation (84,84,4)"""
    stacked = []
    stacked_obs = np.zeros((84, 84, 4))
    for i in range(len(frames)):
        if i >= 3:
            stacked_obs[:, :, 0] = frames[i-3]
            stacked_obs[:, :, 1] = frames[i-2]
            stacked_obs[:, :, 2] = frames[i-1]
            stacked_obs[:, :, 3] = frames[i]
            stacked.append(np.expand_dims(copy.deepcopy(stacked_obs), 0))
    return stacked

def MaxSkipReward(rewards):
    """take a list of rewards and max over every 3rd and 4th observation"""
    num_frames = len(rewards)
    skip = 4
    max_frames = []
    obs_buffer = np.zeros((2,))
    for i in range(num_frames):
        r = rewards[i]
        if i % skip == skip - 2:

            obs_buffer[0] = r
        if i % skip == skip - 1:

            obs_buffer[1] = r
            rew = obs_buffer.max(axis=0)
            max_frames.append(rew)
    return max_frames


def StackReward(rewards):
    import copy
    """combine every four frames to make an observation"""
    stacked = []
    stacked_obs = np.zeros((1,))
    for i in range(len(rewards)):
        if i >= 3:
            # Sum over the rewards across four frames
            stacked_obs = rewards[i-3]
            stacked_obs = stacked_obs + rewards[i-2]
            stacked_obs = stacked_obs + rewards[i-1]
            stacked_obs = stacked_obs + rewards[i]

            stacked.append(np.expand_dims(copy.deepcopy(stacked_obs), 0))
    return stacked


def get_sorted_traj_indices(env_name, dataset):
    # need to pick out a subset of demonstrations based on desired performance
    # first let's sort the demos by performance, we can use the trajectory number to index into the demos so just
    # need to sort indices based on 'score'
    game = env_name
    traj_indices = []
    traj_scores = []
    traj_dirs = []
    traj_rewards = []
    traj_gaze = []
    traj_frames = []
    print('traj length: ', len(dataset.trajectories[game]))
    for t in dataset.trajectories[game]:
        traj_indices.append(t)
        traj_scores.append(dataset.trajectories[game][t][-1]['score'])
        # a separate img_dir defined for every frame of the trajectory as two different trials could comprise an episode
        traj_dirs.append([dataset.trajectories[game][t][i]['img_dir']
                          for i in range(len(dataset.trajectories[game][t]))])
        traj_rewards.append([dataset.trajectories[game][t][i]['reward']
                             for i in range(len(dataset.trajectories[game][t]))])
        traj_gaze.append([dataset.trajectories[game][t][i]['gaze_positions']
                          for i in range(len(dataset.trajectories[game][t]))])
        traj_frames.append([dataset.trajectories[game][t][i]['frame']
                            for i in range(len(dataset.trajectories[game][t]))])

    sorted_traj_indices = [x for _, x in sorted(
        zip(traj_scores, traj_indices), key=lambda pair: pair[0])]
    sorted_traj_scores = sorted(traj_scores)
    sorted_traj_dirs = [x for _, x in sorted(
        zip(traj_scores, traj_dirs), key=lambda pair: pair[0])]
    sorted_traj_rewards = [x for _, x in sorted(
        zip(traj_scores, traj_rewards), key=lambda pair: pair[0])]
    sorted_traj_gaze = [x for _, x in sorted(
        zip(traj_scores, traj_gaze), key=lambda pair: pair[0])]
    sorted_traj_frames = [x for _, x in sorted(
        zip(traj_scores, traj_frames), key=lambda pair: pair[0])]

    print("Max human score", max(sorted_traj_scores))
    print("Min human score", min(sorted_traj_scores))

    # so how do we want to get demos? how many do we have if we remove duplicates?
    seen_scores = set()
    non_duplicates = []
    for i, s, d, r, g, f in zip(sorted_traj_indices, sorted_traj_scores, sorted_traj_dirs, sorted_traj_rewards, sorted_traj_gaze, sorted_traj_frames):
        if s not in seen_scores:
            seen_scores.add(s)
            non_duplicates.append((i, s, d, r, g, f))
    print("num non duplicate scores", len(seen_scores))
    if env_name == "spaceinvaders":
        start = 0
        skip = 3
    elif env_name == "revenge":
        start = 0
        skip = 1
    elif env_name == "qbert":
        start = 0
        skip = 3
    elif env_name == "mspacman":
        start = 0
        skip = 1
    else:   
        start = 0
        skip = 3
    num_demos = 12

    demos = non_duplicates  # don't skip any demos
    return demos


def get_preprocessed_trajectories(env_name, dataset, data_dir):
    """returns an array of trajectories corresponding to what you would get running checkpoints from PPO
       demonstrations are grayscaled, maxpooled, stacks of 4 with normalized values between 0 and 1 and
       top section of screen is masked
    """

    demos = get_sorted_traj_indices(env_name, dataset)
    human_scores = []
    human_demos = []
    human_rewards = []
    human_gaze = []

    print('len demos: ', len(demos))
    for indx, score, img_dir, rew, gaze, frame in demos:
        human_scores.append(score)

        traj_dir = path.join(data_dir, env_name)
        maxed_traj = MaxSkipAndWarpFrames(traj_dir, img_dir, frame)
        stacked_traj = StackFrames(maxed_traj)

        demo_norm_mask = []
        # normalize values to be between 0 and 1 and have top part masked
        for ob in stacked_traj:
            demo_norm_mask.append(preprocess(ob, env_name)[0])  # masking
        human_demos.append(demo_norm_mask)

        # skip and stack reward
        maxed_reward = MaxSkipReward(rew)
        stacked_reward = StackReward(maxed_reward)
        human_rewards.append(stacked_reward)

    return human_demos, human_scores, human_rewards

