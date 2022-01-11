import cv2
from scipy import misc
import copy
import gaze.misc_utils as MU
import gaze.input_utils as IU
import keras as K
import tensorflow as tf
import os
import sys
import json
import ipdb
from IPython import embed
from keras.models import Model, Sequential  # keras/engine/training.py
import keras.layers as L
import re
import numpy as np
import time
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os.path as path
import math
import torch

from astropy.convolution import convolve
from astropy.convolution.kernels import Gaussian2DKernel

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7

# creating ground truth gaze heatmap from gaze coordinates
class DatasetWithHeatmap:
    def __init__(self, GHmap=None, heatmap_shape=14):
        self.frameid2pos = None
        self.GHmap = GHmap  # GHmap means gaze heap map
        self.NUM_ACTION = 18
        self.xSCALE, self.ySCALE = 8, 4  # was 6,3
        self.SCR_W, self.SCR_H = 160*self.xSCALE, 210*self.ySCALE
        self.train_size = 10
        self.HEATMAP_SHAPE = heatmap_shape
        self.sigmaH = 28.50 * self.HEATMAP_SHAPE / self.SCR_H
        self.sigmaW = 44.58 * self.HEATMAP_SHAPE / self.SCR_W

    def createGazeHeatmap(self, gaze_coords, heatmap_shape, viz=False, asc=False, asc_file=''):
        print('gaze_coords length: ', len(gaze_coords))
        if not asc:
            self.frameid2pos = self.get_gaze_data(gaze_coords)
        else:
            self.frameid2pos, _, _, _, _ = self.read_gaze_data_asc_file(
                asc_file)

        self.train_size = len(self.frameid2pos.keys())
        self.HEATMAP_SHAPE = heatmap_shape
        self.sigmaH = 28.50 * self.HEATMAP_SHAPE / self.SCR_H
        self.sigmaW = 44.58 * self.HEATMAP_SHAPE / self.SCR_W

        self.GHmap = np.zeros(
            [self.train_size, self.HEATMAP_SHAPE, self.HEATMAP_SHAPE, 1], dtype=np.float32)

        t1 = time.time()

        bad_count, tot_count = 0, 0
        for (i, fid) in enumerate(self.frameid2pos.keys()):
            tot_count += len(self.frameid2pos[fid])
            if asc:
                bad_count += self.convert_gaze_pos_to_heap_map(
                    self.frameid2pos[fid], out=self.GHmap[i])
            else:
                bad_count += self.convert_gaze_coords_to_heap_map(
                    self.frameid2pos[fid], out=self.GHmap[i])

        # print("Bad gaze (x,y) sample: %d (%.2f%%, total gaze sample: %d)" % (bad_count, 100*float(bad_count)/tot_count, tot_count))
        # print("'Bad' means the gaze position is outside the 160*210 screen")

        self.GHmap = self.preprocess_gaze_heatmap(0).astype(np.float32)

        # normalizing such that values range between 0 and 1
        for i in range(len(self.GHmap)):
            max_val = self.GHmap[i].max()
            min_val = self.GHmap[i].min()
            if max_val != min_val:
                self.GHmap[i] = (self.GHmap[i] - min_val)/(max_val - min_val)

            if viz:
                cv2.imwrite('gazeGT/viz/'+str(i)+'.png', self.GHmap[i]*255)

        return self.GHmap

    def get_gaze_data(self, gaze_coords):
        frameid2pos = {}
        frame_id = 0
        for gaze_list in gaze_coords:
            isnan = [x for x in gaze_list if math.isnan(x)]
            if len(isnan) > 0:
                frameid2pos[frame_id] = []
                frame_id += 1
                continue
            gaze_xy_list = []
            for i in range(0, len(gaze_list), 2):
                x, y = gaze_list[i]*self.xSCALE, gaze_list[i+1]*self.ySCALE
                gaze_xy_list.append((x, y))
            frameid2pos[frame_id] = gaze_xy_list
            frame_id += 1

        if len(frameid2pos) < 1000:  # simple sanity check
            print("Warning: did you provide the correct gaze data? Because the data for only %d frames is detected" % (
                len(frameid2pos)))

        few_cnt = 0
        for v in frameid2pos.values():
            if len(v) < 10:
                few_cnt += 1
        # print ("Warning:  %d frames have less than 10 gaze samples. (%.1f%%, total frame: %d)" % \

        return frameid2pos

    def read_gaze_data_asc_file(self, fname):
        """ This function reads a ASC file and returns 
            a dictionary mapping frame ID to a list of gaze positions,
            a dictionary mapping frame ID to action """

        with open(fname, 'r') as f:
            lines = f.readlines()
        frameid, xpos, ypos = "BEFORE-FIRST-FRAME", None, None
        frameid2pos = {frameid: []}
        frameid2action = {frameid: None}
        frameid2duration = {frameid: None}
        frameid2unclipped_reward = {frameid: None}
        frameid2episode = {frameid: None}
        start_timestamp = 0
        scr_msg = re.compile(
            r"MSG\s+(\d+)\s+SCR_RECORDER FRAMEID (\d+) UTID (\w+)")
        freg = r"[-+]?[0-9]*\.?[0-9]+"  # regex for floating point numbers
        gaze_msg = re.compile(r"(\d+)\s+(%s)\s+(%s)" % (freg, freg))
        act_msg = re.compile(r"MSG\s+(\d+)\s+key_pressed atari_action (\d+)")
        reward_msg = re.compile(r"MSG\s+(\d+)\s+reward (\d+)")
        episode_msg = re.compile(r"MSG\s+(\d+)\s+episode (\d+)")

        for (i, line) in enumerate(lines):
            match_sample = gaze_msg.match(line)
            if match_sample:
                timestamp, xpos, ypos = match_sample.group(
                    1), match_sample.group(2), match_sample.group(3)
                xpos, ypos = float(xpos), float(ypos)
                frameid2pos[frameid].append((xpos, ypos))
                continue

            match_scr_msg = scr_msg.match(line)
            if match_scr_msg:  # when a new id is encountered
                old_frameid = frameid
                timestamp, frameid, UTID = match_scr_msg.group(
                    1), match_scr_msg.group(2), match_scr_msg.group(3)
                frameid2duration[old_frameid] = int(
                    timestamp) - start_timestamp
                start_timestamp = int(timestamp)
                frameid = self.make_unique_frame_id(UTID, frameid)
                frameid2pos[frameid] = []
                frameid2action[frameid] = None
                continue

            match_action = act_msg.match(line)
            if match_action:
                timestamp, action_label = match_action.group(
                    1), match_action.group(2)
                if frameid2action[frameid] is None:
                    frameid2action[frameid] = int(action_label)
                else:
                    print("Warning: there is more than 1 action for frame id %s. Not supposed to happen." % str(
                        frameid))
                continue

            match_reward = reward_msg.match(line)
            if match_reward:
                timestamp, reward = match_reward.group(
                    1), match_reward.group(2)
                if frameid not in frameid2unclipped_reward:
                    frameid2unclipped_reward[frameid] = int(reward)
                else:
                    print("Warning: there is more than 1 reward for frame id %s. Not supposed to happen." % str(
                        frameid))
                continue

            match_episode = episode_msg.match(line)
            if match_episode:
                timestamp, episode = match_episode.group(
                    1), match_episode.group(2)
                assert frameid not in frameid2episode, "ERROR: there is more than 1 episode for frame id %s. Not supposed to happen." % str(
                    frameid)
                frameid2episode[frameid] = int(episode)
                continue

        # throw out gazes after the last frame, because the game has ended but eye tracker keeps recording
        frameid2pos[frameid] = []

        if len(frameid2pos) < 1000:  # simple sanity check
            print("Warning: did you provide the correct ASC file? Because the data for only %d frames is detected" % (
                len(frameid2pos)))
            raw_input("Press any key to continue")

        few_cnt = 0
        for v in frameid2pos.values():
            if len(v) < 10:
                few_cnt += 1
        print("Warning:  %d frames have less than 10 gaze samples. (%.1f%%, total frame: %d)" %
              (few_cnt, 100.0*few_cnt/len(frameid2pos), len(frameid2pos)))
        return frameid2pos, frameid2action, frameid2duration, frameid2unclipped_reward, frameid2episode

    # bg_prob_density seems to hurt accuracy. Better set it to 0
    def preprocess_gaze_heatmap(self, bg_prob_density, debug_plot_result=False):
        from scipy.stats import multivariate_normal
        import tensorflow as tf
        # don't move this to the top, as people who import this file might not have keras or tf
        import keras as K

        model = K.models.Sequential()
        model.add(K.layers.Lambda(lambda x: x+bg_prob_density,
                                  input_shape=(self.GHmap.shape[1], self.GHmap.shape[2], 1)))

        if self.sigmaH > 1 and self.sigmaW > 1:  # was 0,0; if too small, dont blur
            lh, lw = int(4*self.sigmaH), int(4*self.sigmaW)
            # so the kernel size is [lh*2+1,lw*2+1]
            x, y = np.mgrid[-lh:lh+1:1, -lw:lw+1:1]
            pos = np.dstack((x, y))
            gkernel = multivariate_normal.pdf(
                pos, mean=[0, 0], cov=[[self.sigmaH*self.sigmaH, 0], [0, self.sigmaW*self.sigmaW]])
            assert gkernel.sum() > 0.95, "Simple sanity check: prob density should add up to nearly 1.0"

            model.add(K.layers.Lambda(lambda x: tf.pad(
                x, [(0, 0), (lh, lh), (lw, lw), (0, 0)], 'REFLECT')))
            # print(gkernel.shape, sigmaH, sigmaW)
            model.add(K.layers.Conv2D(1, kernel_size=gkernel.shape, strides=1, padding="valid", use_bias=False,
                                      activation="linear", kernel_initializer=K.initializers.Constant(gkernel)))
        else:
            print("WARNING: Gaussian filter's sigma is 0, i.e. no blur.")

        model.compile(optimizer='rmsprop',  # not used
                      loss='categorical_crossentropy',  # not used
                      metrics=None)

        output = model.predict(self.GHmap, batch_size=500)

        if debug_plot_result:
            print(r"""debug_plot_result is True. Entering IPython console. You can run:
                    %matplotlib
                    import matplotlib.pyplot as plt
                    f, axarr = plt.subplots(1,2)
                    axarr[0].imshow(gkernel)
                    rnd=np.random.randint(output.shape[0]); print "rand idx:", rnd
                    axarr[1].imshow(output[rnd,...,0])""")
            embed()

        shape_before, shape_after = self.GHmap.shape, output.shape
        assert shape_before == shape_after, """
        Simple sanity check: shape changed after preprocessing. 
        Your preprocessing code might be wrong. Check the shape of output tensor of your tensorflow code above"""
        return output

    def make_unique_frame_id(self, UTID, frameid):
        return (hash(UTID), int(frameid))

    def convert_gaze_coords_to_heap_map(self, gaze_pos_list, out):
        h, w = out.shape[0], out.shape[1]
        bad_count = 0
        if(not np.isnan(gaze_pos_list).all()):
            for j in range(0, len(gaze_pos_list)):
                x = gaze_pos_list[j][0]
                y = gaze_pos_list[j][1]
                try:
                    out[int(y/self.SCR_H*h), int(x/self.SCR_W*w)] += 1
                except IndexError:  # the computed X,Y position is not in the gaze heat map
                    bad_count += 1
        return bad_count

    def convert_gaze_pos_to_heap_map(self, gaze_pos_list, out):
        h, w = out.shape[0], out.shape[1]
        bad_count = 0
        for (x, y) in gaze_pos_list:
            try:
                out[int(y/self.SCR_H*h), int(x/self.SCR_W*w)] += 1
            except IndexError:  # the computed X,Y position is not in the gaze heat map
                bad_count += 1
        return bad_count


# pretrained gaze heat maps
class PretrainedHeatmap():
    def __init__(self, game_name, model_path, meanfile_path):

        # pretrained gaze model  (novice/experts)
        self.model_path = model_path
        

        self.dropout_prob = 0.5  # float(sys.argv[3])
        self.heatmap_shape = 84
        k = 4
        # height * width * channel This cannot read from file and needs to be provided here
        self.SHAPE = (84, 84, k)

        self.meanfile = meanfile_path
        self.mean = np.load(self.meanfile)  # shape: #(1,)

        MU.BMU.save_GPU_mem_keras()
        MU.keras_model_serialization_bug_fix()
        self.model_arch()

    def model_arch(self):
        ###############################
        # Architecture of the network #
        ###############################

        inputs = L.Input(shape=self.SHAPE)
        x = inputs  # inputs is used by the line "Model(inputs, ... )" below

        conv1 = L.Conv2D(32, (8, 8), strides=4, padding='valid')
        x = conv1(x)
        # print(conv1.output_shape)
        x = L.Activation('relu')(x)
        x = L.BatchNormalization()(x)
        x = L.Dropout(self.dropout_prob)(x)

        conv2 = L.Conv2D(64, (4, 4), strides=2, padding='valid')
        x = conv2(x)
        # print(conv2.output_shape)
        x = L.Activation('relu')(x)
        x = L.BatchNormalization()(x)
        x = L.Dropout(self.dropout_prob)(x)

        conv3 = L.Conv2D(64, (3, 3), strides=1, padding='valid')
        x = conv3(x)
        # print(conv3.output_shape)
        x = L.Activation('relu')(x)
        x = L.BatchNormalization()(x)
        x = L.Dropout(self.dropout_prob)(x)

        deconv1 = L.Conv2DTranspose(64, (3, 3), strides=1, padding='valid')
        x = deconv1(x)
        # print(deconv1.output_shape)
        x = L.Activation('relu')(x)
        x = L.BatchNormalization()(x)
        x = L.Dropout(self.dropout_prob)(x)

        deconv2 = L.Conv2DTranspose(32, (4, 4), strides=2, padding='valid')
        x = deconv2(x)
        # print(deconv2.output_shape)
        x = L.Activation('relu')(x)
        x = L.BatchNormalization()(x)
        x = L.Dropout(self.dropout_prob)(x)

        deconv3 = L.Conv2DTranspose(1, (8, 8), strides=4, padding='valid')
        x = deconv3(x)
        # print(deconv3.output_shape)

        outputs = L.Activation(MU.my_softmax)(x)

        self.model = Model(inputs=inputs, outputs=outputs)
        opt = K.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
        self.model.compile(loss=MU.my_kld, optimizer=opt, metrics=[MU.NSS])
        self.model.load_weights(self.model_path)

    def normalize(self, obs):
        # DONE: use softmax here instead!!

        H, W = obs.shape[0], obs.shape[1]
        obs = torch.tensor(obs)
        norm_map = F.softmax(obs.view(1,-1), dim=1).view(H,W)
        norm_map = norm_map.cpu().detach().numpy()
        # max_val, min_val = np.max(obs), np.min(obs)
        # if(max_val-min_val != 0):
        #     norm_map = (obs-min_val)/(max_val-min_val)
        # else:
        #     norm_map = obs
        return norm_map

    def get_heatmap(self, stacked_img, shape, viz=False):
        traj_length = len(stacked_img)
        stacked_obs = np.zeros((traj_length, 84, 84, 4))
        for i in range(traj_length):
            # convert stacked frame list (for a trajectory) into a batch
            img_np = stacked_img[i].squeeze()  # (1,84,84,4) --> (84,84,4)
            img_np = img_np.astype(np.float32) #/ 255.0 # already normalized
            img_np -= self.mean
            stacked_obs[i, :, :, :] = img_np.squeeze()

        val_predict = self.model.predict(stacked_obs)
        
        
        output = np.zeros((traj_length, shape, shape))
        for i in range(traj_length):
            heatmap = val_predict[i, :, :].squeeze()
            #convolve heatmap with gaussian filter
            heatmap = convolve(heatmap, Gaussian2DKernel(x_stddev=1))

            # Normalize predicted heatmap such that largest element is 1.0
            heatmap_norm = self.normalize(heatmap)

            # subsample to desired shape for heatmap (TODO: meanpooling)
            if shape!=84:
                print('reshaping predicted heatmap: ', shape)
                output[i, :, :] = self.normalize(misc.imresize(
                    heatmap_norm, [shape, shape], interp='bilinear'))
            else:
                output[i, :, :] = heatmap_norm
            if viz:
                cv2.imwrite('gazeNetwork/viz/'+str(i)+'_out.png',
                            output[i, :, :].squeeze()*255)
                cv2.imwrite('gazeNetwork/viz/'+str(i) +
                            '_pred.png', heatmap_norm*255)

        # return np.expand_dims(output, axis=3)
        return output
