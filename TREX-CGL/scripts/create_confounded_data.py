import cv2
import argparse
import os
import numpy as np

def mask_score(obs, env_name, scale):
    obs_copy = obs.copy()
    if env_name in ["space_invaders","breakout","pong","spaceinvaders"]:
        #takes a stack of four observations and blacks out (sets to zero) top n rows
        n = 10 * scale
        obs_copy[:,:n,:,:] = 0
    elif env_name in ["beamrider"]:
        n_top = 16 * scale
        n_bottom = 11 * scale
        obs_copy[:,:n_top,:,:] = 0
        obs_copy[:,-n_bottom:,:,:] = 0
    elif env_name in ["enduro","alien"]:
        n_top = 0
        n_bottom = 14 * scale
        obs_copy[:,:n_top,:,:] = 0
        obs_copy[:,-n_bottom:,:,:] = 0
    elif env_name in ["bank_heist"]:
        n_top = 0
        n_bottom = 13 * scale
        obs_copy[:,:n_top,:,:] = 0
        obs_copy[:,-n_bottom:,:,:] = 0
    elif env_name in ["centipede"]:
        n_top = 0
        n_bottom = 10 * scale
        obs_copy[:,:n_top,:,:] = 0
        obs_copy[:,-n_bottom:,:,:] = 0
    elif env_name in ["hero"]:
        n_top = 0
        n_bottom = 30 * scale
        obs_copy[:,:n_top,:,:] = 0
        obs_copy[:,-n_bottom:,:,:] = 0
    elif env_name == "qbert":
        n_top = 12 * scale
        obs_copy[:,:n_top,:,:] = 0
    elif env_name in ["seaquest"]:
        n_top = 12 * scale
        n_bottom = 16 * scale
        obs_copy[:,:n_top,:,:] = 0
        obs_copy[:,-n_bottom:,:,:] = 0
        #cuts out divers and oxygen
    elif env_name in ["mspacman","name_this_game"]:
        n_bottom = 15 * scale #mask score and number lives left
        obs_copy[:,-n_bottom:,:,:] = 0
    elif env_name in ["berzerk"]:
        n_bottom = 11* scale #mask score and number lives left
        obs_copy[:,-n_bottom:,:,:] = 0
    elif env_name in ["riverraid"]:
        n_bottom = 18* scale #mask score and number lives left
        obs_copy[:,-n_bottom:,:,:] = 0
    elif env_name in ["videopinball"]:
        n_top = 15* scale
        obs_copy[:,:n_top,:,:] = 0
    elif env_name in ["montezuma_revenge", "phoenix", "venture","road_runner"]:
        n_top = 10* scale
        obs_copy[:,:n_top,:,:] = 0
    elif env_name in ["asterix","demon_attack","freeway"]:
        n_top = 10* scale
        n_bottom = 10* scale
        obs_copy[:,:n_top,:,:] = 0
        obs_copy[:,-n_bottom:,:,:] = 0
    elif env_name in ["frostbite"]:
        n_top = 12* scale
        n_bottom = 10* scale
        obs_copy[:,:n_top,:,:] = 0
        obs_copy[:,-n_bottom:,:,:] = 0
    else:
        print("NOT MASKING SCORE FOR GAME: " + env_name)
        pass
    return obs_copy

def confound(data_dir, dest_dir, mask, env):
    # loop through all trials
    trials = [t for t in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir,t))]

    for t in trials:

        if not os.path.exists(os.path.join(dest_dir,t)):
            os.mkdir(os.path.join(dest_dir,t))

        # read txt file for action labels
        # action - index 5 in every row
        txt_file = os.path.join(data_dir,t+".txt")
        f = open(txt_file,"r")
        rows = f.readlines()
        

        # imgs in trial
        img_paths = [p for p in os.listdir(os.path.join(data_dir,t)) if p.endswith('.png')]
        img_prefix = '_'.join(img_paths[0].split('_')[:-1])

        action = '0'

        for i in range(len(img_paths)):
            img_path = img_prefix+'_'+str(i+1)+'.png'
            img = cv2.imread(os.path.join(data_dir,t,img_path))
            height, width = img.shape[0], img.shape[1]

            row = f.read()

            if i%16==0 and i>16:
                if ',' in rows[i+1]:
                    row = rows[i+1].split(',')
                    curr_action = row[5]
                    action = curr_action
            
            if mask:
                scale = int(np.ceil(img.shape[0]/84.0))
                img = mask_score(np.expand_dims(img,axis=0),env,scale)
                img = np.squeeze(img,axis=0)

            # overlay action text in white
            cv2.putText(img, action, (int(1/5*width),int(9/10*height)), cv2.FONT_HERSHEY_SIMPLEX, \
                fontScale=1, color=(255, 255, 255), thickness=3)

            # save modified data
            dest_img_name = os.path.join(dest_dir,t,img_path)
            cv2.imwrite(dest_img_name, img)
        


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--data_dir', default='./test_data/asterix/', help="folder to load test images")
    parser.add_argument('--dest_dir', default='./test_data/asterix_confounded/', help="path to save confounded images")
    parser.add_argument('--mask_scores', action='store_true', default=False, help="flag to mask image scores before overlaying actions")
    parser.add_argument('--env_name', default='asterix', help="game env")

    args = parser.parse_args()

    if not os.path.exists(args.dest_dir):
        os.mkdir(args.dest_dir)

    # create data with overlaid actions
    confound(args.data_dir, args.dest_dir, args.mask_scores, args.env_name)