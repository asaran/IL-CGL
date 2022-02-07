# Gaze augmented Behavioral Cloning #

## Setup
This code has been tested with Tensorflow 2.6.0.

## Train agent policies
Learn the agent policy for the baseline behavioral cloning network:
```
python bc.py [demo_tar_file] [demo_txt_file]
```

Learn the agent policy for the baseline AGIL network:
```
python human_gaze.py [demo_tar_file] [demo_txt_file] [game_name] [gaze_hdf5_file]
python agil.py [demo_tar_file] [demo_txt_file] [gaze_npz_file]
```

Learn the agent policy for the CGL network:
```
python human_gaze.py [demo_tar_file] [demo_txt_file] [game_name] [gaze_hdf5_file]
python cgl_bc.py [demo_tar_file] [demo_txt_file] [gaze_npz_file]
```