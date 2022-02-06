# Gaze augmented Behavioral Cloning #

## Setup
This code has been tested with Tensorflow 2.6.0.

* Download the Atari-HEAD dataset: https://zenodo.org/record/3451402
* Store the trained models (.hdf5) and mean files (.npy) to predict human attention in a folder called `trained_gaze_models` under this directory. Note that you can re-train human gaze prediction models using the network architecture provided in `human_gaze.py` if needed.


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