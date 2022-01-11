# Gaze augmented Behavioral Cloning #

This code has been tested with Keras 2.1.5 and Tensorflow 1.8.0.


## Train agent policies
Learn the agent policy for the baseline behavioral cloning network:
```
python bc.py [demo_tar_file] [demo_txt_file]
```

Learn the agent policy for the baseline AGIL network:
```
python human_gaze.py [demo_tar_file] [demo_txt_file] [gaze_hdf5_file]
python agil.py [demo_tar_file] [demo_txt_file] [gaze_npz_file]
```

Learn the agent policy for the CGL network:
```
python human_gaze.py [demo_tar_file] [demo_txt_file] [gaze_hdf5_file]
python cgl_bc.py [demo_tar_file] [demo_txt_file] [gaze_npz_file]
```
