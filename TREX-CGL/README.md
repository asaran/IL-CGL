# Gaze augmented T-REX #

The code for this section is built upon the [T-REX implementation](https://github.com/hiwonjoon/ICML2019-TREX). It has been tested with PyTorch 1.5.0, Tensorflow 1.12.0, Keras 2.2.4, h5py 2.10.0. 


## T-REX reward learning for Atari ##

Learn the reward model for the baseline T-REX model. 
```
python LearnReward.py --env_name breakout --reward_model_path ./reward_models/breakout.params --models_dir . --data_dir ../data/atari-head/
```

Learn the reward model for the gaze augmented T-REX model (T-REX+CGL). The hyperparameter `gaze_reg` is used to tune the balance between the pairwise ranking loss and the coverage loss.
```
python LearnReward.py --env_name breakout --reward_model_path ./reward_models/breakout_cgl.params --models_dir . --data_dir ../data/atari-head/ --cgl --gaze_reg 0.5
```


## RL on learned reward function ##

Given a trained reward network you can run RL as follows:

First baselines must be installed

```
cd baselines
pip install -e .
```

Then you can run RL as follows:

```
OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=[your_log_dir_here] python -m baselines.run --alg=ppo2 --env=[Atari env here] --custom_reward pytorch --custom_reward_path [path_to_learned_reward_model] --seed 0 --num_timesteps=5e7  --save_interval=500 --num_env 9
```


For example

```
OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=/home/tflogs python -m baselines.run --alg=ppo2 --env=BreakoutNoFrameskip-v4 --custom_reward pytorch --custom_reward_path learned_models/breakout.params --seed 0 --num_timesteps=5e7  --save_interval=500 --num_env 9
```



## Evaluation of learned policy ##

After training an RL agent use evaluateLearnedPolicy.py to evaluate the performance

For example:

```python evaluateLearnedPolicy.py --env_name breakout --checkpointpath [path_to_rl_checkpoint]```


