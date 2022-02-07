# IL-CGL: Imitation Learning Agents guided by a Coverage-based Gaze Loss (CGL)

This respository contains code for the following [paper](https://arxiv.org/pdf/2002.12500.pdf):
> A. Saran, R. Zhang, E.S. Short, and S. Niekum. 
Efficiently Guiding Imitation Learning Agents with Human Gaze.
International Conference on Autonomous Agents and Multiagent Systems (AAMAS), May 2021. 

## Data

* Download the demonstrations and corresponding gaze data from the [Atari-HEAD dataset](https://zenodo.org/record/3451402). Store the data for each game under a separate subdirectory inside the `data/atari-head/` folder.
* Download the pretrained [gaze models](https://drive.google.com/file/d/1tSQsMpPt354r69-ebHSQwQ3mjxkzKL6v/view?usp=sharing) (trained on all demos except the high-score ones for each game) and store them inside the `data/trained_gaze_models/` folder.
* Note that you can re-train human gaze prediction models using the network architecture provided in `BC-CGL/human_gaze.py` if needed.

## Code

Please see further instructions to train gaze-augmented models for:
* Behavioral Cloning in the `BC-CGL` folder
* Trajectory-ranked reward extrapolation in the `TREX-CGL` folder

## Bibliography

If you find this repository is useful in your research, please cite the paper:
```
@article{saran2020efficiently,
  title={Efficiently Guiding Imitation Learning Agents with Human Gaze},
  author={Saran, Akanksha and Zhang, Ruohan and Short, Elaine Schaertl and Niekum, Scott},
  booktitle={International Conference on Autonomous Agents and Multiagent Systems (AAMAS)},
  year={2021},
  organization={IFAAMAS}
}
```
