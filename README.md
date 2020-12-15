# DeepSynth
A framework for effective training of deep Reinforcement Learning (RL) agents when the reward is sparse and non-Markovian, but at the same time progress towards the reward requires achieving an unknown sequence of high-level objectives. The framework uses human-interpretable automata, synthesised from trace data generated through exploration of the environment by the deep RL agent to uncover this sequential structure.  <br/>

Based on work presented in 'DeepSynth: Automata Synthesis for Automatic Task Segmentation in Deep Reinforcement Learning' (https://arxiv.org/pdf/1911.10244.pdf)
~~~
@article{deepsynth,
title={{DeepSynth: Program Synthesis for Automatic Task Segmentation in Deep Reinforcement Learning}},
author={Hasanbeig, Mohammadhosein and Yogananda Jeppu, Natasha and Abate, Alessandro and Melham, Tom and Kroening, Daniel},
journal={arXiv preprint arXiv:1911.10244},
year={2019}
} 
~~~

Note that this is a first version of the codebase and will be updated soon.

## Requirements
- Python >= 3.5
- TensorFlow >= 2
- OpenCV >= 4.4
- OpenAI Gym [Atari]

## Training

To train DeepSynth agent on Montezuma’s Revenge run `training.py` in the root folder:<br>
~~~
cd deepsynth
python3 training.py
~~~


