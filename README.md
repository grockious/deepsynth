Note: the current codebase is under code refactoring and a final package will be released upon AAAI proceedings publication.  

# DeepSynth
DeepSynth is a general method for effective training of deep Reinforcement Learning (RL) agents when the reward is sparse and non-Markovian, but at the same time progress towards the reward requires achieving an unknown sequence of high-level objectives. The framework uses human-interpretable automata, synthesised from trace data generated through exploration of the environment by the deep RL agent to uncover this sequential structure.

## Publications
* Hasanbeig, M. , Jeppu, N. Y., Abate, A., Melham, T., Kroening, D., "DeepSynth: Automata Synthesis for Automatic Task Segmentation in Deep Reinforcement Learning", AAAI Conference on Artificial Intelligence, 2021. [[PDF]](https://arxiv.org/pdf/1911.10244.pdf)

## Installation

Navigate to the folder you would like to install DeepSynth in, and clone this repository with its Python dependencies by:
~~~
git clone https://github.com/grockious/deepsynth.git
cd deepsynth
pip3 install .
~~~
DeepSynth requires [CBMC](https://www.cprover.org/cbmc/) for automata synthesis, please follow the installation instructions on [Trace2Model](https://github.com/natasha-jeppu/Trace2Model).

## Usage
#### Training an RL agent:
In each benchmark directory run `learner.py`. For instance,
```
python3 montezuma/learner.py
```

## Reference
Please use this bibtex entry if you want to cite this repository in your publication:

```
@misc{deepsynth_repo,
  author = {Hasanbeig, Mohammadhosein and Jeppu, Natasha Yogananda and Abate, Alessandro and Melham, Tom and Kroening, Daniel},
  title = {DeepSynth: Automata Synthesis for Automatic Task Segmentation in Deep Reinforcement Learning Code Repository},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/grockious/deepsynth}},
}
```

## License
This project is licensed under the terms of the [MIT License](/LICENSE)


