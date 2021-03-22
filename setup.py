import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='deepsynth',
    version='0.0.2',
    author="Hosein Hasanbeig, Natasha Jeppu",
    author_email="hosein.hasanbeig@cs.ox.ac.uk, natasha.yogananda.jeppu@cs.ox.ac.uk",
    keywords='deep rl, logic, sparse reward, environment, agent',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/grockious/deepsynth',
    description='DeepSynth: Automata Synthesis for Automatic Task Segmentation in Deep Reinforcement Learning',
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
    install_requires=[
        'numpy',
        'cmake',
        'tensorflow>=2',
        'opencv-python>=4.4',
        'gym',
        'gym[atari]',
        'tensorflow>=2',
        'dill>=0.3.2',
        'imageio',
        'tqdm'
    ]
)
