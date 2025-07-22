[//]: # (Image References)
[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42386929-76f671f0-8106-11e8-9376-f17da2ae852e.png "Kernel"

This is my submission for Project 1 of Udacity's Deep Reinforcement Learning Nanodegree, Value-Based Methods.

# Environment Details

The environment is a Unity ML Agents environment. It is a a large, square world with yellow bananas and blue bananas:

![Trained Agent][image1]

## Reward Structure
A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. The goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

## State and Action Spaces
The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

## Completion Criteria
The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

# Getting Started

This section will provide instructions on how to setup the repository code. It is tested in a Linux environment.

1. Run the following commands to download and extract the Unity ML Agents environment:
```bash
cd ./p1_navigation
wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip
unzip Banana_Linux.zip
rm Banana_Linux.zip
cd ..
```

2. Create (and activate) a new environment with Python 3.6.
```bash
conda create --name drlnd python=3.6
source activate drlnd
```
	
3. Install the python dependencies into the actiavted `conda` environment:
```bash
cd ./python
pip install .
```

4. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

5. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu. 

![Kernel][image2]


# Instructions

## Running the Training Code
To train the agent, make sure the `conda` environment is activated (if it isn't, run `source activate drlnd`), and that you are in the root of the repository. Then:

1. Navigate into the `p1_navigation` folder with: `cd ./p1_navigation` 
2. Run the training script: `python main.py`

If the environment gets solved, the model weights will get saved in `p1_navigation/checkpoint.pth`, and you will see a simulation of the trained agent.

## Report
The details of the successfully trained agent and the learning algorithm can be found in `Report.ipynb`.

## Watch Trained Agent
To watch the trained agent:

1. Run `jupyter notebook`
2. Run the `p1_navigation/Watch Trained Agent.ipynb` Jupyter notebook.