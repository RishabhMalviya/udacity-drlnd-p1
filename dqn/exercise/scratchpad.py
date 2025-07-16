import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


env = gym.make('LunarLander-v2')


class DummyExperience:
    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done

experiences = []
for i in range(5):
    state, _ = env.reset()
    next_state, _ = env.reset()
    action = random.randint(0, env.action_space.n - 1)
    reward = random.uniform(-1.0, 1.0)
    done = random.choice([True, False])

    experiences.append(DummyExperience(state, action, reward, next_state, done))

states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
