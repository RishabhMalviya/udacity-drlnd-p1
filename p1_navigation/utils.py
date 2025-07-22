from collections import deque

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import numpy as np
import matplotlib.pyplot as plt



class EpsScheduler():
    def __init__(self, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
        self.EPS_START = eps_start
        self.EPS_END = eps_end
        self.EPS_DECAY = eps_decay
        
        self.curr_episode = 0

        self.curr_eps = self.EPS_START


    def get_eps(self, i_episode):
        self.curr_eps = max(self.EPS_END, self.curr_eps*(self.EPS_DECAY**(i_episode-self.curr_episode)))
        self.curr_episode = i_episode
        
        return self.curr_eps
    

def _check_solved(i_episode, scores_window, target=13.0) -> bool:
    print(f'Episode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}', end='\r')
    if i_episode % 100 == 0:
        print(f'Episode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}')
    
    if np.mean(scores_window)>=target:
        print(f'Environment solved in {i_episode-100} episodes!\tAverage Score: {np.mean(scores_window):.2f}')
        return True

    return False


def train_agent(
    env,
    agent,
    n_episodes=2000,
    max_t=300,
    eps_scheduler=EpsScheduler()
):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    brain_name = env.brain_names[0]

    scores = []
    scores_window = deque(maxlen=100)

    for i_episode in range(1, n_episodes+1):
        score = 0
        
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]

        eps = eps_scheduler.get_eps(i_episode)
        
        for t in range(max_t):
            action = agent.act(state, eps)
            
            env_info = env.step(action)[brain_name]
            next_state, reward, done = env_info.vector_observations[0], env_info.rewards[0], env_info.local_done[0]

            agent.step(state, action, reward, next_state, done)
            
            state = next_state
            score += reward
            if done:
                break 

        scores_window.append(score)
        scores.append(score)
        
        if _check_solved(i_episode, scores_window):
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')

    env.close()

    return scores


def plot_scores(scores):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
