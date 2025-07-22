from unityagents import UnityEnvironment

from dqn_agent import Agent
from utils import train_agent, plot_scores


banana_world = UnityEnvironment(file_name='./Banana_Linux/Banana.x86_64', no_graphics=True)
banana_collector = Agent(state_size=37, action_size=4, seed=0)

try:
    scores = train_agent(
        env   = banana_world,
        agent = banana_collector
    )

    plot_scores(scores)
finally:
    banana_world.close()