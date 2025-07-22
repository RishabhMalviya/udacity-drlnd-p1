from unityagents import UnityEnvironment

from dqn_agent import Agent
from utils import train_agent, plot_scores, watch_agent


# Train Agent
scores, solved = train_agent(
    env    = UnityEnvironment(file_name='./Banana_Linux/Banana.x86_64', no_graphics=True),
    agent  = Agent(state_size=37, action_size=4, seed=0)
)
plot_scores(scores)

# Watch Agent
if solved:
    watch_agent(
        env    = UnityEnvironment(file_name='./Banana_Linux/Banana.x86_64'),
        agent  = Agent(state_size=37, action_size=4, seed=0)
    )
