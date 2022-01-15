from tqdm import tqdm

import numpy as np
import torch


from dqn.agent import GreedyVertexSelector
from tsp_env import TspEnv


# TODO: not tested.
def run(training_mode, pretrained, num_episodes=1000, exploration_max=1):

    env = TspEnv()

    observation_space = env.observation_space.shape
    action_space = env.action_space.n

    agent = GreedyVertexSelector(
        state_space=observation_space,
        action_space=action_space,
        max_memory_size=30000,
        batch_size=32,
        gamma=0.90,
        lr=0.00025,
        dropout=0.2,
        exploration_max=1.0,
        exploration_min=0.02,
        exploration_decay=0.99,
        pretrained=pretrained,
    )

    # Restart the enviroment for each episode
    num_episodes = num_episodes
    env.reset()

    total_rewards = []
    # if training_mode and pretrained:
    #     with open("total_rewards.pkl", "rb") as f:
    #         total_rewards = pickle.load(f)

    for ep_num in tqdm(range(num_episodes)):
        state = env.reset()
        state = torch.Tensor([state])
        total_reward = 0
        steps = 0
        while True:
            action = agent.act(state)
            steps += 1

            state_next, reward, terminal, info = env.step(int(action[0]))
            total_reward += reward
            state_next = torch.Tensor([state_next])
            reward = torch.tensor([reward]).unsqueeze(0)

            terminal = torch.tensor([int(terminal)]).unsqueeze(0)

            if training_mode:
                agent.remember(state, action, reward, state_next, terminal)
                agent.experience_replay()

            state = state_next
            if terminal:
                break

        total_rewards.append(total_reward)

        if ep_num != 0 and ep_num % 100 == 0:
            print(
                "Episode {} score = {}, average score = {}".format(
                    ep_num + 1, total_rewards[-1], np.mean(total_rewards)
                )
            )
        num_episodes += 1

    print(
        "Episode {} score = {}, average score = {}".format(
            ep_num + 1, total_rewards[-1], np.mean(total_rewards)
        )
    )

    # Save the trained memory so that we can continue from where we stop using 'pretrained' = True
    if training_mode:
        with open("ending_position.pkl", "wb") as f:
            pickle.dump(agent.ending_position, f)
        with open("num_in_queue.pkl", "wb") as f:
            pickle.dump(agent.num_in_queue, f)
        with open("total_rewards.pkl", "wb") as f:
            pickle.dump(total_rewards, f)

        torch.save(agent.dqn.state_dict(), "DQN.pt")
        torch.save(agent.STATE_MEM, "STATE_MEM.pt")
        torch.save(agent.ACTION_MEM, "ACTION_MEM.pt")
        torch.save(agent.REWARD_MEM, "REWARD_MEM.pt")
        torch.save(agent.STATE2_MEM, "STATE2_MEM.pt")
        torch.save(agent.DONE_MEM, "DONE_MEM.pt")

    env.close()
