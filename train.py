from tqdm import tqdm
import pickle

import numpy as np
import torch


from dqn.agent import GreedyVertexSelector
from tsp_env import TspEnv


# TODO: not tested.
def run(training_mode, pretrained, num_episodes=10000, exploration_max=1):

    env = TspEnv(graph_type="ba", n_nodes=40, dim=2)

    agent = GreedyVertexSelector(
        max_memory_size=30000,
        batch_size=256,
        gamma=0.99,
        lr=0.001,
        exploration_max=1.0,
        exploration_min=0.02,
        exploration_decay=0.999,
        pretrained=pretrained,
    )

    # Restart the enviroment for each episode
    num_episodes = num_episodes
    env.reset(1, new_graph=True)

    total_rewards = []
    lengths = []
    stepss = []
    # if training_mode and pretrained:
    #     with open("total_rewards.pkl", "rb") as f:
    #         total_rewards = pickle.load(f)
    pbar = tqdm(range(num_episodes))
    for ep_num in pbar:
        state = env.reset()

        total_reward = 0
        steps = 0
        while True:
            action = agent.act(state)
            # print(action)
            steps += 1

            state_next, reward, terminal, info = env.step(action)
            total_reward += reward

            reward = torch.tensor([reward]).unsqueeze(0)

            terminal = torch.tensor([int(terminal)]).unsqueeze(0)

            if training_mode:
                agent.remember(state, action, reward, state_next, terminal)
                agent.experience_replay()

            state = state_next
            if terminal:
                length = info["length"]
                break

        total_rewards.append(total_reward)
        stepss.append(steps)

        lengths.append(length)

        pbar.set_description(
            f"Episode: {ep_num}, Reward: {total_reward:.4f}, End Steps: {steps}, Exploration: {agent.exploration_rate:.4f}, Length : {length:.4f}"
        )

        if ep_num != 0 and ep_num % 100 == 0:
            avgr = np.mean(total_rewards[-100:])
            avglen = np.mean(lengths[-100:])
            avgstep = np.mean(stepss[-100:])
            print(
                f"Episode: {ep_num}, Avg Reward: {avgr:.4f}, End Steps: {avgstep}, Exploration: {agent.exploration_rate:.4f}, Avg Length : {avglen:.4f}"
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
        with open("stepss.pkl", "wb") as f:
            pickle.dump(stepss, f)
        with open("lengths.pkl", "wb") as f:
            pickle.dump(lengths, f)

        torch.save(agent.dqn.state_dict(), "DQN.pt")
        torch.save(agent.STATE_PRV_MEM, "STATE_MEM.pt")
        torch.save(agent.ACTION_MEM, "ACTION_MEM.pt")
        torch.save(agent.REWARD_MEM, "REWARD_MEM.pt")
        torch.save(agent.STATE_AFT_MEM, "STATE2_MEM.pt")
        torch.save(agent.DONE_MEM, "DONE_MEM.pt")

    env.close()


if __name__ == "__main__":
    run(True, False)
