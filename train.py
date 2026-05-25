import torch
from nids_env import NIDSEnv
from dqn_agent import DQNAgent
from preprocess import preprocess_data

episodes = 3500
target_update_freq = 10
checkpoint_interval = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training on:", device)

x_train, y_train, x_test, y_test = preprocess_data('UNSW_NB15_training-set.csv', 'UNSW_NB15_testing-set.csv')

env = NIDSEnv(x_train, y_train)
state_dim = x_train.shape[1]
action_dim = 2

agent = DQNAgent(state_dim, action_dim, device=device)

#---------------------------Resume Training Block------------------------------#

# start_episode = 2485
# checkpoint_path = f"checkpoints/checkpoint_{start_episode}.pth"

# checkpoint = torch.load(checkpoint_path, map_location=device)
# agent.policy_net.load_state_dict(checkpoint['policy_state_dict'])
# agent.target_net.load_state_dict(checkpoint['target_state_dict'])
# agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# agent.epsilon = checkpoint['epsilon']

# print(f"Loaded checkpoint from episode {start_episode}")

# print(f"Resuming training from episode {start_episode}...")

#---------------------------Resume Training Block------------------------------#

for episode in range(episodes):
# for episode in range(start_episode, episodes):

    state = env.reset()
    total_reward = 0
    done = False

    count = 0
    max_steps_per_episode = 1000 # Limit steps to avoid infinite loops and speed up episodes
    while not done and count < max_steps_per_episode:
        action = agent.act(state)
        reward, next_state, done, label = env.step(action)

        agent.remember(state, action, reward, next_state, done, label)
        
        # Only replay every 10 steps to improve speed and stability
        if count % 10 == 0:
            agent.replay()

        state = next_state
        total_reward += reward

        count += 1
        if count % 100 == 0:
            print(f'Episode {episode}, Step {count}: Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.4f}')

    if agent.epsilon > agent.epsilon_min:
        agent.epsilon *= agent.epsilon_decay

    # if episode % target_update_freq == 0:
    #     agent.update_target_network()

    if episode % checkpoint_interval == 0:
        checkpoint_path = f"checkpoints/checkpoint_{episode}.pth"
        agent.save_checkpoint(checkpoint_path)

    print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward: .2f}, Epsilon: {agent.epsilon: .4f}")

torch.save(agent.policy_net.state_dict(), 'dqn_nids_model.pth')
