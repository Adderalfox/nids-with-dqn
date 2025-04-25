import torch
from nids_env import NIDSEnv
from dqn_agent import DQNAgent
from preprocess import preprocess_data

episodes = 1000
target_update_freq = 10
checkpoint_interval = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training on:", device)

x_train, y_train, x_test, y_test = preprocess_data('UNSW_NB15_training-set.csv', 'UNSW_NB15_testing-set.csv')

env = NIDSEnv(x_train, y_train)
state_dim = x_train.shape[1]
action_dim = 2

agent = DQNAgent(state_dim, action_dim, device=device)

for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    done = False

    count = 0
    while not done:
        action = agent.act(state)
        reward, next_state, done = env.step(action)

        agent.remember(state, action, reward, next_state, done)
        agent.replay()

        state = next_state
        total_reward += reward

        count += 1
        print(f'In step {count}: Action Taken:-{action}, Total Reward:-{total_reward}')

    if agent.epsilon > agent.epsilon_min:
        agent.epsilon *= agent.epsilon_decay

    if episode % target_update_freq == 0:
        agent.update_target_network()

    if episode % checkpoint_interval == 0:
        checkpoint_path = f"checkpoints/checkpoint_{episode}.pth"
        agent.save_checkpoint(checkpoint_path)

    print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward: .2f}, Epsilon: {agent.epsilon: .4f}")

torch.save(agent.policy_net.state_dict(), 'dqn_nids_model.pth')
# try:
#     last_checkpoint = 'checkpoint'
#     agent.load(last_checkpoint)
# except:
