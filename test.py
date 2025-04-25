import torch
import numpy as np
from dqn_agent import DQNAgent
from preprocess import preprocess_data

x_train, y_train, x_test, y_test = preprocess_data('UNSW_NB15_training-set.csv', 'UNSW_NB15_testing-set.csv')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state_dim = x_test.shape[1]
action_dim = 2
agent = DQNAgent(state_dim, action_dim, device=device)

agent.policy_net.load_state_dict(torch.load('dqn_nids_model.pth', map_location=device))
# checkpoint = torch.load('checkpoints/checkpoint_90.pth', map_location=device)
# agent.policy_net.load_state_dict(checkpoint['policy_state_dict'])
agent.policy_net.eval()

correct = 0
total = len(x_test)

for i in range(total):
    state = x_test[i]
    true_label = y_test[i]

    action = agent.act(state)

    if action == true_label:
        correct += 1

    print(f"Sample {i + 1}/{total} | Predicted: {action}, True: {true_label}")

accuracy = correct / total * 100
print(f"\n Test accuracy: {accuracy}%")