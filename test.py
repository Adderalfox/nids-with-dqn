import torch
import numpy as np
from dqn_agent import DQNAgent
from preprocess import preprocess_data

# Load data
x_train, y_train, x_test, y_test = preprocess_data('UNSW_NB15_training-set.csv', 'UNSW_NB15_testing-set.csv')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state_dim = x_test.shape[1]
action_dim = 2  # Assuming binary classification: 0 (benign) or 1 (malicious)

# Initialize agent
agent = DQNAgent(state_dim, action_dim, device=device)

# Load trained model
# agent.policy_net.load_state_dict(torch.load('dqn_nids_model.pth', map_location=device))
checkpoint = torch.load('checkpoints/checkpoint_395.pth', map_location=device)
agent.policy_net.load_state_dict(checkpoint['policy_state_dict'])
agent.policy_net.eval()

# Initialize counters
TP = 0
TN = 0
FP = 0
FN = 0
total = len(x_test)

# Testing loop
for i in range(total):
    state = x_test[i]
    true_label = y_test[i]

    state = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
    with torch.no_grad():
        q_values = agent.policy_net(state)
    action = torch.argmax(q_values).item()

    if action == 1 and true_label == 1:
        TP += 1
    elif action == 0 and true_label == 0:
        TN += 1
    elif action == 1 and true_label == 0:
        FP += 1
    elif action == 0 and true_label == 1:
        FN += 1

    print(f"Sample {i + 1}/{total} | Predicted: {action}, True: {true_label}")

# Metrics calculation
accuracy = (TP + TN) / total * 100
precision = TP / (TP + FP) * 100 if (TP + FP) > 0 else 0
recall = TP / (TP + FN) * 100 if (TP + FN) > 0 else 0  # Optional: recall

# Results
print("\n=== Test Results ===")
print(f"True Positives (TP): {TP}")
print(f"True Negatives (TN): {TN}")
print(f"False Positives (FP): {FP}")
print(f"False Negatives (FN): {FN}")
print(f"Accuracy: {accuracy:.2f}%")
print(f"Precision: {precision:.2f}%")
