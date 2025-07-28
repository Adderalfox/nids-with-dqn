# 🔐 Intrusion Detection System using Deep Q-Learning
A reinforcement learning-based Intrusion Detection System (IDS) built with Deep Q-Networks (DQN) to detect and classify network threats using the UNSW-NB15 dataset.

## 🚀 Overview
- Technique: Deep Q-Learning (a form of Reinforcement Learning)
- Goal: Automatically learn optimal detection policies for network intrusions
- Dataset: UNSW-NB15 – a modern benchmark dataset for network traffic analysis

## 🧠 Model Highlights
- Implemented Deep Q-Network (DQN) with experience replay and target network
- State space: Network traffic features from the UNSW-NB15 dataset
- Action space: Binary (Normal or Attack) or Multi-class (various attack types)
- Reward mechanism encourages accurate classification and penalizes misclassifications

## 📊 Dataset: UNSW-NB15
- Contains normal and malicious traffic labeled with 9 attack categories
- Extracted using Argus and Bro tools, processed into CSV format
- Preprocessing steps include normalization, encoding, and dimensionality reduction

## 🛠️ Features
- Intelligent agent learns to detect intrusions through trial-and-error
- Suitable for real-time and adaptive cybersecurity systems
- Customizable reward shaping and exploration strategies (e.g., ε-greedy)
- Evaluated using precision, recall, F1-score, and detection rate
