# Reinforcement Learning for Cartpole Control

This project implements the **Twin Delayed Deep Deterministic Policy Gradient (TD3)** algorithm to solve the Cartpole problem. The goal is to train an agent to balance a pole on a moving cart by learning an optimal policy through interaction with the environment.

## Problem Description
The **Cartpole environment** consists of:
- A pole linked to a cart, initialized in a vertical (inverted) position.
- A cart that moves on a frictionless rail.
- The task is to keep the pole upright by applying forces to the cart.

The state and action spaces are:
- **State** $`s \in \mathbb{R}^4`$: Includes the cart position, cart velocity, pole angle, and pole angular velocity.
- **Action** $`a \in \mathbb{R}`$: Continuous forces applied to the cart.

The **reward** is proportional to the duration the pole remains balanced.


![cart-pole-neural-network](https://media1.tenor.com/m/kff3i4Rx4wMAAAAd/cart-pole-neural-network.gif)


## Approach: TD3 Algorithm
TD3 is an off-policy actor-critic algorithm designed for continuous action spaces. Key features include:

### **1. Actor-Critic Architecture**
- **Actor**: Maps states to deterministic actions.

  ```math
  \pi(s) = \tanh(\text{MLP}(s)) \cdot \text{scale} + \text{bias}
  ```

- **Critic**: Estimates Q-values for state-action pairs.

  ```math
  Q(s, a) = \text{MLP}([s, a])
  ```

### **2. Target Networks**
- Stabilize learning by slowly updating target networks via soft updates:

  ```math
  \theta_{\text{target}} \gets \tau \theta + (1 - \tau) \theta_{\text{target}}
  ```

### **3. Delayed Policy Updates**
- The actor is updated less frequently than the critic to reduce variance in policy gradients.

### **4. Action Noise**
- Gaussian noise is added during training to encourage exploration:

  ```math
  a_{\text{train}} = \pi(s) + \mathcal{N}(0, \sigma^2)
  ```

### **5. Clipped Double Q-Learning**
- Two critics are used to mitigate overestimation bias:

  ```math
  Q_{\text{target}} = r + \gamma \min(Q_1(s', \pi(s')), Q_2(s', \pi(s')))
  ```

## Implementation
### **Key Components**
1. **Actor**: Determines the best action for a given state.
2. **Critic**: Evaluates the quality of actions taken by the actor.
3. **Replay Buffer**: Stores transitions $`(s, a, r, s', d)`$ for training.
4. **TD3 Training Loop**:
   - Sample mini-batches from the replay buffer.
   - Update critics using the Bellman equation.
   - Update the actor using policy gradients.
   - Perform soft updates for target networks.

### **Hyperparameters**
- Discount factor $`\gamma`$: 0.99
- Learning rate: $`5 \times 10^{-4}`$
- Batch size: 256
- Exploration noise: 0.3
- Target noise: 0.2 (clipped at 0.5)
- Policy update frequency: 2
- Soft update coefficient $`\tau`$: 0.005

### **Training and Evaluation**
1. **Warm-up**: Random actions for initial exploration.
2. **Training**: Update actor-critic networks using TD3.
3. **Testing**: Evaluate the learned policy over multiple episodes.

## Results
The TD3 agent successfully learns to balance the pole by optimizing the cumulative reward. Key observations:
- Effective exploration through Gaussian noise.
- Stable learning due to delayed updates and target networks.
- Accurate Q-value estimation with clipped double Q-learning.
- **Final reward** $`-21.7`$ (ranked 96/277).
