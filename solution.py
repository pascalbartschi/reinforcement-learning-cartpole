import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.utils import seeding
from utils import ReplayBuffer, get_env, run_episode

# added libs
import copy

# docker build --tag task4 .;docker run --rm -v "%cd%:/results" task4


class MLP(nn.Module):
    '''
    A simple ReLU MLP constructed from a list of layer widths.
    '''
    def __init__(self, sizes):
        super().__init__()
        layers = []
        for i, (in_size, out_size) in enumerate(zip(sizes[:-1], sizes[1:])):
            layers.append(nn.Linear(in_size, out_size))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)


class Critic(nn.Module):
    '''
    Simple MLP Q-function. Policy that evaluates actor actions by outputting Q-values

    Role: The critic evaluates the quality of actions taken by the actor in terms of the 
    expected cumulative reward. In TD3, there are two critics to address overestimation bias,
    a common issue in Q-value estimation.

    Output: The critics output Q-values, which represent the estimated future rewards for taking
    a given action in a particular state.

    Optimization Goal: Each critic learns to minimize the Bellman error, ensuring accurate Q-value
    predictions for the current policy.
    '''
    def __init__(self, obs_size, action_size, num_layers, num_units):
        super().__init__()
        #####################################################################
        # TODO: add components as needed (if needed)

        self.net1 = MLP([obs_size + action_size] + ([num_units] * num_layers) + [1])

        self.net2 = MLP([obs_size + action_size] + ([num_units] * num_layers) + [1])

        #####################################################################

    def forward(self, x, a):

        # the critic receives a batch of observations and a batch of actions
        # of shape (batch_size x obs_size) and (batch_size x action_size) respectively
        # and output a batch of values of shape (batch_size x 1)

        # to learn the interaction, we concatenate state and action of batches
        xa = torch.cat([x, a], dim = 1)
        # pass through first MLP
        value1 = self.net1(xa)
        # pass through second MLP
        value2 = self.net2(xa)
        
        return value1, value2
    
    def Q1(self, x, a): 

        xa = torch.cat([x,a], dim = 1)

        return self.net1(xa)


class Actor(nn.Module):
    '''
    Simple Tanh deterministic actor given state, is optimized to maximize the expected cumulative reward
    using the critics value function.
        
    Role: The actor is responsible for deciding the actions the agent should take in the environment. 
    It represents the policy, which maps the current state to an action.

    Output: For continuous action spaces (as in TD3), the actor outputs a deterministic action given a state.

    Optimization Goal: The actor is optimized to maximize the expected cumulative reward, 
    as estimated by the critic. It uses the critic's value function to understand which actions
    are beneficial for long-term rewards.
    '''
    def __init__(self, action_low, action_high,  obs_size, action_size, num_layers, num_units):
        super().__init__()
        #####################################################################
        # TODO: add components as needed (if needed)

        self.net = MLP([obs_size] + ([num_units] * num_layers) + [action_size])
        self.tanh = nn.Tanh()

        #####################################################################
        # store action scale and bias: the actor's output can be squashed to [-1, 1]
        self.action_scale = (action_high - action_low) / 2
        self.action_bias = (action_high + action_low) / 2

    def forward(self, x):
        #####################################################################
        # TODO: code the forward pass
        # the actor will receive a batch of observations of shape (batch_size x obs_size)
        # and output a batch of actions of shape (batch_size x action_size)

        action = self.tanh(self.net(x)) * self.action_scale + self.action_bias

        #####################################################################
        return action


class Agent:
    """
    TD3 incorporated delayed policy updates (actor is updated less frequently than critic), where two critics
    help mitigate overestimation

    By iteratively improving both the actor and the critics through interaction with the environment 
    and stored experiences (replay buffer), TD3 enables the agent to learn an effective
    policy for solving the cartpole problem.
    """

    # automatically select compute device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    buffer_size: int = 50_000  # no need to change

    #########################################################################
    # TODO: store and tune hyperparameters here

    batch_size: int = 256
    gamma: float = 0.99  # MDP discount factor, 
    exploration_noise: float = 0.3  # epsilon for epsilon-greedy exploration

    train_iter = 0
    tau=0.005
    policy_noise=0.2
    noise_clip=0.5
    policy_freq=2

    learning_rate = 5e-4
    
    #########################################################################

    def __init__(self, env):

        # extract informations from the environment
        self.obs_size = np.prod(env.observation_space.shape)  # size of observations
        self.action_size = np.prod(env.action_space.shape)  # size of actions
        # extract bounds of the action space
        self.action_low = torch.tensor(env.action_space.low).float()
        self.action_high = torch.tensor(env.action_space.high).float()

        #####################################################################
        # TODO: initialize actor, critic and attributes

        # The main networks and target networks in TD3 are connected via a mechanism called soft updates.
        # During training, the main networks (actor and critics) learn directly from the data and the TD3 algorithm,
        # while the target networks provide stable targets to guide this learning.

        # main actor network -> determines actions for the current state
        self.actor = Actor(self.action_low, self.action_high, self.obs_size, self.action_size, 
                            num_layers=2, num_units=256).to(self.device)
        # target actor network -> stabilization, slow update via main actor
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        # optimizer of main actor
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = self.learning_rate)
        

        # the main critic network -> used to predict Q-values for the state action pairs
        self.critic = Critic(self.obs_size, self.action_size, 
                                num_layers=2, num_units=256).to(self.device)
         # target critic network -> stabilization, slow update via main critic
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
         # target critic network -> stabilization, slow update via mai
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr = self.learning_rate)


        # define the loss for the critic
        self.mse_loss = nn.MSELoss()

        


        
        


        #####################################################################
        # create buffer
        self.buffer = ReplayBuffer(self.buffer_size, self.obs_size, self.action_size, self.device)
        self.train_step = 0
    
    def train(self):
        '''
        Updates actor and critic with one batch from the replay buffer.
        '''
        self.train_iter += 1

        obs, action, next_obs, done, reward = self.buffer.sample(self.batch_size)
        

        #####################################################################
        with torch.no_grad():
            # add noise:  improve the stability of critic training by smoothing the Q-value updates.
            noise = (
			    	torch.randn_like(action) * self.policy_noise
			        ).clamp(-self.noise_clip, self.noise_clip)
            # we get the next action of the target actor
            next_action = (
                        self.actor_target(next_obs) + noise
            ).clamp(self.action_low, self.action_high)

            # we calculate the Q-values of the target critic
            critic_target_Q1, critic_target_Q2 = self.critic_target(next_obs, next_action)
            critic_target_Q = torch.min(critic_target_Q1, critic_target_Q2)
            target_Q = reward.view(-1, 1) + self.gamma * critic_target_Q
        
        # obtain the current Q estimate
        current_Q1, current_Q2 = self.critic(obs, action)
        
        # compute the MSE loss to target Q
        critic_loss = self.mse_loss(current_Q1, target_Q) + self.mse_loss(current_Q2, target_Q)

        # optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update the policy (actor) only with certain frequency
        if self.train_iter % self.policy_freq == 0: 

            # compute the loss for the actor: Aims to maximize the action selected by the actor leading to highest reward
            actor_loss = -self.critic.Q1(obs, self.actor(obs)).mean() 

            # optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # apply the soft updates to the target networks: target_param = tau*param + (1-tau)  * target_param
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)



        #####################################################################

    def get_action(self, obs, mode):
        '''
        Returns the agent's action for a given observation.
        The mode parameter can be used to control stochastic behavior.
        '''
        #####################################################################
        # TODO: return the agent's action for an observation (np.array
        # of shape (obs_size, )). The action should be a np.array of
        # shape (act_size, )

        # Convert NumPy array to PyTorch tensor
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, device=self.device).float()

        noise = torch.randn(self.action_size) * self.exploration_noise if mode == "train" else 0

        with torch.no_grad():
            action = self.actor(obs) + noise
        
        #####################################################################
        return action

    def store(self, transition):
        '''
        Stores the observed transition in a replay buffer containing all past memories.
        '''
        obs, action, reward, next_obs, terminated = transition
        self.buffer.store(obs, next_obs, action, reward, terminated)


# This main function is provided here to enable some basic testing. 
# ANY changes here WON'T take any effect while grading.
if __name__ == '__main__':

    WARMUP_EPISODES = 10  # initial episodes of uniform exploration
    TRAIN_EPISODES = 50  # interactive episodes
    TEST_EPISODES = 300  # evaluation episodes
    save_video = False
    verbose = True
    seeds = np.arange(10)  # seeds for public evaluation

    start = time.time()
    print(f'Running public evaluation.') 
    test_returns = {k: [] for k in seeds}

    for seed in seeds:

        # seeding to ensure determinism
        seed = int(seed)
        for fn in [random.seed, np.random.seed, torch.manual_seed]:
            fn(seed)
        torch.backends.cudnn.deterministic = True

        env = get_env()
        env.action_space.seed(seed)
        env.np_random, _ = seeding.np_random(seed)

        agent = Agent(env)

        for _ in range(WARMUP_EPISODES):
            run_episode(env, agent, mode='warmup', verbose=verbose, rec=False)

        for _ in range(TRAIN_EPISODES):
            run_episode(env, agent, mode='train', verbose=verbose, rec=False)

        for n_ep in range(TEST_EPISODES):
            video_rec = (save_video and n_ep == TEST_EPISODES - 1)  # only record last episode
            with torch.no_grad():
                episode_return = run_episode(env, agent, mode='test', verbose=verbose, rec=video_rec)
            test_returns[seed].append(episode_return)

    avg_test_return = np.mean([np.mean(v) for v in test_returns.values()])
    within_seeds_deviation = np.mean([np.std(v) for v in test_returns.values()])
    across_seeds_deviation = np.std([np.mean(v) for v in test_returns.values()])
    print(f'Score for public evaluation: {avg_test_return}')
    print(f'Deviation within seeds: {within_seeds_deviation}')
    print(f'Deviation across seeds: {across_seeds_deviation}')

    print("Time :", (time.time() - start)/60, "min")
