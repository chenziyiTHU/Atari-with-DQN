import os
import argparse
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import ale_py
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

from dqn import QNetwork
from replay_buffer import ReplayBuffer
from atari_wrapper import make_atari_env


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='BreakoutNoFrameskip-v4')
    parser.add_argument('--eps_test', type=float, default=0.005)
    parser.add_argument("--eps_train_start", type=float, default=1.0)
    parser.add_argument("--eps_train_end", type=float, default=0.05)
    parser.add_argument("--eps_train_decay", type=float, default=1-1e-5)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--n_steps_per_epoch', type=int, default=100000)
    parser.add_argument('--n_test_episodes', type=int, default=20)
    parser.add_argument('--buffer_size', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--update_every', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--k_target', type=int, default=500)
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--beta', type=float, default=0.4)
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoint')
    parser.add_argument('--from_pretrained', type=str, default=None, help="load the model from checkpoint")
    parser.add_argument('--watch', default=False, action="store_true", help="watch the play of pre-trained policy only")
    return parser.parse_args()


class Agent:
    def __init__(self, state_size, action_size, buffer_size, batch_size, update_every, gamma, lr, k_target, alpha, beta, device):
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.update_every = update_every
        self.gamma = gamma
        self.lr = lr
        self.k_target = k_target
        self.alpha = alpha
        self.beta = beta
        self.device = device

        self.qnetwork_local = QNetwork(self.state_size[0], self.action_size).to(self.device)
        self.qnetwork_target = QNetwork(self.state_size[0], self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)
        self.memory = ReplayBuffer(self.state_size, self.buffer_size, self.alpha)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        '''Save experience in replay memory, and use random sample from buffer to learn.
        
        Params:
            state (array_like): current state
            action (int): action taken
            reward (float): reward received
            next_state (array_like): next state
            done (bool): whether the episode is finished
        '''
        self.memory.add(state, action, reward, next_state, done)
        self.t_step += 1

        if self.t_step % self.update_every == 0:
            if len(self.memory) > self.batch_size:
                self.learn()

        if self.t_step % self.k_target == 0:
            self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

    def act(self, state, eps=0.):
        '''Returns actions for given state as per current policy.
        
        Params:
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        '''
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action_values = self.qnetwork_local(state)

        # Epsilon-greedy action selection
        if random.random() > eps:
            return torch.argmax(action_values, -1)
        else:
            return random.choice(np.arange(self.action_size))
        
    def learn(self):
        '''Update value parameters using given batch of experience tuples by Double Q-learning.

        Params:
            experiences (Tuple[torch.Variable]): dictionary of state, action, reward, next_state, done, weights, indices
            weights (torch.Variable): importance sampling weights to correct the bias
        '''
        experiences = self.memory.sample(self.batch_size, self.beta)
        states = torch.tensor(experiences['state'], dtype=torch.float32).to(self.device)
        actions = torch.tensor(experiences['action'], dtype=torch.int64).to(self.device)
        rewards = torch.tensor(experiences['reward'], dtype=torch.float32).to(self.device)
        next_states = torch.tensor(experiences['next_state'], dtype=torch.float32).to(self.device)
        dones = torch.tensor(experiences['done'], dtype=torch.float32).to(self.device)
        weights = torch.tensor(experiences['weights'], dtype=torch.float32).to(self.device)

        # Double Q-learning
        q_next_action = self.qnetwork_local(next_states).max(1)[1]
        q_next = self.qnetwork_target(next_states).gather(1, q_next_action.long().unsqueeze(1)).squeeze(1)
        q_targets = rewards + self.gamma * q_next * (1 - dones)
        q_preds = self.qnetwork_local(states).gather(1, actions.long().unsqueeze(1)).squeeze(1)
        td_errors = q_targets - q_preds

        self.memory.update_priorities(experiences['indices'], np.abs(td_errors.cpu().detach().numpy()) + 1e-5)

        loss = F.mse_loss(q_preds * weights, q_targets * weights)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, PATH, filename):
        if not os.path.exists(PATH):
            os.makedirs(PATH)
        torch.save(self.qnetwork_local.state_dict(), os.path.join(PATH, filename))

    def load(self, PATH):
        self.qnetwork_local.load_state_dict(torch.load(PATH))
    

def train(agent: Agent, 
          env_train: gym.Env, 
          env_test: gym.Env,
          n_epochs=100, 
          n_steps_per_epoch=1000,
          n_test_episodes=20,
          eps_test=0.005,
          eps_train_start=1.0, 
          eps_train_end=0.01, 
          eps_decay=1-1e-5, 
          checkpoint_path='./checkpoint.pth', 
          from_pretrained=False):
    
    '''Deep Q-Learning.
    
    Params:
        agent (Agent): the agent to be trained
        env_train (gym.Env): training environment
        env_test (gym.Env): testing environment
        n_epochs (int): maximum number of training epochs
        n_steps_per_epoch (int): maximum number of steps per epoch
        n_test_episodes (int): number of episodes for testing
        eps_test (float): epsilon for testing
        eps_train_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_train_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        checkpoint_path (str): path to save the model or load the model
        from_pretrained (str): load the model from checkpoint (if not None), default is None
    '''
    if from_pretrained is not None:
        agent.load(from_pretrained)

    eps = eps_train_start
    env_step = 0
    total_train_scores = []
    total_test_scores = []
    agent.qnetwork_local.train()
    for epoch in range(n_epochs):
        state, info = env_train.reset()
        score = 0
        scores_train = []
        done = False
        for step in tqdm(range(n_steps_per_epoch)):
            action = agent.act(state, eps)
            next_state, reward, done, _, _ = env_train.step(action)
            agent.step(state, action, reward, next_state, done)
            score += reward
            if done:
                scores_train.append(score)
                score = 0
                state, info = env_train.reset()
            else:
                state = next_state
            
            if env_step <= 1e6:
                eps = eps_train_start - env_step / 1e6 * (eps_train_start - eps_train_end)
            else:
                eps = eps_train_end
            env_step += 1
        print('Epoch:', epoch, 'Average Training Score:', np.mean(scores_train))
        total_train_scores.append(np.mean(scores_train))
        agent.save(checkpoint_path, 'epoch_{}.pth'.format(epoch))

        # test the agent
        scores_test = []
        for _ in range(n_test_episodes):
            state, info = env_test.reset()
            score = 0
            done = False
            while not done:
                action = agent.act(state, eps_test)
                next_state, reward, done, _, _ = env_test.step(action)
                score += reward
                state = next_state
            scores_test.append(score)
        print('Epoch:', epoch, 'Average Test Score:', np.mean(scores_test))
        total_test_scores.append(np.mean(scores_test))
        print(scores_test)

    plot_scores(total_train_scores, filename='training_curve.jpg')
    plot_scores(total_test_scores, filename='test_curve.jpg')


def plot_scores(scores, filename='training_curve.jpg'):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.savefig(filename)
    plt.close('all')


if __name__ == '__main__':
    args = get_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print('Device:', device)
    env_train, env_test = make_atari_env(args.task, mode='rgb_array')
    print('State shape:', env_train.observation_space.shape)
    print('Number of actions:', env_train.action_space.n)

    agent = Agent(state_size=env_train.observation_space.shape, 
                  action_size=env_train.action_space.n, 
                  buffer_size=args.buffer_size,
                  batch_size=args.batch_size,
                  update_every=args.update_every,
                  gamma=args.gamma,
                  lr=args.lr,
                  k_target=args.k_target,
                  alpha=args.alpha,
                  beta=args.beta,
                  device=device
                  )
    
    if args.watch:
        print('Setup test envs ...')
        agent.load(args.from_pretrained)

        env_test = RecordVideo(env_test, video_folder='./video/{}_dqn_test'.format(args.task), episode_trigger=lambda t: True, disable_logger=True)
        state, info = env_test.reset()
        score = 0
        done = False
        while not done:
            action = agent.act(state, args.eps_test)
            next_state, reward, done, _, _ = env_test.step(action)
            state = next_state
            score += reward
        env_test.close()
        print('Test score:', score)

    else:
        print('Start training... Number of epochs:', args.n_epochs)
        env_train = RecordVideo(env_train, video_folder='./video/{}_dqn'.format(args.task), episode_trigger=lambda t: t % 100 == 0, disable_logger=True)
        train(agent, 
            env_train,
            env_test,
            n_epochs=args.n_epochs,
            n_steps_per_epoch=args.n_steps_per_epoch,
            n_test_episodes=args.n_test_episodes, 
            eps_test=args.eps_test,
            eps_train_start=args.eps_train_start, 
            eps_train_end=args.eps_train_end, 
            eps_decay=args.eps_train_decay, 
            checkpoint_path='./{}_dqn'.format(args.task),
            from_pretrained=args.from_pretrained)