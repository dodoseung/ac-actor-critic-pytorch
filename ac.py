# Action-Value Actor Critic (Q-value)
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import gym
from collections import deque

class ACNet(nn.Module):
    def __init__(self, input, output):
        super(ACNet, self).__init__()
        self.input = nn.Linear(input, 16)
        self.fc = nn.Linear(16, 16)
	
        self.value = nn.Linear(16, output)
        self.policy = nn.Linear(16, output)
    
    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.fc(x))
        
        value = self.value(x)
        policy = F.softmax(self.policy(x))
        return value, policy
    
class AC():
    def __init__(self, env, actor_ratio=0.5, gamma=0.95, learning_rate=1e-3):
        super(AC, self).__init__()
        self.env = env
        self.state_num = self.env.observation_space.shape[0]
        self.action_num = self.env.action_space.n
               
        # Torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.ac_net = ACNet(self.state_num, self.action_num).to(self.device)
        self.optimizer = optim.Adam(self.ac_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.actor_ratio = actor_ratio
        
    # Get the action
    def get_action(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        _, policy = self.ac_net(state)
        policy = policy.cpu().detach().numpy()
        action = np.random.choice(self.action_num, 1, p=policy[0])
        return action[0]

    # Learn the policy
    # j: Policy objective function
    def learn(self, states, actions, rewards, next_states, dones):
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        action_values, policies = self.ac_net(states)
        next_action_values, _ = self.ac_net(next_states)

        values = action_values.gather(1, actions.view(-1, 1)).view(1, -1)
        next_values = (rewards + self.gamma * torch.max(next_action_values, 1)[0] * (1-dones))
        
        log_prob = torch.log(policies)
        j = values * log_prob[range(len(actions)), actions]

        actor_loss = -j.mean()
        critic_loss = F.smooth_l1_loss(values, next_values)
        loss = self.actor_ratio * actor_loss + critic_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def main():
    ep_rewards = deque(maxlen=100)
    
    env = gym.make("CartPole-v0")
    agent = AC(env, actor_ratio=0.2, gamma=0.99, learning_rate=3e-4)
    total_episode = 10000
    
    for i in range(total_episode):
        state = env.reset()
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        while True:
            action = agent.get_action(state)
            next_state, reward , done, _ = env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            
            if done:
                ep_rewards.append(sum(rewards))
                agent.learn(states, actions, rewards, next_states, dones)
                
                if i % 100 == 0:
                    print("episode: {}\treward: {}".format(i, round(np.mean(ep_rewards), 3)))
                break

            state = next_state

if __name__ == '__main__':
    main()
