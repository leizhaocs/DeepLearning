import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import torch
import torch.optim as optim
import torch.nn.functional as F

from DQN import DQN
from ReplayMem import ReplayMemory
from Strategy import EpsilonGreedyStrategy
from Agent import Agent
from Environment import CartPoleEnvManager
from QValues import QValues

###########################################
# helper functions
###########################################

# get the average of all values, computed of if a speicific number of values have been collected 
def get_moving_average(period, values):

    values = torch.tensor(values, dtype=torch.float)
    if len(values) >= period:
        moving_avg = values.unfold(dimension=0, size=period, step=1).mean(dim=1).flatten(start_dim=0)
        moving_avg = torch.cat((torch.zeros(period-1), moving_avg))
        return moving_avg.numpy()
    else:
        moving_avg = torch.zeros(len(values))
        return moving_avg.numpy()

# plot the result
def plot(values, moving_avg_period):

    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(values)
    moving_avg = get_moving_average(moving_avg_period, values)
    plt.plot(moving_avg)
    plt.pause(0.00001)
    print("Episode", len(values), "\n", moving_avg_period, "episode moving avg:", moving_avg[-1])

# eg. (s_1, a_1, r_1, n_1), (s_2, a_2, r_2, n_2)
# =>  (s_1, s_2), (a_1, a_2), (r_1, r_2), (n_1, n_2)
def extract_tensors(experiences):

    batch = Experience(*zip(*experiences))
    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.next_state)
    return (t1, t2, t3, t4)

###########################################
# global variables
###########################################

batch_size = 256       # batch size
gamma = 0.999          # discount rate
eps_start = 1          # epsilon start value
eps_end = 0.01         # epsilon end value
eps_decay = 0.001      # epsilon decay value
target_update = 10     # copy weights from policy net to target net every this episodes
memory_size = 100000   # repaly memory size
lr = 0.001             # learning rate
num_episodes = 2000    # total number of episodes to run
screen_height = 40#84
screen_width = 90#84
sequence_len = 4

###########################################
# execute
###########################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
em = CartPoleEnvManager(sequence_len, screen_height, screen_width, device)
strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
agent = Agent(strategy, em.num_actions_available(), device)
memory = ReplayMemory(memory_size)
Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward'))

policy_net = DQN(sequence_len, screen_height, screen_width, em.num_actions_available()).to(device)
target_net = DQN(sequence_len, screen_height, screen_width, em.num_actions_available()).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)

episode_durations = []
for episode in range(num_episodes):
    em.reset()
    state = em.get_state()

    for timestep in count():
        action = agent.select_action(state, policy_net)
        reward = em.take_action(action)
        next_state = em.get_state()
        memory.push(Experience(state, action, next_state, reward))
        state = next_state

        #screen1 = state.cpu().numpy().squeeze(0)
        #screen1 = screen1[0]
        #print(screen1.shape)
        #plt.figure(3)
        #plt.imshow(screen1, interpolation='none', cmap='gray')
        #plt.title('pp')
        #plt.pause(0.00001)

        if memory.can_provide_sample(batch_size):
            experiences = memory.sample(batch_size)
            states, actions, rewards, next_states = extract_tensors(experiences)

            current_q_values = QValues.get_current(policy_net, states, actions)
            next_q_values = QValues.get_next(target_net, next_states)
            target_q_values = (next_q_values * gamma) + rewards

            loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if em.done:
            episode_durations.append(timestep)
            plot(episode_durations, 100)
            break

    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

em.close()
