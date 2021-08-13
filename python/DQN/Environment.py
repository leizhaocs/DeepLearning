import gym
import torch
import numpy as np
import torchvision.transforms as T

# gym environment wrapper
class CartPoleEnvManager():

    # constructor
    def __init__(self, sequence_len, screen_height, screen_width, device):

        self.device = device
        self.env = gym.make('CartPole-v0').unwrapped
        self.done = False
        self.screen_height = screen_height
        self.screen_width = screen_width
        self.sequences = []
        self.seq_length = sequence_len
        self.seq_pos = 0

        self.env.reset()
        for i in range(self.seq_length):
            screen = torch.zeros(1, 1, self.screen_height, self.screen_width)
            self.sequences.append(screen)

    # reset the game
    def reset(self):

        self.env.reset()

    # close the environment
    def close(self):

        self.env.close()

    # render the scene
    def render(self, mode='human'):

        return self.env.render(mode)

    # get the action space
    def num_actions_available(self):

        return self.env.action_space.n

    # let the environment take an action, get the reward and next state
    def take_action(self, action):

        _, reward, self.done, _ = self.env.step(action.item())
        return torch.tensor([reward], device=self.device)

    # get the state
    def get_state(self):

        if self.done:
            self.sequences[self.seq_pos] = torch.zeros(1, 1, self.screen_height, self.screen_width)
        else:
            self.sequences[self.seq_pos] = self.get_processed_screen()
        self.seq_pos += 1
        if self.seq_pos == self.seq_length:
            self.seq_pos = 0

        state = None
        for i in range(self.seq_length):
            index = (self.seq_pos+i) % self.seq_length
            screen = self.sequences[index]
            if state is None:
                state = screen
            else:
                state = torch.cat((state, screen), axis=1)
        return state.to(self.device)

    # preprocess the image, (1, 1, height, width)
    def get_processed_screen(self):

        screen = self.render('rgb_array')
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = np.dot(screen[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.float32)

        top = int(screen.shape[0] * 0.4)
        bottom = int(screen.shape[0] * 0.8)
        screen = screen[top:bottom]

        screen = torch.from_numpy(screen)
        screen = screen.unsqueeze(0)
        resize = T.Compose([T.ToPILImage(), T.Resize((self.screen_height, self.screen_width)), T.ToTensor()])
        return resize(screen).unsqueeze(0)
