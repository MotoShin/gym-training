import gym
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from collections import deque


class InputScreens(object):
    def __init__(self):
        self.screens = deque([np.array([[0 for _ in range(84)] for _ in range(84)]) for _ in range(4)])

    def push(self, screen):
        self.screens.append(screen)
        self.screens.popleft()

    def get(self):
        return self.screens

resize_and_grayscale = T.Compose([T.ToPILImage(),
                                T.Resize((84, 84), interpolation=Image.BICUBIC),
                                T.Grayscale(num_output_channels=1),
                                T.ToTensor()])

env = gym.make('CartPole-v1').unwrapped
env.reset()

screens = InputScreens()
done = False
while not done:
    screen = env.render(mode='rgb_array')
    gray_screen = resize_and_grayscale(screen).numpy()[0]
    screens.push(gray_screen)
    _, _, done, _ = env.step(env.action_space.sample())

env.close()

fig = plt.figure()

ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)

ax1.imshow(screens.get()[0], cmap='gray')
ax1.annotate("1", xy=(30, 5))
ax2.imshow(screens.get()[1], cmap='gray')
ax2.annotate("2", xy=(30, 5))
ax3.imshow(screens.get()[2], cmap='gray')
ax3.annotate("3", xy=(30, 5))
ax4.imshow(screens.get()[3], cmap='gray')
ax4.annotate("4", xy=(30, 5))

plt.show()
