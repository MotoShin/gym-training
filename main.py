import gym
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np


def pillow_grayscale(screen):
    gamma22LUT  = [pow(x/255.0, 2.2)*255 for x in range(256)] * 3
    gamma045LUT = [pow(x/255.0, 1.0/2.2)*255 for x in range(256)]
    
    img = Image.fromarray(screen)

    img_resize = img.resize((84, 84))

    img_rgb = img_resize.convert("RGB") # any format to RGB
    img_rgbL = img_rgb.point(gamma22LUT)
    img_grayL = img_rgbL.convert("L")  # RGB to L(grayscale)
    img_gray = img_grayL.point(gamma045LUT)
    return np.asarray(img_gray) 

def pytorch_grayscale(screen):
    resize_and_grayscale = T.Compose([T.ToPILImage(),
                                T.Resize((84, 84), interpolation=Image.BICUBIC),
                                T.Grayscale(num_output_channels=1),
                                T.ToTensor()])

    return resize_and_grayscale(screen).numpy()[0]

env = gym.make('CartPole-v1').unwrapped
env.reset()
screen = env.render(mode='rgb_array')

img_gray_array_pillow = pillow_grayscale(screen)
img_gray_array_pytorch = pytorch_grayscale(screen)

fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

ax1.imshow(img_gray_array_pillow, cmap='gray')
ax1.annotate("pillow", xy=(30, 5))

ax2.imshow(img_gray_array_pytorch, cmap='gray')
ax2.annotate("torchvision", xy=(30, 5))

plt.show()
