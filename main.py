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

def 

env = gym.make('CartPole-v1').unwrapped
env.reset()
screen = env.render(mode='rgb_array')

img_gray_array = pillow_grayscale(screen)

img_gray = Image.fromarray(img_gray_array)
plt.imshow(img_gray, cmap="Greys_r")
plt.show()

img_gray_array = np.asarray(img_gray)
print(img_gray_array)
