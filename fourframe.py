import gym
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from collections import deque
import copy


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

def get_cart_location(env, screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0) # MIDDLE OF CART

def get_screen(env):
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height*0.4):int(screen_height*0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(env, screen_width)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2, cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    # screen = np.ascontiguousarray(screen, dtype=np.float32)
    # Resize, and add a batch dimension (BCHW)
    return screen

def pillow_grayscale(screen):
    gamma22LUT  = [pow(x/255.0, 2.2)*255 for x in range(256)] * 3
    gamma045LUT = [pow(x/255.0, 1.0/2.2)*255 for x in range(256)]

    screen = torch.from_numpy(np.array(screen)).permute(1, 2, 0).numpy() 
    img = Image.fromarray(screen)

    img_resize = img.resize((84, 84))

    img_rgb = img_resize.convert("RGB") # any format to RGB
    img_rgbL = img_rgb.point(gamma22LUT)
    img_grayL = img_rgbL.convert("L")  # RGB to L(grayscale)
    img_gray = img_grayL.point(gamma045LUT)
    return np.asarray(img_gray)

env = gym.make('CartPole-v1').unwrapped
env.reset()

screens = InputScreens()
done = False
while not done:
    # screen = env.render(mode='rgb_array')
    # gray_screen = resize_and_grayscale(screen).numpy()[0]
    screen = get_screen(env)
    # screens.push(resize_and_grayscale(screen))
    gray_screen = copy.deepcopy(pillow_grayscale(screen))
    screens.push(torch.from_numpy(gray_screen))
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
