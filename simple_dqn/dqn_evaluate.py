import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from matplotlib import set_loglevel
from PIL import Image
from simple_dqn import DQN, SAVE_DIR, device, h_dims, n_actions, o_space, select_action


def get_screen(env) -> None:
    def get_cart_location(screen_width):
        world_width = env.x_threshold * 2
        scale = screen_width / world_width
        return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

    resize = T.Compose(
        [T.ToPILImage(), T.Resize(40, interpolation=Image.CUBIC), T.ToTensor()]
    )

    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render(mode="rgb_array").transpose((2, 0, 1))
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height * 0.4) : int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(screen_width)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(
            cart_location - view_width // 2, cart_location + view_width // 2
        )
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    # Resize, and add a batch dimension (BCHW)
    return resize(screen)


def eval_policy(dir: str, is_render=False):
    """function to evaluate the policy"""
    env = gym.make("CartPole-v0").unwrapped
    policy_net = DQN(o_space, h_dims, n_actions).to(device)
    # load trained weight
    policy_net.load_net(SAVE_DIR)
    policy_net.eval()
    T_MAX = 100
    time_step = 0
    im_list = []
    angle_list = []
    while 1:
        state = env.reset()
        state = torch.tensor(state)
        state = state.unsqueeze(0)
        action = select_action(state, policy_net, 0)
        state, reward, done, _ = env.step(action.item())
        angle_list.append(state[2])
        if is_render:
            img = get_screen(env)
            im_list.append(img)
        if done or time_step == T_MAX:
            break
        time_step += 1
    print("time step", time_step)
    plt.plot(list(range(len(angle_list))), angle_list)
    plt.savefig("pole_angle.png")


if __name__ == "__main__":
    # env = gym.make("CartPole-v0")
    # render = lambda: plt.imshow(env.render(mode="rgb_array"))
    # env.reset()
    # render()

    eval_policy(SAVE_DIR, is_render=False)
