
import gymnasium as gym
import ale_py

from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation


# Shared environment configuration
ENV_NAME = "ALE/Qbert-v5"

SCREEN_SIZE = 84
FRAME_STACK_SIZE = 4
FRAME_SKIP = 4


def make_env(render_mode=None):
    """
    Create and preprocess the Q*bert Atari environment.
    """

    gym.register_envs(ale_py)

    env = gym.make(
        ENV_NAME,
        frameskip=1,
        repeat_action_probability=0.25,
        render_mode=render_mode,
    )

    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=FRAME_SKIP,
        screen_size=SCREEN_SIZE,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        scale_obs=False,
    )

    env = FrameStackObservation(
        env,
        stack_size=FRAME_STACK_SIZE,
    )

    return env
