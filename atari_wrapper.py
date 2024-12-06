# Borrow a lot from Tianshou
# https://github.com/thu-ml/tianshou/blob/master/examples/atari/atari_wrapper.py
import numpy as np
import os
from collections import deque
import gymnasium as gym
from gymnasium import spaces
import cv2
from typing import Any

def _parse_reset_result(reset_result: tuple) -> tuple[tuple, dict, bool]:
    contains_info = (
        isinstance(reset_result, tuple)
        and len(reset_result) == 2
        and isinstance(reset_result[1], dict)
    )
    if contains_info:
        return reset_result[0], reset_result[1], contains_info
    return reset_result, {}, contains_info

def get_space_dtype(obs_space: gym.spaces.Box) -> type[np.floating] | type[np.integer]:
    obs_space_dtype: type[np.integer] | type[np.floating]
    if np.issubdtype(obs_space.dtype, np.integer):
        obs_space_dtype = np.integer
    elif np.issubdtype(obs_space.dtype, np.floating):
        obs_space_dtype = np.floating
    else:
        raise TypeError(
            f"Unsupported observation space dtype: {obs_space.dtype}. "
            f"This might be a bug in tianshou or gymnasium, please report it!",
        )
    return obs_space_dtype


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0
        assert hasattr(env.unwrapped, "get_action_meanings")
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'
    
    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        _, info, return_info = _parse_reset_result(self.env.reset(**kwargs))
        noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        for _ in range(noops):
            step_result = self.env.step(self.noop_action)
            if len(step_result) == 4:
                obs, rew, done, info = step_result  # type: ignore[unreachable]  # mypy doesn't know that Gym version <0.26 has only 4 items (no truncation)
            else:
                obs, rew, term, trunc, info = step_result
                done = term or trunc
            if done:
                obs, info, _ = _parse_reset_result(self.env.reset())
        if return_info:
            return obs, info
        return obs, {}


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        obs_list = []
        total_reward = 0.0
        for _ in range(self._skip):
            step_result = self.env.step(action)
            obs, reward, term, trunc, info = step_result
            done = term or trunc
            obs_list.append(obs)
            total_reward += float(reward)
            if done:
                break
        max_frame = np.max(obs_list[-2:], axis=0)
        return max_frame, total_reward, term, trunc, info


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True
        self._return_info = False

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        step_result = self.env.step(action)
        obs, reward, term, trunc, info = step_result
        done = term or trunc
        reward = float(reward)
        self.was_real_done = done
        # check current lives, make loss of life terminal, then update lives to handle bonus lives
        assert hasattr(self.env.unwrapped, "ale")
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            done = True
            term = True
        self.lives = lives
        return obs, reward, term, trunc, info

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        """Calls the Gym environment reset, only when lives are exhausted.

        This way all states are still reachable even though lives are episodic, and
        the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs, info, self._return_info = _parse_reset_result(self.env.reset(**kwargs))
        else:
            # no-op step to advance from terminal/lost life state
            step_result = self.env.step(0)
            obs, info = step_result[0], step_result[-1]
        assert hasattr(self.env.unwrapped, "ale")
        self.lives = self.env.unwrapped.ale.lives()
        if self._return_info:
            return obs, info
        return obs, {}
 

class FireResetEnv(gym.Wrapper):
    """Take action on reset for environments that are fixed until firing."""
    def __init__(self, env):
        super().__init__(env)
        assert hasattr(env.unwrapped, "get_action_meanings")
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs: Any) -> tuple[Any, dict]:
        _, _, return_info = _parse_reset_result(self.env.reset(**kwargs))
        obs = self.env.step(1)[0]
        return obs, {}
    

class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.reward_range = (-1, 1)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(float(reward))


class WarpFrame(gym.ObservationWrapper):
    """Warp frames to 84x84 as done in the Nature paper and later work.

    :param gym.Env env: the environment to wrap.
    """
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.size = 84
        obs_space = env.observation_space
        assert isinstance(obs_space, gym.spaces.Box)
        obs_space_dtype = get_space_dtype(obs_space)
        self.observation_space = gym.spaces.Box(
            low=np.min(obs_space.low),
            high=np.max(obs_space.high),
            shape=(self.size, self.size),
            dtype=obs_space_dtype,
        )

    def observation(self, frame: np.ndarray) -> np.ndarray:
        """Returns the current observation from a frame."""
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return cv2.resize(frame, (self.size, self.size), interpolation=cv2.INTER_AREA)


class FrameStack(gym.Wrapper):
    """Stack n_frames last frames.

    :param gym.Env env: the environment to wrap.
    :param int n_frames: the number of frames to stack.
    """

    def __init__(self, env: gym.Env, n_frames: int) -> None:
        super().__init__(env)
        self.n_frames: int = n_frames
        self.frames: deque[tuple[Any, ...]] = deque([], maxlen=n_frames)
        obs_space = env.observation_space
        obs_space_shape = env.observation_space.shape
        assert obs_space_shape is not None
        shape = (n_frames, *obs_space_shape)
        assert isinstance(obs_space, gym.spaces.Box)
        obs_space_dtype = get_space_dtype(obs_space)
        self.observation_space = gym.spaces.Box(
            low=np.min(obs_space.low),
            high=np.max(obs_space.high),
            shape=shape,
            dtype=obs_space_dtype,
        )

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict]:
        obs, info, return_info = _parse_reset_result(self.env.reset(**kwargs))
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return (self._get_ob(), info) if return_info else (self._get_ob(), {})

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        step_result = self.env.step(action)
        obs, reward, term, trunc, info = step_result
        self.frames.append(obs)
        reward = float(reward)
        return self._get_ob(), reward, term, trunc, info

    def _get_ob(self) -> np.ndarray:
        return np.stack(self.frames, axis=0)


class ScaledFloatFrame(gym.ObservationWrapper):
    """Normalize observations to 0~1.

    :param gym.Env env: the environment to wrap.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        obs_space = env.observation_space
        assert isinstance(obs_space, gym.spaces.Box)
        low = np.min(obs_space.low)
        high = np.max(obs_space.high)
        self.bias = low
        self.scale = high - low
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=obs_space.shape,
            dtype=np.float32,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return (observation - self.bias) / self.scale


def wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=True, scale=False, warp_frame: bool = True):
    """Configure environment for DeepMind-style Atari.
    """
    assert 'NoFrameskip' in env.spec.id
    wrapped_env = NoopResetEnv(env, noop_max=30)
    wrapped_env = MaxAndSkipEnv(wrapped_env, skip=4)
    if episode_life:
        wrapped_env = EpisodicLifeEnv(wrapped_env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        wrapped_env = FireResetEnv(wrapped_env)
    if warp_frame:
        wrapped_env = WarpFrame(wrapped_env)
    if scale:
        wrapped_env = ScaledFloatFrame(wrapped_env)
    if clip_rewards:
        wrapped_env = ClipRewardEnv(wrapped_env)
    if frame_stack:
        wrapped_env = FrameStack(wrapped_env, 4)
    return wrapped_env


def make_atari_env(env_id, mode):
    env_train = gym.make(env_id, render_mode=mode)
    env_train = wrap_deepmind(env_train, episode_life=True, clip_rewards=True, frame_stack=True)
    env_test = gym.make(env_id, render_mode=mode)
    env_test = wrap_deepmind(env_test, episode_life=False, clip_rewards=False, frame_stack=True)
    return env_train, env_test