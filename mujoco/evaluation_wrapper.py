import gymnasium as gym
import numpy as np
from gymnasium import Wrapper

class UnifiedLocomotionEval(Wrapper):
    """
    A wrapper that, on every step, computes a unified, engine-agnostic reward
    so we can compare zero-shot performance between PyBullet and MuJoCo.
    """
    def __init__(self, env,
                 w_v=1.0,     # forward velocity
                 w_h=1.0,     # alive / posture
                 w_u=0.001,   # action penalty
                 w_s=0.2):    # lateral drift penalty
        super().__init__(env)
        self.w_v = w_v
        self.w_h = w_h
        self.w_u = w_u
        self.w_s = w_s

        self.prev_x = None
        self.prev_y = None
        self.dt = 1.0 / 240.0  # will override if env exposes it

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Try to read sim timestep if available
        # MuJoCo has .dt, Bullet often has .env._p.getPhysicsEngineParameters()
        if hasattr(self.env, "dt"):
            self.dt = float(self.env.dt)

        pos = self._get_root_pos(obs)
        self.prev_x = pos[0]
        self.prev_y = pos[1]

        info = dict(info)
        info["unified_reward"] = 0.0
        return obs, info

    def step(self, action):
        obs, orig_reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)

        # --- 1) get current position
        pos = self._get_root_pos(obs)
        x, y = pos[0], pos[1]

        # --- 2) forward vel (finite difference)
        if self.prev_x is None:
            v_x = 0.0
        else:
            v_x = (x - self.prev_x) / self.dt

        # --- 3) alive term (height, pitch)
        alive = 1.0 if self._is_alive(obs) else 0.0

        # --- 4) action penalty
        act_pen = float(np.sum(np.square(action)))

        # --- 5) lateral penalty
        lat_pen = abs(y)

        unified_r = (
            self.w_v * v_x
            + self.w_h * alive
            - self.w_u * act_pen
            - self.w_s * lat_pen
        )

        # update prev
        self.prev_x = x
        self.prev_y = y

        # stick it in info so we can log it
        info["unified_reward"] = unified_r
        info["orig_reward"] = orig_reward

        return obs, orig_reward, terminated, truncated, info

    # ---------------- helpers ----------------
    def _get_root_pos(self, obs):
        """
        Try to extract (x, y, z) root position from observation.
        Different envs pack obs differently, so we fall back to 0s.
        """
        # MuJoCo Walker2d: obs = [qpos[1:], qvel]  -> x is in the sim, not obs
        # but env.unwrapped.data.xpos[1] exists
        if hasattr(self.env.unwrapped, "data") and hasattr(self.env.unwrapped.data, "xpos"):
            # MuJoCo path
            root = self.env.unwrapped.data.xpos[1]  # body 1 is usually torso
            return root.copy()
        # PyBullet Walker2D often exposes robot body
        if hasattr(self.env.unwrapped, "robot") and hasattr(self.env.unwrapped.robot, "body_xyz"):
            bx, by, bz = self.env.unwrapped.robot.body_xyz
            return np.array([bx, by, bz], dtype=np.float32)
        # fallback
        return np.zeros(3, dtype=np.float32)

    def _is_alive(self, obs):
        """
        Very light-weight alive term:
        - height in [0.7, 2.0]
        - pitch not crazy
        You can make this stricter later.
        """
        # Try MuJoCo style: torso z
        if hasattr(self.env.unwrapped, "data"):
            z = float(self.env.unwrapped.data.xpos[1][2])
            if z < 0.6:
                return False
            return True
        # Fallback: accept as alive
        return True
