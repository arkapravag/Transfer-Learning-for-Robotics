import numpy as np
import gymnasium as gym
import time

class RandomizedWalker2d(gym.Wrapper):
    """
    Comprehensive Domain Randomization wrapper for PyBullet or MuJoCo Walker2D.

    Randomizes:
      - body mass
      - geom friction
      - joint damping
      - gravity
      - actuator gain/bias (motor strength)
      - timestep (integration frequency)
      - observation noise
    """
    def __init__(self, env,
                 mass_range=(0.6, 1.4),
                 friction_range=(0.6, 1.4),
                 damping_range=(0.5, 1.5),
                 gravity_range=(0.9, 1.1),
                 actuator_range=(0.8, 1.2),
                 obs_noise_std=0.02,
                 timestep_range=(1/300, 1/200),
                 enable_timestep_randomization=True):
        super().__init__(env)
        self.mass_range = mass_range
        self.friction_range = friction_range
        self.damping_range = damping_range
        self.gravity_range = gravity_range
        self.actuator_range = actuator_range
        self.obs_noise_std = obs_noise_std
        self.timestep_range = timestep_range
        self.enable_timestep_randomization = enable_timestep_randomization
        self.step_count = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._randomize_physics()
        self.step_count = 0
        # add small observation noise
        obs = self._add_obs_noise(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._add_obs_noise(obs)
        self.step_count += 1
        return obs, reward, terminated, truncated, info

    # --------------------------------------------------------
    # Internal randomization helpers
    # --------------------------------------------------------
    def _randomize_physics(self):
        """
        Apply randomization to physical and simulation parameters.
        Works for both MuJoCo-style and PyBullet-style envs.
        """
        if hasattr(self.env.unwrapped, "model"):
            # ---------------- MuJoCo ----------------
            model = self.env.unwrapped.model

            # Randomize mass, friction, damping
            model.body_mass[:] *= np.random.uniform(*self.mass_range)
            model.geom_friction[:, 0] *= np.random.uniform(*self.friction_range)
            model.dof_damping[:] *= np.random.uniform(*self.damping_range)

            # Randomize gravity
            model.opt.gravity[:] = [0, 0, -9.81 * np.random.uniform(*self.gravity_range)]

            # Randomize actuator gain/bias (motor strength)
            if hasattr(model, "actuator_gainprm"):
                model.actuator_gainprm[:, 0] *= np.random.uniform(*self.actuator_range)
            if hasattr(model, "actuator_biasprm"):
                model.actuator_biasprm[:, 1] *= np.random.uniform(*self.actuator_range)

            # Randomize timestep
            if self.enable_timestep_randomization and hasattr(model.opt, "timestep"):
                model.opt.timestep = np.random.uniform(*self.timestep_range)

        elif hasattr(self.env.unwrapped, "_p"):
            # ---------------- PyBullet ----------------
            p = self.env.unwrapped._p
            robot = self.env.unwrapped.robot

            # Randomize timestep
            if self.enable_timestep_randomization:
                p.setPhysicsEngineParameter(fixedTimeStep=np.random.uniform(*self.timestep_range))

            # Randomize gravity
            p.setGravity(0, 0, -9.8 * np.random.uniform(*self.gravity_range))

            # Randomize body mass and friction
            for j in range(p.getNumJoints(robot.robot_body.bodies[0])):
                mass = p.getDynamicsInfo(robot.robot_body.bodies[0], j)[0]
                new_mass = mass * np.random.uniform(*self.mass_range)
                p.changeDynamics(robot.robot_body.bodies[0], j, mass=new_mass)

                friction = np.random.uniform(*self.friction_range)
                p.changeDynamics(robot.robot_body.bodies[0], j, lateralFriction=friction)

                damping = np.random.uniform(*self.damping_range)
                p.changeDynamics(robot.robot_body.bodies[0], j, linearDamping=damping)

            # Randomize actuator power if available
            if hasattr(robot, "motor_power"):
                robot.motor_power *= np.random.uniform(*self.actuator_range)

    def _add_obs_noise(self, obs):
        """Adds Gaussian sensor noise to observation."""
        noise = np.random.normal(0, self.obs_noise_std, size=obs.shape)
        return obs + noise
