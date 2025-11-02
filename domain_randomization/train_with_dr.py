import os
import gymnasium as gym
import pybullet_envs_gymnasium
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

# =========================================================
# Progressive Domain Randomization Wrapper
# =========================================================
class ProgressiveRandomizedWalker2d(gym.Wrapper):
    def __init__(self, env,
                 total_timesteps,
                 mass_range=(0.8, 1.2),
                 friction_range=(0.8, 1.2),
                 damping_range=(0.8, 1.2),
                 gravity_range=(0.95, 1.05),
                 actuator_range=(0.9, 1.1),
                 obs_noise_std=0.01,
                 torque_noise_std=0.02,
                 timestep_range=(1/400, 1/250),
                 enable_timestep_randomization=True,
                 auto_expand_randomization=False,
                 expand_every=750_000):
        super().__init__(env)
        self.total_timesteps = total_timesteps
        self.enable_timestep_randomization = enable_timestep_randomization
        self.auto_expand_randomization = auto_expand_randomization
        self.expand_every = expand_every
        self.expansion_factor = 0.05
        self.last_expansion_step = 0

        self.param_ranges = dict(
            mass=(mass_range, (0.6, 1.4)),
            friction=(friction_range, (0.6, 1.4)),
            damping=(damping_range, (0.5, 1.5)),
            gravity=(gravity_range, (0.9, 1.1)),
            actuator=(actuator_range, (0.8, 1.2)),
        )
        self.obs_noise_std_base = obs_noise_std
        self.torque_noise_std_base = torque_noise_std
        self.timestep_range_base = timestep_range
        self.alpha = 0.0
        self.last_randomization_params = {}

    def set_progress(self, timestep):
        self.alpha = min(timestep / self.total_timesteps, 1.0)
        # Optionally expand ranges
        if self.auto_expand_randomization and timestep - self.last_expansion_step >= self.expand_every:
            self._expand_randomization()
            self.last_expansion_step = timestep

    def _expand_randomization(self):
        """Gradually widen randomization ranges."""
        for key, (base, full) in self.param_ranges.items():
            new_min = max(0.5, base[0] - self.expansion_factor)
            new_max = min(1.5, base[1] + self.expansion_factor)
            self.param_ranges[key] = ((new_min, new_max), full)
        print(f"🔧 Expanded randomization ranges at {self.last_expansion_step} steps")

    def _interp_range(self, base, full):
        low = np.interp(self.alpha, [0, 1], [base[0], full[0]])
        high = np.interp(self.alpha, [0, 1], [base[1], full[1]])
        return (low, high)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._randomize_physics()
        obs = self._add_obs_noise(obs)
        return obs, info

    def step(self, action):
        action += np.random.normal(0, self.torque_noise_std_base * self.alpha, size=action.shape)
        action = np.clip(action, -1, 1)
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._add_obs_noise(obs)
        return obs, reward, terminated, truncated, info

    def _randomize_physics(self):
        if hasattr(self.env.unwrapped, "_p"):  # PyBullet
            p = self.env.unwrapped._p
            robot = self.env.unwrapped.robot

            mass_rng = self._interp_range(*self.param_ranges["mass"])
            friction_rng = self._interp_range(*self.param_ranges["friction"])
            damping_rng = self._interp_range(*self.param_ranges["damping"])
            gravity_rng = self._interp_range(*self.param_ranges["gravity"])
            actuator_rng = self._interp_range(*self.param_ranges["actuator"])

            if self.enable_timestep_randomization:
                ts_rng = (
                    np.interp(self.alpha, [0, 1], [self.timestep_range_base[0], 1/200]),
                    np.interp(self.alpha, [0, 1], [self.timestep_range_base[1], 1/600])
                )
                p.setPhysicsEngineParameter(fixedTimeStep=np.random.uniform(*ts_rng))

            p.setGravity(0, 0, -9.8 * np.random.uniform(*gravity_rng))

            for j in range(p.getNumJoints(robot.robot_body.bodies[0])):
                mass = p.getDynamicsInfo(robot.robot_body.bodies[0], j)[0]
                new_mass = mass * np.random.uniform(*mass_rng)
                p.changeDynamics(robot.robot_body.bodies[0], j, mass=new_mass)
                p.changeDynamics(robot.robot_body.bodies[0], j, lateralFriction=np.random.uniform(*friction_rng))
                p.changeDynamics(robot.robot_body.bodies[0], j, linearDamping=np.random.uniform(*damping_rng))

            if hasattr(robot, "motor_power"):
                robot.motor_power *= np.random.uniform(*actuator_rng)

            self.last_randomization_params = dict(
                alpha=self.alpha,
                mass_range=mass_rng,
                friction_range=friction_rng,
                damping_range=damping_rng,
                gravity_range=gravity_rng,
                actuator_range=actuator_rng,
            )

    def _add_obs_noise(self, obs):
        noise = np.random.normal(0, self.obs_noise_std_base * self.alpha, size=obs.shape)
        return obs + noise


# =========================================================
# Callback: Logging, Plotting, and Randomization Progression
# =========================================================
class ProgressiveDRLoggingCallback(BaseCallback):
    def __init__(self, env_wrapper, log_excel_path="training_log.xlsx", plot_every=1_000_000, verbose=0):
        super().__init__(verbose)
        self.env_wrapper = env_wrapper
        self.log_excel_path = log_excel_path
        self.plot_every = plot_every
        self.instant_rewards = []
        self.timesteps = []
        self.run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def _on_step(self):
        timestep = self.num_timesteps
        self.env_wrapper.set_progress(timestep)
        reward = float(self.locals.get("rewards", [0])[-1])
        self.instant_rewards.append(reward)
        self.timesteps.append(timestep)

        # Plot every N steps
        if timestep % self.plot_every == 0 and timestep > 0:
            self._save_reward_plot(timestep)
        return True

    def _save_reward_plot(self, timestep):
        """Save convergence plot for instantaneous rewards."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.timesteps, self.instant_rewards, linewidth=1.2)
        plt.title(f"Reward Convergence up to {timestep:,} Steps")
        plt.xlabel("Timesteps")
        plt.ylabel("Instantaneous Reward")
        plt.grid(True)
        plt.tight_layout()
        fname = f"reward_convergence_{timestep//1_000_000}M.png"
        plt.savefig(fname, dpi=300)
        plt.close()
        print(f"📈 Saved reward convergence plot: {fname}")

    def _on_training_end(self):
        df = pd.DataFrame({
            "timestep": self.timesteps,
            "instant_reward": self.instant_rewards,
        })

        # Add last randomization info
        rand_info = self.env_wrapper.last_randomization_params
        for key, val in rand_info.items():
            df[key] = str(val)

        mode = "a" if os.path.exists(self.log_excel_path) else "w"
        writer = pd.ExcelWriter(self.log_excel_path, mode=mode, engine="openpyxl", if_sheet_exists="overlay")
        with writer:
            df.to_excel(writer, sheet_name="Rewards", index=False)

        print(f"📊 Logged training data to {self.log_excel_path}")


# =========================================================
# Train or Resume PPO with Progressive DR
# =========================================================
ENV_ID = "Walker2DBulletEnv-v0"
# TOTAL_TIMESTEPS = 6_000_000
TOTAL_TIMESTEPS = 10_000
MODEL_PATH = "ppo_walker_progressive_dr.zip"
VECNORM_PATH = "vec_normalize.pkl"
EXCEL_LOG_PATH = "training_log.xlsx"
SEED = 42

base_env = gym.make(ENV_ID)
dr_env = ProgressiveRandomizedWalker2d(
    base_env,
    total_timesteps=TOTAL_TIMESTEPS,
    auto_expand_randomization=True,   # turn off for fixed ranges
    expand_every=750_000              # widen ranges every 750k steps
)
vec_env = DummyVecEnv([lambda: dr_env])

# Load or create VecNormalize
if os.path.exists(VECNORM_PATH):
    print("🔄 Loading VecNormalize statistics...")
    vec_env = VecNormalize.load(VECNORM_PATH, vec_env)
else:
    print("🚀 Initializing new VecNormalize...")
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

callback = ProgressiveDRLoggingCallback(dr_env, log_excel_path=EXCEL_LOG_PATH, plot_every=10_000)

# Load or initialize model
if os.path.exists(MODEL_PATH):
    print(f"🔄 Found checkpoint at {MODEL_PATH}. Loading model...")
    model = PPO.load(MODEL_PATH, env=vec_env, seed=SEED)
    print("✅ Model loaded. Continuing training...")
else:
    print("🚀 No checkpoint found. Starting fresh training...")
    model = PPO("MlpPolicy", vec_env, verbose=1, seed=SEED)

# Train and save
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)

model.save(MODEL_PATH)
vec_env.save(VECNORM_PATH)
print(f"✅ Model saved as {MODEL_PATH}")
print(f"💾 VecNormalize stats saved as {VECNORM_PATH}")
