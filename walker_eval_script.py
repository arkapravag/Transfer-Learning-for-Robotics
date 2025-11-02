import gym
import pybullet_envs
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# -----------------------------
# Parameters
# -----------------------------
ENV_ID = "Walker2DBulletEnv-v0"
MODEL_PATH = "ppo_walker.zip"
N_EVAL_EPISODES = 100

# -----------------------------
# Create environment
# -----------------------------
# We don't need rendering for evaluation, so we can create a standard env
env = gym.make(ENV_ID)

# -----------------------------
# Load the trained model
# -----------------------------
try:
    model = PPO.load(MODEL_PATH, env=env)
    print(f"Model loaded from {MODEL_PATH}")
except FileNotFoundError:
    print(f"Error: Model not found at {MODEL_PATH}")
    print("Please run the training script first to generate the model file.")
    exit()

# -----------------------------
# Evaluate the model
# -----------------------------
# evaluate_policy runs the agent for N_EVAL_EPISODES and returns the mean and standard deviation of the rewards
print(f"Evaluating model over {N_EVAL_EPISODES} episodes...")
mean_reward, std_reward = evaluate_policy(
    model, 
    env, 
    n_eval_episodes=N_EVAL_EPISODES, 
    deterministic=True # Use deterministic actions for evaluation
)

print("\n--- Evaluation Complete ---")
print(f"Environment: {ENV_ID}")
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
print("---------------------------\n")

# -----------------------------
# Cleanup
# -----------------------------
env.close()

