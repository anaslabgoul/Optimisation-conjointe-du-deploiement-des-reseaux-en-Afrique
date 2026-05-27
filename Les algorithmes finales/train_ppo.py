from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from ng_deployment_env import NGDeploymentEnv
# ■■ 1. Create vectorised environments (parallelism) ■■■■■■■■■■
# Running 8 environments in parallel means 8x more data per second
data_dir = '/Users/anasstakfa/Desktop/PSC/instance_PSC_MAYENNE/INPUTS'
n_envs = 8
def make_env():
    return NGDeploymentEnv(data_dir=data_dir, noise_std=0.05)
vec_env = DummyVecEnv([make_env] * n_envs) # 8 parallel environments
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, gamma=0.99)
model = PPO(
policy = 'MlpPolicy', # Multi-Layer Perceptron policy
env = vec_env,
learning_rate = 3e-4, # Adam optimizer step size
n_steps = 128, # rollout length per env before update
batch_size = 256, # mini-batch size for gradient updates
n_epochs = 10, # number of gradient steps per rollout
gamma = 0.99, # discount factor
gae_lambda = 0.95, # GAE lambda (variance reduction)
clip_range = 0.2, # PPO clipping epsilon
ent_coef = 0.01, # entropy bonus coefficient
vf_coef = 0.5, # value function loss coefficient
max_grad_norm = 0.5, # gradient clipping
policy_kwargs = dict( # neural network architecture
net_arch = [256, 256, 128] # 3-layer MLP, 256-256-128 neurons
),
verbose = 1)# print training logs
eval_env = DummyVecEnv([make_env])
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)
eval_cb = EvalCallback(
eval_env, eval_freq=5000, n_eval_episodes=10,
best_model_save_path='./best_model/',
verbose=1
)
model.learn(total_timesteps=100_000, callback=eval_cb)
model.save('ng_deployment_ppo')
vec_env.save('ng_deployment_norm.pkl') 