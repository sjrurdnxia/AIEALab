# PPO from scratch – continuous (CarRacing‑v3 ready)
# -------------------------------------------------------------
# Minimal, **educational** implementation (~220 lines).
# Trains a Gaussian‑policy PPO agent on Gymnasium's CarRacing‑v3
# or any classic‑control discrete task by switching ENV_ID.
# -------------------------------------------------------------
import gymnasium as gym
import torch, torch.nn as nn, torch.optim as optim
import numpy as np, random, os, collections, time
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal, Categorical

# ------------------------------
# 1 ▸  Config / Hyper‑parameters
# ------------------------------
ENV_ID          = os.getenv("ENV", "CarRacing-v3")      # e.g. "CartPole-v1"
TOTAL_STEPS     = 250_000
ROLLOUT_STEPS   = 2048
MINIBATCH_SIZE  = 256
PPO_EPOCHS      = 10
GAMMA           = 0.99
LAMBDA          = 0.95
CLIP_EPS        = 0.2
LR              = 3e-4
ENTROPY_COEF    = 0.0
VALUE_COEF      = 0.5
MAX_GRAD_NORM   = 0.5
LOG_DIR         = os.path.expanduser("~/ppo_from_scratch/logs/ppo_scratch1")
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# fix seeds for (some) reproducibility
random.seed(0); np.random.seed(0); torch.manual_seed(0)

# ------------------------------------------------
# 2 ▸  Env helpers / obs‑action meta‑data
# ------------------------------------------------

def make_env(seed: int = 0):
    env = gym.make(ENV_ID, render_mode=None)
    env.reset(seed=seed)
    return env

env = make_env()
OBS_SHAPE = env.observation_space.shape
obs_dim   = int(np.prod(OBS_SHAPE))

continuous = isinstance(env.action_space, gym.spaces.Box)
if continuous:
    act_dim   = env.action_space.shape[0]
    act_low   = torch.as_tensor(env.action_space.low,  dtype=torch.float32, device=DEVICE)
    act_high  = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=DEVICE)
else:
    act_dim   = env.action_space.n

# -----------------------------------
# 3 ▸  Networks (Policy & Value)
# -----------------------------------
class PolicyNet(nn.Module):
    def __init__(self, continuous: bool):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(obs_dim, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
        )
        if continuous:
            self.mu_head  = nn.Linear(128, act_dim)
            self.log_std  = nn.Parameter(torch.zeros(act_dim))
        else:
            self.logits_head = nn.Linear(128, act_dim)
        self.continuous = continuous

    def forward(self, x):
        x = self.shared(x)
        if self.continuous:
            mu  = self.mu_head(x)
            std = self.log_std.exp().expand_as(mu)
            return Normal(mu, std)
        else:
            logits = self.logits_head(x)
            return Categorical(logits=logits)

class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(obs_dim, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)

policy   = PolicyNet(continuous).to(DEVICE)
value    = ValueNet().to(DEVICE)
optimizer = optim.Adam(list(policy.parameters())+list(value.parameters()), lr=LR)

writer = SummaryWriter(LOG_DIR)
global_ep = 0

# ---------------------------------------------------
# 4 ▸  Advantage / return helper (GAE)
# ---------------------------------------------------
Transition = collections.namedtuple("Transition", "obs act logp reward done value")

def compute_gae_rollout(rollout, next_value):
    rewards, values, dones, logps = [], [], [], []
    for tr in rollout:
        rewards.append(tr.reward)
        values.append(tr.value)
        dones.append(tr.done)
        logps.append(tr.logp)
    rewards  = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
    values   = torch.tensor(values + [next_value], dtype=torch.float32, device=DEVICE)
    dones    = torch.tensor(dones, dtype=torch.float32, device=DEVICE)
    gae = 0.0
    advantages = []
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + GAMMA * values[t+1] * (1-dones[t]) - values[t]
        gae   = delta + GAMMA * LAMBDA * (1-dones[t]) * gae
        advantages.insert(0, gae)
    advantages = torch.stack(advantages)
    returns    = advantages + values[:-1]
    logps      = torch.stack(logps)
    return advantages, returns, logps

# ------------------------------------
# 5 ▸  Training loop
# ------------------------------------
obs, _ = env.reset()
obs = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
step = 0
episode_return = 0.0

while step < TOTAL_STEPS:
    rollout = []
    for _ in range(ROLLOUT_STEPS):
        with torch.no_grad():
            dist   = policy(obs.unsqueeze(0))
            action = dist.sample()[0]
            logp   = dist.log_prob(action).sum() if continuous else dist.log_prob(action)
            v_est  = value(obs.unsqueeze(0))[0]
        if continuous:
            act_tanh = torch.tanh(action)
            act_env  = act_low + (act_tanh + 1) * (act_high - act_low) / 2
            act_np   = act_env.cpu().numpy()
        else:
            act_np   = action.cpu().numpy()

        next_obs, reward, terminated, truncated, _ = env.step(act_np)
        done = terminated or truncated
        episode_return += reward
        rollout.append(Transition(obs, action, logp, float(reward), done, v_est))
        obs = torch.tensor(next_obs, dtype=torch.float32, device=DEVICE)
        step += 1

        if done:
            writer.add_scalar("episode_reward", episode_return, global_ep)
            global_ep += 1
            episode_return = 0.0
            obs, _ = env.reset()
            obs = torch.tensor(obs, dtype=torch.float32, device=DEVICE)

        if step >= TOTAL_STEPS:
            break

    # ▸ GAE & returns
    with torch.no_grad():
        next_val = value(obs.unsqueeze(0))[0]
    adv, ret, old_logp = compute_gae_rollout(rollout, next_val)

    obs_b  = torch.stack([tr.obs for tr in rollout])
    act_b  = torch.stack([tr.act for tr in rollout]) if continuous else torch.tensor([tr.act for tr in rollout], device=DEVICE)

    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    for _ in range(PPO_EPOCHS):
        idx = torch.randperm(len(rollout), device=DEVICE)
        for start in range(0, len(rollout), MINIBATCH_SIZE):
            mb_idx = idx[start:start+MINIBATCH_SIZE]
            mb_obs, mb_act = obs_b[mb_idx], act_b[mb_idx]
            mb_adv, mb_ret, mb_oldp = adv[mb_idx], ret[mb_idx], old_logp[mb_idx]

            dist = policy(mb_obs)
            logp = dist.log_prob(mb_act).sum(-1) if continuous else dist.log_prob(mb_act)
            ratio = torch.exp(logp - mb_oldp)
            surr1, surr2 = ratio * mb_adv, torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()
            value_pred  = value(mb_obs)
            value_loss  = ((mb_ret - value_pred) ** 2).mean()
            entropy     = dist.entropy().mean()
            loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(list(policy.parameters()) + list(value.parameters()), MAX_GRAD_NORM)
            optimizer.step()

    if step % 10_000 < ROLLOUT_STEPS:
        print(f"Step {step:>7} | Return mean {ret.mean():6.1f} | Adv mean {adv.mean():6.3f}")
        writer.flush()

print("Training complete ✔️")
writer.close()
env.close()
