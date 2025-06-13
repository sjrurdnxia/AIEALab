# dqn_car_racing.py
import os, random, math, time
from collections import deque, namedtuple
from typing import Deque, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ─────────────────────────────── Globals ────────────────────────────────────
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENV_ID      = "CarRacing-v2"
STACK       = 4
N_ACTIONS   = 5                    # left, right, gas, brake, noop
MEM_CAP     = 100_000
N_STEPS     = 3                    # n-step return (1 = classic DQN)
GAMMA       = 0.99
BATCH       = 64
LR          = 2.5e-4
TARGET_SYNC = 5_000                # steps
TRAIN_START = 10_000               # fill replay before learning
TOTAL_STEPS = 400_000
EPS_START   = 1.0
EPS_END     = .05
EPS_DECAY   = 300_000              # steps
SAVE_EVERY  = 25_000               # steps

Transition = namedtuple("T", "state action reward next_state done")

# ────────────────────────────── Utils ───────────────────────────────────────
def preprocess(frame: np.ndarray) -> np.ndarray:
    """RGB(96,96,3) -> grayscale(84,84) uint8"""
    import cv2
    gray   = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized= cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return resized

def stack_reset(deq: Deque[np.ndarray], frame: np.ndarray):
    deq.clear()
    deq.extend([frame] * STACK)

def epsilon_by_step(step: int) -> float:
    return EPS_END + (EPS_START - EPS_END) * math.exp(-step / EPS_DECAY)

# ────────────────────────── Replay Buffer ───────────────────────────────────
class ReplayBuffer:
    def __init__(self, cap: int):
        self.buf: Deque[Transition] = deque(maxlen=cap)

    def push(self, *args):
        self.buf.append(Transition(*args))

    def sample(self, k: int):
        batch = random.sample(self.buf, k)
        s  = torch.as_tensor(np.stack([b.state      for b in batch]), device=DEVICE)
        a  = torch.as_tensor([b.action  for b in batch], device=DEVICE)
        r  = torch.as_tensor([b.reward  for b in batch], device=DEVICE)
        s2 = torch.as_tensor(np.stack([b.next_state for b in batch]), device=DEVICE)
        d  = torch.as_tensor([b.done    for b in batch], device=DEVICE)
        return s, a, r, s2, d

    def __len__(self): return len(self.buf)

# ───────────────────────────── Dueling DQN ──────────────────────────────────
class DuelingCNN(nn.Module):
    def __init__(self, n_actions: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(STACK, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),    nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),    nn.ReLU(),
        )
        # compute conv output size
        with torch.no_grad():
            dummy = torch.zeros(1, STACK, 84, 84)
            conv_out = self.features(dummy).view(1, -1).size(1)
        self.fc_val = nn.Sequential(nn.Linear(conv_out, 512), nn.ReLU(),
                                    nn.Linear(512, 1))
        self.fc_adv = nn.Sequential(nn.Linear(conv_out, 512), nn.ReLU(),
                                    nn.Linear(512, n_actions))

        # He init
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() / 255.0
        feat = self.features(x).flatten(1)
        val  = self.fc_val(feat)
        adv  = self.fc_adv(feat)
        return val + adv - adv.mean(dim=1, keepdim=True)

    def act(self, state: np.ndarray, eps: float) -> int:
        if random.random() < eps:
            return random.randrange(N_ACTIONS)
        with torch.no_grad():
            t = torch.as_tensor(state, device=DEVICE).unsqueeze(0)
            return int(self(t).argmax(1).item())

# ───────────────────────────── Training Loop ────────────────────────────────
def main():
    env = gym.make(ENV_ID, continuous=False, render_mode=None)
    online = DuelingCNN(N_ACTIONS).to(DEVICE)
    target = DuelingCNN(N_ACTIONS).to(DEVICE).eval()
    target.load_state_dict(online.state_dict())
    opt = torch.optim.Adam(online.parameters(), lr=LR, eps=1e-4)
    memory = ReplayBuffer(MEM_CAP)

    rewards_history, ep_reward, step = [], 0.0, 0
    frame_stack: Deque[np.ndarray] = deque(maxlen=STACK)
    obs, _ = env.reset()
    f = preprocess(obs)
    stack_reset(frame_stack, f)

    nstep_q: Deque[Tuple[np.ndarray, int, float]] = deque(maxlen=N_STEPS)

    while step < TOTAL_STEPS:
        eps = epsilon_by_step(step)
        state = np.stack(frame_stack, axis=0)
        action = online.act(state, eps)

        obs_next, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        reward = np.clip(reward, -1.0, 1.0)   # stabilise gradients

        f_next = preprocess(obs_next)
        frame_stack.append(f_next)
        next_state = np.stack(frame_stack, axis=0)

        nstep_q.append((state, action, reward))
        if len(nstep_q) == N_STEPS:
            R = sum(t[2]*(GAMMA**i) for i,t in enumerate(nstep_q))
            s0, a0, _ = nstep_q[0]
            memory.push(s0, a0, R, next_state, done)

        ep_reward += reward
        obs = obs_next
        step += 1

        # Episode end housekeeping
        if done:
            rewards_history.append(ep_reward)
            obs, _ = env.reset()
            f = preprocess(obs)
            stack_reset(frame_stack, f)
            nstep_q.clear()
            ep_reward = 0.0

        # Learn
        if step > TRAIN_START and len(memory) >= BATCH:
            s, a, r, s2, d = memory.sample(BATCH)
            q = online(s).gather(1, a.unsqueeze(1)).squeeze(1)

            # Double-DQN target
            with torch.no_grad():
                online_best = online(s2).argmax(1)
                tgt_q = target(s2).gather(1, online_best.unsqueeze(1)).squeeze(1)
                y = r + (GAMMA**N_STEPS) * tgt_q * (~d)

            loss = F.smooth_l1_loss(q, y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(online.parameters(), 10.0)
            opt.step()

        # Sync target net
        if step % TARGET_SYNC == 0:
            target.load_state_dict(online.state_dict())

        # Save snapshot & plot
        if step % SAVE_EVERY == 0 and step:
            ckpt = f"checkpoint_{step//1000}k.pt"
            torch.save(online.state_dict(), ckpt)
            print(f"Saved {ckpt}")

    # Plot results
    plt.plot(rewards_history)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("DQN on CarRacing-v2")
    plt.savefig("training_curve.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    main()
