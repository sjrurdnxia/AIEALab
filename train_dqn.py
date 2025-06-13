# dqn_carracing_train.py
import random, math, cv2, gymnasium as gym
from collections import deque, namedtuple
from typing import Deque, Tuple

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────── hyper-params ───────────────────────────────────────────────
EPISODES       = 500
MEM_CAPACITY   = 100_000
BATCH_SIZE     = 64
GAMMA          = 0.99
LR             = 2.5e-4
TARGET_SYNC    = 2_000            # steps
EPS_START      = 1.0
EPS_END        = 0.05
EPS_DECAY      = 300_000          # steps
STACK          = 4
FRAME_SKIP     = 4                # env step repeat
GRAD_CLIP      = 10.0

# ──────────────── helpers ───────────────────────────────────────────────────
def preprocess(frame):
    gray   = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized= cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return resized

def epsilon(step):
    return EPS_END + (EPS_START-EPS_END) * math.exp(-step / EPS_DECAY)

Transition = namedtuple("T", "s a r s2 d")

class ReplayBuffer:
    def __init__(self, cap): self.buf: Deque[Transition] = deque(maxlen=cap)
    def push(self,*args):     self.buf.append(Transition(*args))
    def sample(self,k):
        batch = random.sample(self.buf,k)
        s  = torch.as_tensor(np.stack([b.s  for b in batch]), device=DEVICE)
        a  = torch.as_tensor([b.a for b in batch], device=DEVICE)
        r  = torch.as_tensor([b.r for b in batch], device=DEVICE)
        s2 = torch.as_tensor(np.stack([b.s2 for b in batch]),device=DEVICE)
        d  = torch.as_tensor([b.d for b in batch], device=DEVICE)
        return s,a,r,s2,d
    def __len__(self): return len(self.buf)

# ─────────────── Dueling CNN ────────────────────────────────────────────────
class DQN(nn.Module):
    def __init__(self,n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(STACK,32,8,4), nn.ReLU(),
            nn.Conv2d(32,64,4,2),    nn.ReLU(),
            nn.Conv2d(64,64,3,1),    nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1,STACK,84,84)
            conv_out = self.conv(dummy).flatten(1).size(1)
        self.val = nn.Sequential(nn.Linear(conv_out,512), nn.ReLU(),
                                 nn.Linear(512,1))
        self.adv = nn.Sequential(nn.Linear(conv_out,512), nn.ReLU(),
                                 nn.Linear(512,n_actions))
        for m in self.modules():
            if isinstance(m,(nn.Conv2d,nn.Linear)):
                nn.init.kaiming_normal_(m.weight,nonlinearity="relu")
                nn.init.zeros_(m.bias)
    def forward(self,x):
        x = x.float()/255.0
        feat = self.conv(x).flatten(1)
        v = self.val(feat)
        a = self.adv(feat)
        return v + a-a.mean(1,keepdim=True)

    def act(self,state,eps):
        if random.random()<eps: return random.randrange(self.adv[-1].out_features)
        with torch.no_grad():
            t = torch.as_tensor(state,device=DEVICE).unsqueeze(0)
            return int(self(t).argmax(1).item())

# ───────────── Frame-stack & skip wrapper ───────────────────────────────────
class SkipFrame(gym.Wrapper):
    def __init__(self,env,k):
        super().__init__(env); self.k = k
    def step(self,action):
        total_r, done = 0.0, False
        for _ in range(self.k):
            obs,r,term,trunc,info = self.env.step(action)
            total_r += r; done = term or trunc
            if done: break
        return obs,total_r,term,trunc,info

# ───────────── main training loop ───────────────────────────────────────────
def train():
    env = SkipFrame(gym.make("CarRacing-v3",continuous=False), FRAME_SKIP)
    n_actions = env.action_space.n
    online, target = DQN(n_actions).to(DEVICE), DQN(n_actions).to(DEVICE)
    target.load_state_dict(online.state_dict()); target.eval()
    opt = torch.optim.Adam(online.parameters(), lr=LR, eps=1e-4)
    memory = ReplayBuffer(MEM_CAPACITY)

    rewards, global_step = [], 0
    for ep in range(EPISODES):
        obs,_ = env.reset(); f = preprocess(obs)
        frames = deque([f]*STACK, maxlen=STACK)
        state = np.stack(frames,0); done = False; ep_r = 0
        while not done:
            eps  = epsilon(global_step)
            act  = online.act(state,eps)
            nxt_obs, r, term, trunc, _ = env.step(act)
            done = term or trunc
            r_clipped = np.clip(r,-1,1)
            f2 = preprocess(nxt_obs); frames.append(f2)
            nxt_state = np.stack(frames,0)
            memory.push(state,act,r_clipped,nxt_state,done)
            state = nxt_state; ep_r += r; global_step += 1

            # learn
            if len(memory) >= BATCH_SIZE:
                S,A,R,S2,D = memory.sample(BATCH_SIZE)
                q = online(S).gather(1,A.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    online_best = online(S2).argmax(1)
                    tgt_q = target(S2).gather(1,online_best.unsqueeze(1)).squeeze(1)
                    y = R + GAMMA*tgt_q*(~D)
                loss = F.smooth_l1_loss(q,y)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(),GRAD_CLIP)
                opt.step()
            if global_step % TARGET_SYNC == 0:
                target.load_state_dict(online.state_dict())

        rewards.append(ep_r)
        print(f"Ep {ep:3d} | reward {ep_r:6.1f} | ε {eps:.3f}")

    env.close(); torch.save(online.state_dict(),"dqn_carracing.pt")
    plt.plot(rewards); plt.xlabel("Episode"); plt.ylabel("Reward")
    plt.title("Dueling-Double DQN on CarRacing-v3"); plt.tight_layout()
    plt.savefig("curve.png",dpi=150); plt.show()

if __name__ == "__main__":
    train()
