from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# 路径和训练参数
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "bc_dataset_v3.npz"
SAVE_PATH = ROOT / "bc_policy_v3.pt"

SEED = 42
BATCH_SIZE = 256
EPOCHS = 250
LR = 1e-3
WEIGHT_DECAY = 1e-5
VAL_RATIO = 0.10

HIDDEN_DIM = 256


# ============================================================
# 工具函数
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_scalar(data, key, default_value):
    if key not in data:
        return default_value
    arr = np.asarray(data[key])
    return arr.reshape(-1)[0].item()


def build_episode_indices(episode_lengths):
    indices = []
    start = 0

    for length in episode_lengths:
        end = start + int(length)
        indices.append(np.arange(start, end))
        start = end

    return indices


# ============================================================
# BC 网络
# ============================================================

class BCPolicy(nn.Module):
    def __init__(self, obs_dim=24, act_dim=5, hidden_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
        )

    def forward(self, obs):
        return self.net(obs)


# ============================================================
# 数据加载
# ============================================================

def load_bc_dataset(path):
    data = np.load(path, allow_pickle=True)

    obs = data["obs"].astype(np.float32)
    actions = data["actions"].astype(np.float32)
    episode_lengths = data["episode_lengths"].astype(np.int32)

    print("加载数据集：", path)
    print("obs shape:", obs.shape)
    print("actions shape:", actions.shape)
    print("episode_lengths shape:", episode_lengths.shape)
    print("demo 数:", len(episode_lengths))
    print("总样本数:", len(obs))

    print("\naction min:", actions.min(axis=0))
    print("action max:", actions.max(axis=0))
    print("action mean:", actions.mean(axis=0))

    env_config = {
        "max_tcp_delta": float(get_scalar(data, "max_tcp_delta", 0.024)),
        "max_finger_delta": float(get_scalar(data, "max_finger_delta", 0.040)),
        "max_lifter_delta": float(get_scalar(data, "max_lifter_delta", 0.025)),
        "frame_skip": int(get_scalar(data, "frame_skip", 30)),
    }

    print("\n数据采集时 env_config:")
    for k, v in env_config.items():
        print(f"{k}: {v}")

    return obs, actions, episode_lengths, env_config


def split_by_episode(obs, actions, episode_lengths, val_ratio=0.1):
    episode_indices = build_episode_indices(episode_lengths)

    num_episodes = len(episode_indices)
    episode_ids = np.arange(num_episodes)
    np.random.shuffle(episode_ids)

    num_val = max(1, int(num_episodes * val_ratio))

    val_episode_ids = episode_ids[:num_val]
    train_episode_ids = episode_ids[num_val:]

    train_indices = np.concatenate([episode_indices[i] for i in train_episode_ids])
    val_indices = np.concatenate([episode_indices[i] for i in val_episode_ids])

    train_obs = obs[train_indices]
    train_actions = actions[train_indices]

    val_obs = obs[val_indices]
    val_actions = actions[val_indices]

    print("\n划分数据：")
    print("train episodes:", len(train_episode_ids))
    print("val episodes:", len(val_episode_ids))
    print("train samples:", len(train_obs))
    print("val samples:", len(val_obs))

    return train_obs, train_actions, val_obs, val_actions


# ============================================================
# 训练
# ============================================================

def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用 device:", device)

    obs, actions, episode_lengths, env_config = load_bc_dataset(DATA_PATH)

    train_obs, train_actions, val_obs, val_actions = split_by_episode(
        obs,
        actions,
        episode_lengths,
        val_ratio=VAL_RATIO,
    )

    obs_mean = train_obs.mean(axis=0)
    obs_std = train_obs.std(axis=0) + 1e-6

    action_mean = train_actions.mean(axis=0)
    action_std = train_actions.std(axis=0) + 1e-6

    train_obs_norm = (train_obs - obs_mean) / obs_std
    val_obs_norm = (val_obs - obs_mean) / obs_std

    train_actions_norm = (train_actions - action_mean) / action_std
    val_actions_norm = (val_actions - action_mean) / action_std

    train_dataset = TensorDataset(
        torch.from_numpy(train_obs_norm).float(),
        torch.from_numpy(train_actions_norm).float(),
    )

    val_obs_tensor = torch.from_numpy(val_obs_norm).float().to(device)
    val_actions_tensor = torch.from_numpy(val_actions_norm).float().to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )

    model = BCPolicy(
        obs_dim=obs.shape[1],
        act_dim=actions.shape[1],
        hidden_dim=HIDDEN_DIM,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_epoch = -1

    print("\n开始训练 BC policy")
    print(f"EPOCHS={EPOCHS}, BATCH_SIZE={BATCH_SIZE}, LR={LR}")

    for epoch in range(1, EPOCHS + 1):
        model.train()

        train_losses = []

        for batch_obs, batch_actions in train_loader:
            batch_obs = batch_obs.to(device)
            batch_actions = batch_actions.to(device)

            pred_actions = model(batch_obs)
            loss = loss_fn(pred_actions, batch_actions)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))

        model.eval()
        with torch.no_grad():
            val_pred = model(val_obs_tensor)
            val_loss = loss_fn(val_pred, val_actions_tensor).item()

            # 还原到真实 action 空间，额外看 MAE
            val_pred_real = (
                val_pred.cpu().numpy() * action_std + action_mean
            )
            val_action_real = val_actions

            val_mae = np.mean(np.abs(val_pred_real - val_action_real), axis=0)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "obs_dim": int(obs.shape[1]),
                "act_dim": int(actions.shape[1]),
                "hidden_dim": int(HIDDEN_DIM),

                "obs_mean": torch.from_numpy(obs_mean).float(),
                "obs_std": torch.from_numpy(obs_std).float(),
                "action_mean": torch.from_numpy(action_mean).float(),
                "action_std": torch.from_numpy(action_std).float(),

                "env_config": env_config,

                "data_path": str(DATA_PATH),
                "epoch": int(epoch),
                "best_val_loss": float(best_val_loss),
                "val_mae": torch.from_numpy(val_mae).float(),
            }

            torch.save(checkpoint, SAVE_PATH)

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"epoch={epoch:03d} | "
                f"train_loss={train_loss:.6f} | "
                f"val_loss={val_loss:.6f} | "
                f"best_val_loss={best_val_loss:.6f} @ epoch {best_epoch}"
            )
            print(
                "    val_action_MAE:",
                np.array2string(val_mae, precision=4, suppress_small=True),
            )

    print("\n训练完成")
    print("best_epoch:", best_epoch)
    print("best_val_loss:", best_val_loss)
    print("模型已保存：", SAVE_PATH)


if __name__ == "__main__":
    main()