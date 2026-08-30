from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "bc_dataset_v5_1.npz"
SAVE_PATH = ROOT / "bc_policy_phase_v5_1.pt"

BATCH_SIZE = 512
EPOCHS = 300
LR = 1e-3
HIDDEN_DIM = 256
TRAIN_EP_RATIO = 0.90
SEED = 42


DEFAULT_PHASE_NAMES = [
    "open_gripper",
    "move_pregrasp",
    "move_grasp",
    "close_gripper",
    "hold_grasp",
    "lift_object",
    "move_preplace",
    "move_release",
    "open_release",
    "done",
    "done_hold",
]


class PhaseBCPolicy(nn.Module):
    def __init__(self, input_dim, act_dim=5, hidden_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, act_dim),
        )

    def forward(self, x):
        return self.net(x)


def encode_phases(phases, phase_names):
    phase_to_id = {name: i for i, name in enumerate(phase_names)}
    onehot = np.zeros((len(phases), len(phase_names)), dtype=np.float32)

    for i, p in enumerate(phases):
        p = str(p)
        if p not in phase_to_id:
            raise ValueError(f"未知 phase: {p}")
        onehot[i, phase_to_id[p]] = 1.0

    return onehot


def split_by_episode(episode_lengths, train_ratio=0.9, seed=42):
    rng = np.random.default_rng(seed)

    n_eps = len(episode_lengths)
    ep_indices = np.arange(n_eps)
    rng.shuffle(ep_indices)

    n_train = int(round(n_eps * train_ratio))
    train_eps = set(ep_indices[:n_train].tolist())
    val_eps = set(ep_indices[n_train:].tolist())

    starts = np.zeros(n_eps, dtype=np.int64)
    ends = np.zeros(n_eps, dtype=np.int64)

    cursor = 0
    for i, length in enumerate(episode_lengths):
        starts[i] = cursor
        ends[i] = cursor + int(length)
        cursor = ends[i]

    train_indices = []
    val_indices = []

    for ep in range(n_eps):
        idx = np.arange(starts[ep], ends[ep], dtype=np.int64)
        if ep in train_eps:
            train_indices.append(idx)
        else:
            val_indices.append(idx)

    train_indices = np.concatenate(train_indices, axis=0)
    val_indices = np.concatenate(val_indices, axis=0)

    return train_indices, val_indices


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"找不到数据集：{DATA_PATH}")

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("使用 device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    data = np.load(DATA_PATH, allow_pickle=True)

    obs = data["obs"].astype(np.float32)
    actions = data["actions"].astype(np.float32)
    phases = data["phases"].astype(str)
    episode_lengths = data["episode_lengths"].astype(np.int64)

    if "phase_names" in data:
        phase_names = [str(x) for x in data["phase_names"]]
    else:
        phase_names = DEFAULT_PHASE_NAMES

    if "env_config" in data:
        env_config = data["env_config"].item()
    else:
        env_config = {
            "frame_skip": 30,
            "max_tcp_delta": 0.024,
            "max_finger_delta": 0.040,
            "max_lifter_delta": 0.025,
            "max_episode_steps": 460,
            "cube_x_range": (0.47, 0.58),
            "cube_y_range": (-0.08, 0.08),
        }

    phase_onehot = encode_phases(phases, phase_names)
    obs_aug = np.concatenate([obs, phase_onehot], axis=1).astype(np.float32)

    obs_dim = obs.shape[1]
    phase_dim = phase_onehot.shape[1]
    input_dim = obs_aug.shape[1]
    act_dim = actions.shape[1]

    train_idx, val_idx = split_by_episode(
        episode_lengths,
        train_ratio=TRAIN_EP_RATIO,
        seed=SEED,
    )

    x_train = obs_aug[train_idx]
    y_train = actions[train_idx]
    x_val = obs_aug[val_idx]
    y_val = actions[val_idx]

    obs_aug_mean_np = x_train.mean(axis=0).astype(np.float32)
    obs_aug_std_np = (x_train.std(axis=0) + 1e-6).astype(np.float32)

    action_mean_np = y_train.mean(axis=0).astype(np.float32)
    action_std_np = (y_train.std(axis=0) + 1e-6).astype(np.float32)

    x_train_norm = (x_train - obs_aug_mean_np) / obs_aug_std_np
    x_val_norm = (x_val - obs_aug_mean_np) / obs_aug_std_np

    y_train_norm = (y_train - action_mean_np) / action_std_np
    y_val_norm = (y_val - action_mean_np) / action_std_np

    train_ds = TensorDataset(
        torch.from_numpy(x_train_norm).float(),
        torch.from_numpy(y_train_norm).float(),
    )
    val_x_tensor = torch.from_numpy(x_val_norm).float().to(device)
    val_y_tensor = torch.from_numpy(y_val_norm).float().to(device)
    val_y_raw_tensor = torch.from_numpy(y_val).float().to(device)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )

    model = PhaseBCPolicy(
        input_dim=input_dim,
        act_dim=act_dim,
        hidden_dim=HIDDEN_DIM,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    action_mean_t = torch.from_numpy(action_mean_np).float().to(device)
    action_std_t = torch.from_numpy(action_std_np).float().to(device)

    best_val_loss = float("inf")
    best_epoch = -1
    best_val_mae = None

    print("")
    print("=" * 60)
    print("开始训练 Phase-aware BC v5")
    print("=" * 60)
    print("DATA_PATH:", DATA_PATH)
    print("SAVE_PATH:", SAVE_PATH)
    print("obs shape:", obs.shape)
    print("actions shape:", actions.shape)
    print("phases shape:", phases.shape)
    print("episode_lengths shape:", episode_lengths.shape)
    print("phase_names:", phase_names)
    print("obs_dim:", obs_dim)
    print("phase_dim:", phase_dim)
    print("input_dim:", input_dim)
    print("act_dim:", act_dim)
    print("train episodes:", int(round(len(episode_lengths) * TRAIN_EP_RATIO)))
    print("val episodes:", len(episode_lengths) - int(round(len(episode_lengths) * TRAIN_EP_RATIO)))
    print("train samples:", len(train_idx))
    print("val samples:", len(val_idx))
    print("env_config:", env_config)
    print("")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)
            loss = loss_fn(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            val_pred_norm = model(val_x_tensor)
            val_loss = float(loss_fn(val_pred_norm, val_y_tensor).item())

            val_pred_raw = val_pred_norm * action_std_t + action_mean_t
            val_mae = float(torch.mean(torch.abs(val_pred_raw - val_y_raw_tensor)).item())

        train_loss = float(np.mean(train_losses))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_val_mae = val_mae

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "input_dim": input_dim,
                "obs_dim": obs_dim,
                "phase_dim": phase_dim,
                "act_dim": act_dim,
                "hidden_dim": HIDDEN_DIM,
                "phase_names": phase_names,
                "obs_aug_mean": torch.from_numpy(obs_aug_mean_np).float(),
                "obs_aug_std": torch.from_numpy(obs_aug_std_np).float(),
                "action_mean": torch.from_numpy(action_mean_np).float(),
                "action_std": torch.from_numpy(action_std_np).float(),
                "env_config": env_config,
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "val_mae": best_val_mae,
                "data_path": str(DATA_PATH),
            }

            torch.save(checkpoint, SAVE_PATH)

        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
            print(
                f"epoch {epoch:03d} | "
                f"train_loss {train_loss:.6f} | "
                f"val_loss {val_loss:.6f} | "
                f"val_mae {val_mae:.6f} | "
                f"best_epoch {best_epoch:03d} | "
                f"best_val_loss {best_val_loss:.6f}"
            )

    print("")
    print("=" * 60)
    print("Phase-aware BC v5 训练完成")
    print("=" * 60)
    print("best_epoch:", best_epoch)
    print("best_val_loss:", best_val_loss)
    print("best_val_mae:", best_val_mae)
    print("模型已保存:", SAVE_PATH)


if __name__ == "__main__":
    main()