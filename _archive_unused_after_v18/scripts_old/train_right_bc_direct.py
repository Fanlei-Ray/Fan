from pathlib import Path
import json
import time
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


# ============================================================
# Train right-arm DIRECT actuator BC policy v1
#
# 输入:
#   outputs/right_bc_v1/right_bc_direct_dataset_v1.npz
#
# 输出:
#   outputs/right_bc_v1/right_bc_direct_policy_v1.pt
#   outputs/right_bc_v1/right_bc_direct_train_log_v1.json
#
# obs:
#   24 dim
#
# action:
#   9 dim direct actuator ctrl:
#     right_joint1_ctrl
#     right_joint2_ctrl
#     right_joint3_ctrl
#     right_joint4_ctrl
#     right_joint5_ctrl
#     right_joint6_ctrl
#     right_joint7_ctrl
#     right_finger1_ctrl
#     lifter_ctrl
# ============================================================


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "outputs" / "right_bc_v1"

DATASET_PATH = RESULT_DIR / "right_bc_direct_dataset_v1.npz"
MODEL_PATH = RESULT_DIR / "right_bc_direct_policy_v1.pt"
LOG_PATH = RESULT_DIR / "right_bc_direct_train_log_v1.json"


SEED = 202602
BATCH_SIZE = 512
EPOCHS = 450
LR = 2e-4
WEIGHT_DECAY = 1e-5
VAL_RATIO = 0.12
PATIENCE = 80

OBS_DIM = 24
ACTION_DIM = 9

# action 维度:
# 0~6: right arm joints
# 7: right_finger1_ctrl
# 8: lifter_ctrl
#
# 夹爪和 lifter 对成功非常关键，所以权重稍微加大。
ACTION_WEIGHTS = torch.tensor(
    [
        1.0,  # j1
        1.0,  # j2
        1.0,  # j3
        1.0,  # j4
        1.0,  # j5
        1.0,  # j6
        1.2,  # j7
        1.5,  # finger
        1.5,  # lifter
    ],
    dtype=torch.float32,
)


class DirectBCDataset(Dataset):
    def __init__(self, obs, actions):
        self.obs = torch.from_numpy(obs.astype(np.float32))
        self.actions = torch.from_numpy(actions.astype(np.float32))

    def __len__(self):
        return self.obs.shape[0]

    def __getitem__(self, idx):
        return self.obs[idx], self.actions[idx]


class RightDirectBCPolicy(nn.Module):
    def __init__(self, obs_dim=24, action_dim=9):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),

            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.SiLU(),

            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.SiLU(),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.SiLU(),

            nn.Linear(128, action_dim),
        )

    def forward(self, obs):
        return self.net(obs)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"找不到数据集：{DATASET_PATH}")

    data = np.load(DATASET_PATH, allow_pickle=True)

    obs = data["obs"].astype(np.float32)
    actions = data["actions"].astype(np.float32)
    episode_lengths = data["episode_lengths"].astype(np.int32)

    if "action_actuator_names" in data:
        action_actuator_names = data["action_actuator_names"]
    else:
        action_actuator_names = np.array(
            [
                "right_joint1_ctrl",
                "right_joint2_ctrl",
                "right_joint3_ctrl",
                "right_joint4_ctrl",
                "right_joint5_ctrl",
                "right_joint6_ctrl",
                "right_joint7_ctrl",
                "right_finger1_ctrl",
                "lifter_ctrl",
            ]
        )

    assert obs.ndim == 2 and obs.shape[1] == OBS_DIM, obs.shape
    assert actions.ndim == 2 and actions.shape[1] == ACTION_DIM, actions.shape
    assert obs.shape[0] == actions.shape[0]

    print("=" * 80)
    print("Load right DIRECT BC dataset")
    print("=" * 80)
    print("DATASET_PATH:", DATASET_PATH)
    print("obs shape:", obs.shape)
    print("actions shape:", actions.shape)
    print("episode_lengths shape:", episode_lengths.shape)
    print("episodes:", len(episode_lengths))
    print("samples:", len(obs))
    print("action_actuator_names:", action_actuator_names)
    print("obs min:", obs.min(axis=0))
    print("obs max:", obs.max(axis=0))
    print("action min:", actions.min(axis=0))
    print("action max:", actions.max(axis=0))
    print("action mean:", actions.mean(axis=0))
    print("action std:", actions.std(axis=0))
    print("=" * 80)

    return obs, actions, episode_lengths, action_actuator_names


def normalize_dataset(obs, actions):
    obs_mean = obs.mean(axis=0).astype(np.float32)
    obs_std = obs.std(axis=0).astype(np.float32)
    obs_std = np.maximum(obs_std, 1e-6).astype(np.float32)

    action_mean = actions.mean(axis=0).astype(np.float32)
    action_std = actions.std(axis=0).astype(np.float32)
    action_std = np.maximum(action_std, 1e-6).astype(np.float32)

    obs_norm = ((obs - obs_mean) / obs_std).astype(np.float32)
    actions_norm = ((actions - action_mean) / action_std).astype(np.float32)

    return obs_norm, actions_norm, obs_mean, obs_std, action_mean, action_std


def weighted_mse(pred, target, weights):
    loss = (pred - target) ** 2
    loss = loss * weights.view(1, -1)
    return loss.mean()


def eval_model(model, loader, device, action_mean, action_std):
    model.eval()

    total_loss = 0.0
    total_count = 0

    abs_err_sum = torch.zeros(ACTION_DIM, device=device)
    mse_sum = torch.zeros(ACTION_DIM, device=device)

    weights = ACTION_WEIGHTS.to(device)

    with torch.no_grad():
        for obs, actions_norm in loader:
            obs = obs.to(device)
            actions_norm = actions_norm.to(device)

            pred_norm = model(obs)
            loss = weighted_mse(pred_norm, actions_norm, weights)

            batch = obs.shape[0]
            total_loss += float(loss.item()) * batch
            total_count += batch

            pred = pred_norm * action_std + action_mean
            target = actions_norm * action_std + action_mean

            err = pred - target
            abs_err_sum += err.abs().sum(dim=0)
            mse_sum += (err ** 2).sum(dim=0)

    avg_loss = total_loss / max(1, total_count)
    mae = abs_err_sum / max(1, total_count)
    rmse = torch.sqrt(mse_sum / max(1, total_count))

    return avg_loss, mae.detach().cpu().numpy(), rmse.detach().cpu().numpy()


def main():
    seed_everything(SEED)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("Train right DIRECT BC policy v1")
    print("=" * 80)
    print("ROOT:", ROOT)
    print("RESULT_DIR:", RESULT_DIR)
    print("DATASET_PATH:", DATASET_PATH)
    print("MODEL_PATH:", MODEL_PATH)
    print("LOG_PATH:", LOG_PATH)
    print("device:", device)
    if torch.cuda.is_available():
        print("cuda device:", torch.cuda.get_device_name(0))
    print("=" * 80)

    obs, actions, episode_lengths, action_actuator_names = load_dataset()

    action_min = actions.min(axis=0).astype(np.float32)
    action_max = actions.max(axis=0).astype(np.float32)

    obs_norm, actions_norm, obs_mean, obs_std, action_mean_np, action_std_np = normalize_dataset(
        obs=obs,
        actions=actions,
    )

    dataset = DirectBCDataset(obs_norm, actions_norm)

    val_size = max(1, int(len(dataset) * VAL_RATIO))
    train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(SEED)

    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = RightDirectBCPolicy(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=LR * 0.08,
    )

    action_mean = torch.from_numpy(action_mean_np).to(device)
    action_std = torch.from_numpy(action_std_np).to(device)
    weights = ACTION_WEIGHTS.to(device)

    best_val_loss = float("inf")
    best_epoch = -1
    bad_epochs = 0

    history = []

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()

        train_loss_sum = 0.0
        train_count = 0

        for batch_obs, batch_actions_norm in train_loader:
            batch_obs = batch_obs.to(device)
            batch_actions_norm = batch_actions_norm.to(device)

            pred_norm = model(batch_obs)
            loss = weighted_mse(pred_norm, batch_actions_norm, weights)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            batch = batch_obs.shape[0]
            train_loss_sum += float(loss.item()) * batch
            train_count += batch

        scheduler.step()

        train_loss = train_loss_sum / max(1, train_count)

        val_loss, val_mae, val_rmse = eval_model(
            model=model,
            loader=val_loader,
            device=device,
            action_mean=action_mean,
            action_std=action_std,
        )

        lr_now = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_mae": val_mae.tolist(),
            "val_rmse": val_rmse.tolist(),
            "lr": float(lr_now),
        }

        history.append(row)

        improved = val_loss < best_val_loss

        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            bad_epochs = 0

            save_obj = {
                "model_state_dict": model.state_dict(),
                "obs_dim": OBS_DIM,
                "action_dim": ACTION_DIM,
                "obs_mean": obs_mean,
                "obs_std": obs_std,
                "action_mean": action_mean_np,
                "action_std": action_std_np,
                "action_min": action_min,
                "action_max": action_max,
                "dataset_path": str(DATASET_PATH),
                "episode_count": int(len(episode_lengths)),
                "sample_count": int(len(obs)),
                "best_epoch": int(best_epoch),
                "best_val_loss": float(best_val_loss),
                "val_mae": val_mae,
                "val_rmse": val_rmse,
                "seed": int(SEED),
                "action_actuator_names": action_actuator_names,
                "model_class": "RightDirectBCPolicy",
                "notes": "Right arm direct-actuator BC policy v1. Obs dim 24, action dim 9.",
            }

            torch.save(save_obj, MODEL_PATH)
        else:
            bad_epochs += 1

        if epoch == 1 or epoch % 10 == 0 or improved:
            print(
                f"epoch={epoch:04d} "
                f"train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f} "
                f"best={best_val_loss:.6f}@{best_epoch} "
                f"val_mae={np.array2string(val_mae, precision=5)} "
                f"lr={lr_now:.2e}"
            )

        with open(LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dataset_path": str(DATASET_PATH),
                    "model_path": str(MODEL_PATH),
                    "device": str(device),
                    "epochs_requested": EPOCHS,
                    "best_epoch": int(best_epoch),
                    "best_val_loss": float(best_val_loss),
                    "history": history,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        if bad_epochs >= PATIENCE:
            print("")
            print(f"Early stop: {bad_epochs} epochs without improvement.")
            break

    elapsed = time.time() - start_time

    print("")
    print("=" * 80)
    print("Right DIRECT BC training 总结")
    print("=" * 80)
    print("best_epoch:", best_epoch)
    print("best_val_loss:", best_val_loss)
    print("MODEL_PATH:", MODEL_PATH)
    print("LOG_PATH:", LOG_PATH)
    print("耗时:", f"{elapsed / 60:.1f} min")

    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)

    print("saved episode_count:", ckpt["episode_count"])
    print("saved sample_count:", ckpt["sample_count"])
    print("saved action_actuator_names:", ckpt["action_actuator_names"])
    print("saved val_mae:", ckpt["val_mae"])
    print("saved val_rmse:", ckpt["val_rmse"])
    print("saved action_min:", ckpt["action_min"])
    print("saved action_max:", ckpt["action_max"])


if __name__ == "__main__":
    main()