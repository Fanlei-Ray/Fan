from pathlib import Path
import shutil
from datetime import datetime


ROOT = Path(__file__).resolve().parents[1]

ARCHIVE = ROOT / "_archive_experiments" / datetime.now().strftime("%Y%m%d_%H%M%S")

MOVE_PATHS = [
    # 右臂探索
    "scripts/scan_right_workspace.py",
    "scripts/scan_right_workspace_poseik.py",
    "scripts/debug_right_pick_viewer.py",
    "scripts/calibrate_right_pick.py",
    "scripts/calibrate_right_autoyaw.py",

    # 左臂姿态/策略探索
    "scripts/scan_left_workspace.py",
    "scripts/calibrate_left_autoyaw.py",
    "scripts/calibrate_left_position_strategy.py",

    # v5 / v5.1 实验脚本
    "scripts/collect_bc_data_v5.py",
    "scripts/train_bc_phase_v5.py",
    "scripts/test_bc_phase_v5.py",

    # PPO residual 实验脚本，暂时不是主线
    "scripts/openarm_phase_rl_env.py",
    "scripts/train_ppo_phase.py",
    "scripts/test_ppo_phase.py",

    # 坏数据/坏模型
    "bc_dataset_v5.npz",
    "bc_dataset_v5_1.npz",
    "bc_dataset_v5_bad_633.npz",
    "bc_dataset_v5_1_bad_700.npz",
    "bc_policy_phase_v5.pt",
    "bc_policy_phase_v5_1.pt",
    "bc_policy_phase_v5_bad_633.pt",
    "bc_policy_phase_v5_1_bad_700.pt",
]


KEEP_HINTS = [
    "scripts/right_pick_place.py",
    "scripts/openarm_pickplace_env.py",
    "scripts/collect_bc_data.py",
    "scripts/train_bc_phase.py",
    "scripts/test_bc_phase_policy.py",
    "bc_dataset_v3.npz",
    "bc_policy_phase_v4.pt",
    "bc_policy_phase_v4_done_967.pt",
    "bc_policy_phase_v4_wide_933.pt",
    "bc_policy_best_current.pt",
]


def move_one(rel_path):
    src = ROOT / rel_path

    if not src.exists():
        print(f"[skip] 不存在：{rel_path}")
        return

    dst = ARCHIVE / rel_path
    dst.parent.mkdir(parents=True, exist_ok=True)

    shutil.move(str(src), str(dst))
    print(f"[move] {rel_path} -> {dst.relative_to(ROOT)}")


def main():
    print("=" * 80)
    print("OpenArm 实验文件归档")
    print("=" * 80)
    print("项目根目录:", ROOT)
    print("归档目录:", ARCHIVE)
    print("")
    print("这会移动探索脚本和失败模型，不会删除文件。")
    print("保留主线 v4 / 当前最佳模型。")
    print("")

    ARCHIVE.mkdir(parents=True, exist_ok=True)

    for rel_path in MOVE_PATHS:
        move_one(rel_path)

    print("")
    print("=" * 80)
    print("归档完成")
    print("=" * 80)
    print("")
    print("建议保留的主线文件：")
    for item in KEEP_HINTS:
        p = ROOT / item
        mark = "OK" if p.exists() else "missing"
        print(f"{mark:8s} {item}")

    print("")
    print("当前结论：")
    print("1. 主力模型继续使用 bc_policy_phase_v4_wide_933.pt / bc_policy_best_current.pt")
    print("2. v5/v5.1 暂不使用")
    print("3. PPO residual 暂不作为主策略")
    print("4. 右臂训练暂停，等右臂 expert 调通后再说")


if __name__ == "__main__":
    main()