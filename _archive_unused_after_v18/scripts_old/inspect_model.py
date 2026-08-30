from pathlib import Path
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
XML_PATH = ROOT / "v2" / "demo.xml"
OUTPUT_PATH = ROOT / "scripts" / "model_inspect_output.txt"


KEYWORDS = [
    "left",
    "finger",
    "gripper",
    "hand",
    "wrist",
    "tool",
    "tcp",
    "end",
]


def name_contains_keyword(name):
    if name is None:
        return False

    lower = name.lower()
    return any(k in lower for k in KEYWORDS)


def log(lines, text=""):
    print(text)
    lines.append(str(text))


def get_name(model, obj_type, idx):
    name = mujoco.mj_id2name(model, obj_type, idx)
    return name if name is not None else ""


def load_home(model, data):
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")

    if key_id != -1:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        print("已加载 home keyframe。")
    else:
        print("没有找到 home keyframe，使用默认姿态。")

    for aid in range(model.nu):
        joint_id = model.actuator_trnid[aid, 0]
        if joint_id < 0:
            continue

        qpos_addr = model.jnt_qposadr[joint_id]
        low, high = model.actuator_ctrlrange[aid]
        data.ctrl[aid] = np.clip(data.qpos[qpos_addr], low, high)

    mujoco.mj_forward(model, data)


def print_actuators(model, lines):
    log(lines, "\n========== LEFT ACTUATORS ==========")

    for aid in range(model.nu):
        name = get_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)

        if "left" not in name.lower() and "lifter" not in name.lower():
            continue

        low, high = model.actuator_ctrlrange[aid]
        joint_id = model.actuator_trnid[aid, 0]
        joint_name = get_name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)

        log(
            lines,
            f"{aid:02d}  actuator={name:30s}  joint={joint_name:30s}  "
            f"ctrlrange=[{low:.4f}, {high:.4f}]"
        )


def print_joints(model, lines):
    log(lines, "\n========== LEFT JOINTS ==========")

    for jid in range(model.njnt):
        name = get_name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)

        if "left" not in name.lower() and "lifter" not in name.lower():
            continue

        jtype = model.jnt_type[jid]
        axis = model.jnt_axis[jid]
        qpos_addr = model.jnt_qposadr[jid]

        # range 只有 limited joint 更有意义
        jrange = model.jnt_range[jid]

        log(
            lines,
            f"{jid:02d}  joint={name:35s}  "
            f"type={int(jtype)}  qpos_addr={qpos_addr:02d}  "
            f"axis={axis}  range=[{jrange[0]:.4f}, {jrange[1]:.4f}]"
        )


def print_sites(model, data, lines):
    log(lines, "\n========== CANDIDATE SITES ==========")

    found = False

    for sid in range(model.nsite):
        name = get_name(model, mujoco.mjtObj.mjOBJ_SITE, sid)

        if not name_contains_keyword(name):
            continue

        found = True
        pos = data.site_xpos[sid]
        log(lines, f"{sid:02d}  site={name:35s}  world_pos={pos}")

    if not found:
        log(lines, "没有找到名字里带 left/finger/gripper/hand/wrist/tool/tcp/end 的 site。")


def print_bodies(model, data, lines):
    log(lines, "\n========== CANDIDATE BODIES ==========")

    for bid in range(model.nbody):
        name = get_name(model, mujoco.mjtObj.mjOBJ_BODY, bid)

        if not name_contains_keyword(name):
            continue

        parent_id = model.body_parentid[bid]
        parent_name = get_name(model, mujoco.mjtObj.mjOBJ_BODY, parent_id)

        pos = data.xpos[bid]
        log(
            lines,
            f"{bid:03d}  body={name:40s}  "
            f"parent={parent_name:35s}  world_pos={pos}"
        )


def print_left_body_tree(model, lines):
    log(lines, "\n========== LEFT BODY TREE ==========")

    for bid in range(model.nbody):
        name = get_name(model, mujoco.mjtObj.mjOBJ_BODY, bid)

        if "left" not in name.lower():
            continue

        parent_id = model.body_parentid[bid]
        parent_name = get_name(model, mujoco.mjtObj.mjOBJ_BODY, parent_id)

        log(lines, f"{bid:03d}  {name:40s}  parent -> {parent_name}")


def print_cube_and_frame(model, data, lines):
    log(lines, "\n========== OBJECT POSITIONS ==========")

    for body_name in ["orange_cube", "black_frame"]:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)

        if bid == -1:
            log(lines, f"{body_name}: 找不到")
            continue

        pos = data.xpos[bid]
        log(lines, f"{body_name:20s} world_pos={pos}")


def main():
    print(f"加载模型：{XML_PATH}")

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    load_home(model, data)

    # 稍微 forward 一下，确保 xpos/site_xpos 都更新
    mujoco.mj_forward(model, data)

    lines = []

    log(lines, f"XML_PATH = {XML_PATH}")
    log(lines, f"nbody={model.nbody}, njnt={model.njnt}, nsite={model.nsite}, nu={model.nu}")

    print_actuators(model, lines)
    print_joints(model, lines)
    print_sites(model, data, lines)
    print_bodies(model, data, lines)
    print_left_body_tree(model, lines)
    print_cube_and_frame(model, data, lines)

    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print("\n检查完成。输出文件已保存到：")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()