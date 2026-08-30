import time
import numpy as np
import mujoco
import mujoco.viewer

import scan_right_workspace as rs
import scan_right_workspace_poseik as poseik


TEST_X = 0.516
TEST_Y = 0.050

USE_POSE_IK = True

SLOW_SLEEP = 0.01


def step_n_viewer(model, data, viewer, n, label=""):
    if label:
        print(label)

    for _ in range(n):
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(SLOW_SLEEP)


def print_state(model, data, label):
    cube = rs.body_pos(model, data, rs.CUBE_BODY)
    tcp = rs.site_pos(model, data, rs.RIGHT_SITE_NAME)

    inner = rs.body_pos(model, data, "openarm_right_ee_inner_finger")
    outer = rs.body_pos(model, data, "openarm_right_ee_outer_finger")

    print("")
    print("=" * 70)
    print(label)
    print("=" * 70)
    print("cube:", np.array2string(cube, precision=4))
    print("right_tcp:", np.array2string(tcp, precision=4))
    print("inner_finger body:", np.array2string(inner, precision=4))
    print("outer_finger body:", np.array2string(outer, precision=4))
    print("tcp - cube:", np.array2string(tcp - cube, precision=4))

    right_finger_act = rs.get_act_id(model, rs.RIGHT_FINGER_ACT)
    left_finger_act = rs.get_act_id(model, "left_finger1_ctrl")

    print("right finger ctrl:", float(data.ctrl[right_finger_act]))
    print("left finger ctrl:", float(data.ctrl[left_finger_act]))
    print("is_success:", rs.is_success(model, data))


def move_right_tcp(model, data, viewer, target, target_mat=None, steps=100, label=""):
    print("")
    print(label)
    print("target:", np.array2string(target, precision=4))

    if USE_POSE_IK:
        ok, pos_err, ori_err = poseik.move_right_tcp_to_pose(
            model=model,
            data=data,
            target_pos=target,
            target_mat=target_mat,
            steps=steps,
        )
        print("pose IK ok:", ok, "pos_err:", pos_err, "ori_err:", ori_err)
    else:
        ok = rs.move_right_tcp_to(model, data, target, steps=steps)
        print("pos IK ok:", ok)

    for _ in range(30):
        viewer.sync()
        time.sleep(SLOW_SLEEP)

    return ok


def set_gripper_viewer(model, data, viewer, value, steps=80, label=""):
    print("")
    print(label, "value =", value)

    act_id = rs.get_act_id(model, rs.RIGHT_FINGER_ACT)
    start = float(data.ctrl[act_id])

    low, high = model.actuator_ctrlrange[act_id]
    target = float(np.clip(value, low, high))

    for i in range(steps):
        alpha = (i + 1) / steps
        data.ctrl[act_id] = (1.0 - alpha) * start + alpha * target
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(SLOW_SLEEP)


def set_lifter_viewer(model, data, viewer, value, steps=90, label=""):
    print("")
    print(label, "value =", value)

    act_id = rs.get_act_id(model, rs.LIFTER_ACT)
    start = float(data.ctrl[act_id])

    low, high = model.actuator_ctrlrange[act_id]
    target = float(np.clip(value, low, high))

    for i in range(steps):
        alpha = (i + 1) / steps
        data.ctrl[act_id] = (1.0 - alpha) * start + alpha * target
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(SLOW_SLEEP)


def main():
    print("加载模型:", rs.XML_PATH)
    print("测试点:", TEST_X, TEST_Y)
    print("USE_POSE_IK:", USE_POSE_IK)

    model = mujoco.MjModel.from_xml_path(str(rs.XML_PATH))
    data = mujoco.MjData(model)

    rs.load_home(model, data)
    rs.set_cube_pose(model, data, TEST_X, TEST_Y, z=1.05)
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("")
        print("viewer 已打开。")
        print("重点观察：")
        print("1. 右夹爪张开/闭合方向是否正确")
        print("2. grasp 时 TCP 是否在方块中心")
        print("3. 闭合时是夹住方块，还是把方块推走")
        print("4. lift 时方块有没有被带起来")
        print("")

        step_n_viewer(model, data, viewer, 120, "settle cube")
        print_state(model, data, "初始状态")

        # 把右手姿态锁在 home 姿态
        target_mat = poseik.get_home_right_site_mat(model, data)

        # 先试当前假设：负数张开，0 闭合
        set_gripper_viewer(
            model,
            data,
            viewer,
            rs.RIGHT_FINGER_PRE_OPEN,
            steps=100,
            label="右夹爪预张开",
        )
        print_state(model, data, "预张开后")

        cube = rs.body_pos(model, data, rs.CUBE_BODY).copy()
        cube_initial_z = float(cube[2])

        pregrasp = cube + rs.PICK_PREGRASP_OFFSET
        ok = move_right_tcp(
            model,
            data,
            viewer,
            pregrasp,
            target_mat=target_mat,
            steps=120,
            label="移动到 pregrasp",
        )
        print_state(model, data, "pregrasp 后")
        if not ok:
            print("pregrasp IK 失败，停止")
            return

        grasp = cube + rs.PICK_GRASP_OFFSET
        ok = move_right_tcp(
            model,
            data,
            viewer,
            grasp,
            target_mat=target_mat,
            steps=140,
            label="移动到 grasp",
        )
        print_state(model, data, "grasp 后")
        if not ok:
            print("grasp IK 失败，停止")
            return

        set_gripper_viewer(
            model,
            data,
            viewer,
            rs.RIGHT_FINGER_CLOSE,
            steps=120,
            label="右夹爪闭合",
        )
        step_n_viewer(model, data, viewer, 80, "闭合后 hold")
        print_state(model, data, "闭合后")

        set_lifter_viewer(
            model,
            data,
            viewer,
            rs.LIFTER_UP,
            steps=120,
            label="lifter 抬起",
        )
        step_n_viewer(model, data, viewer, 120, "抬起后 hold")
        print_state(model, data, "抬起后")

        cube_after = rs.body_pos(model, data, rs.CUBE_BODY)
        print("")
        print("cube lift delta z:", float(cube_after[2] - cube_initial_z))
        print("is_lifted:", rs.is_lifted(cube_initial_z, model, data))

        print("")
        print("=" * 70)
        print("观察结束")
        print("=" * 70)
        print("如果方块没被夹住，请看下面几种情况：")
        print("A. 夹爪闭合方向反了：闭合时反而张开")
        print("B. TCP 太靠前/靠后：夹爪碰方块边缘")
        print("C. TCP 高度不对：从上方压到方块，没夹住")
        print("D. 手爪姿态不对：两个指头不是夹在方块两侧")
        print("")
        print("关闭 viewer 后脚本结束。")

        while viewer.is_running():
            viewer.sync()
            time.sleep(0.03)


if __name__ == "__main__":
    main()