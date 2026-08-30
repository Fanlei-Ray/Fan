import numpy as np

import scan_right_workspace as rs


TEST_POINTS = [
    (0.516, 0.050),
    (0.516, 0.075),
    (0.516, 0.000),
    (0.534, 0.050),
    (0.497, 0.075),
]


CONFIGS = [
    {
        "name": "A_current_negative_open",
        "open": -0.49,
        "pre_open": -0.445,
        "close": 0.0,
        "grasp_offset": [-0.010, 0.000, -0.005],
        "pregrasp_offset": [-0.005, 0.000, 0.100],
    },
    {
        "name": "B_reverse_finger_direction",
        "open": 0.0,
        "pre_open": 0.0,
        "close": -0.49,
        "grasp_offset": [-0.010, 0.000, -0.005],
        "pregrasp_offset": [-0.005, 0.000, 0.100],
    },
    {
        "name": "C_current_grasp_higher",
        "open": -0.49,
        "pre_open": -0.445,
        "close": 0.0,
        "grasp_offset": [-0.010, 0.000, 0.010],
        "pregrasp_offset": [-0.005, 0.000, 0.110],
    },
    {
        "name": "D_current_grasp_x_plus",
        "open": -0.49,
        "pre_open": -0.445,
        "close": 0.0,
        "grasp_offset": [0.010, 0.000, -0.005],
        "pregrasp_offset": [0.005, 0.000, 0.100],
    },
    {
        "name": "E_current_y_plus",
        "open": -0.49,
        "pre_open": -0.445,
        "close": 0.0,
        "grasp_offset": [-0.010, 0.012, -0.005],
        "pregrasp_offset": [-0.005, 0.012, 0.100],
    },
    {
        "name": "F_current_y_minus",
        "open": -0.49,
        "pre_open": -0.445,
        "close": 0.0,
        "grasp_offset": [-0.010, -0.012, -0.005],
        "pregrasp_offset": [-0.005, -0.012, 0.100],
    },
    {
        "name": "G_reverse_grasp_higher",
        "open": 0.0,
        "pre_open": 0.0,
        "close": -0.49,
        "grasp_offset": [-0.010, 0.000, 0.010],
        "pregrasp_offset": [-0.005, 0.000, 0.110],
    },
    {
        "name": "H_reverse_x_plus",
        "open": 0.0,
        "pre_open": 0.0,
        "close": -0.49,
        "grasp_offset": [0.010, 0.000, -0.005],
        "pregrasp_offset": [0.005, 0.000, 0.100],
    },
]


def apply_config(cfg):
    rs.RIGHT_FINGER_OPEN = float(cfg["open"])
    rs.RIGHT_FINGER_PRE_OPEN = float(cfg["pre_open"])
    rs.RIGHT_FINGER_CLOSE = float(cfg["close"])

    rs.PICK_GRASP_OFFSET = np.array(cfg["grasp_offset"], dtype=np.float64)
    rs.PICK_PREGRASP_OFFSET = np.array(cfg["pregrasp_offset"], dtype=np.float64)


def main():
    print("加载模型:", rs.XML_PATH)
    model = rs.mujoco.MjModel.from_xml_path(str(rs.XML_PATH))

    print("")
    print("=" * 80)
    print("开始右臂抓取参数校准")
    print("=" * 80)
    print("测试点:", TEST_POINTS)
    print("")

    summary = []

    for cfg in CONFIGS:
        apply_config(cfg)

        success_count = 0
        reasons = []

        print("")
        print("=" * 80)
        print("配置:", cfg["name"])
        print("open/pre_open/close:", cfg["open"], cfg["pre_open"], cfg["close"])
        print("grasp_offset:", cfg["grasp_offset"])
        print("pregrasp_offset:", cfg["pregrasp_offset"])
        print("=" * 80)

        for x, y in TEST_POINTS:
            success, reason, cube_pos = rs.run_one_position(model, x, y)

            if success:
                success_count += 1

            reasons.append(reason)

            mark = "O" if success else "X"
            print(
                f"x={x:.3f}, y={y:.3f} -> {mark}, "
                f"reason={reason:18s}, "
                f"cube={np.array2string(cube_pos, precision=3)}"
            )

        reason_stat = {r: reasons.count(r) for r in sorted(set(reasons))}
        summary.append((cfg["name"], success_count, reason_stat))

    print("")
    print("=" * 80)
    print("右臂校准总结")
    print("=" * 80)

    for name, success_count, reason_stat in summary:
        print(f"{name}: {success_count}/{len(TEST_POINTS)} success, reasons={reason_stat}")

    print("")
    print("判断：")
    print("1. 如果某个配置出现成功 O，下一步就用这个配置重新扫右臂全图。")
    print("2. 如果全部还是 lift_fail，说明右臂不是简单 offset 问题，要做右臂姿态 IK。")
    print("3. 如果 reverse_finger_direction 配置明显变好，说明右夹爪开合方向之前设反了。")


if __name__ == '__main__':
    main()