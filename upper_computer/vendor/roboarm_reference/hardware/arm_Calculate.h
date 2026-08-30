#ifndef __ARM_CALCULATE_H__
#define __ARM_CALCULATE_H__

#include <stdint.h>

/* 七轴机械臂的关节数量 */
#define ARM_CALCULATE_JOINT_COUNT 7U

/* 参考工程采用的改进 DH 参数，角度单位为 rad，长度单位为 m */
typedef struct
{
  float alpha;        /* 连杆扭转角：绕 X 轴旋转，单位 rad */
  float a;            /* 连杆长度：沿 X 轴平移，单位 m */
  float theta_offset; /* 关节安装零位偏角，单位 rad */
  float d;            /* 连杆偏距：沿 Z 轴平移，单位 m */
} Arm_DH_Param_t;

/* 逆运动学迭代参数 */
typedef struct
{
  uint16_t max_iterations;    /* 最大迭代次数 */
  float position_tolerance;   /* 末端位置收敛容差，单位 m */
  float orientation_tolerance;/* 末端姿态收敛容差，单位 rad */
  float damping;              /* 阻尼系数，增大可提高奇异点附近的稳定性 */
  float max_joint_step;       /* 单次迭代允许的最大关节变化量，单位 rad */
  float orientation_weight;   /* 姿态误差相对于位置误差的权重 */
} Arm_IK_Config_t;

/* 逆运动学解算结果 */
typedef enum
{
  ARM_IK_SUCCESS = 0,       /* 已在容差范围内找到逆解 */
  ARM_IK_NOT_CONVERGED,     /* 达到最大迭代次数仍未收敛 */
  ARM_IK_SINGULAR,          /* 线性方程组无法求解 */
  ARM_IK_INVALID_ARGUMENT   /* 空指针或配置参数不合法 */
} Arm_IK_Status_t;

/* J1-J7 的 DH 参数表，顺序与 arm_ctrl.c 中的电机顺序一致 */
extern const Arm_DH_Param_t Arm_DH_Table[ARM_CALCULATE_JOINT_COUNT];

/* J1-J7 相对机械零位的关节角范围，单位为 rad，不包含电机协议的 12.5rad 偏置 */
extern const float Arm_Joint_Min[ARM_CALCULATE_JOINT_COUNT];
extern const float Arm_Joint_Max[ARM_CALCULATE_JOINT_COUNT];

/* ============================ 正运动学解算 ============================ */

/*
 * 根据单个关节的改进 DH 参数生成 4x4 齐次变换矩阵。
 * matrix 使用行优先排列，位置坐标 X/Y/Z 分别位于 matrix[3]/[7]/[11]。
 */
void Arm_Calculate_DHMatrix(float matrix[16], float theta, float alpha, float a, float d);

/*
 * 根据七个相对机械零位关节角计算基座到末端的齐次变换矩阵。
 * joint_angle 单位为 rad，不能包含达妙 Position 使用的 12.5rad 协议偏置。
 */
void Arm_Calculate_ForwardKinematics(const float joint_angle[ARM_CALCULATE_JOINT_COUNT],
                                     float transform[16]);

/* ============================ 逆运动学解算 ============================ */

/* 获取一组适用于当前七轴机械臂的默认逆解参数 */
void Arm_Calculate_GetDefaultIKConfig(Arm_IK_Config_t *config);

/*
 * 根据末端目标齐次变换矩阵迭代求解七个关节角。
 * target_transform 按行优先排列，其中 [3]/[7]/[11] 是目标 X/Y/Z，单位为 m。
 * initial_joint_angle 为迭代初值，七轴冗余机械臂的解会受到该初值影响。
 * initial_joint_angle 应使用当前反馈角并减去 12.5rad 协议偏置。
 * config 可以传入 0，函数将自动使用默认逆解配置。
 * 仅返回 ARM_IK_SUCCESS 时 joint_angle_out 才是新解，否则保持为初始关节角。
 * 逆解结果仍需经过轨迹规划，禁止直接作为阶跃位置指令发送给电机。
 */
Arm_IK_Status_t Arm_Calculate_InverseKinematics(
    const float target_transform[16],
    const float initial_joint_angle[ARM_CALCULATE_JOINT_COUNT],
    float joint_angle_out[ARM_CALCULATE_JOINT_COUNT],
    const Arm_IK_Config_t *config);

#endif
