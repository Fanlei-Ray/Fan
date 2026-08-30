#ifndef Arm_Ctrl_H
#define Arm_Ctrl_H

#include "main.h"

/* 末端相对于基座的位置坐标，单位为 m，顺序为 X/Y/Z */
extern float End_Position[3];

/* 初始化七轴关节控制状态和软件计时器 */
void Arm_Ctrl_Init(void);
/* 上电时依次发送各关节的达妙使能命令 */
void Arm_Ctrl_MotorEnableInit(void);
/* 周期性重发使能命令，保持电机使能状态 */
void Arm_Ctrl_MotorEnableLoop(void);
/* 七轴动作状态机与位置速度控制的周期执行函数 */
void Arm_Ctrl_Step(void);
/* 机械臂计算层单步逻辑更新函数 */
void Arm_Calculate_task_logical(void);
/* 机械臂运行逻辑执行代码 */
void arm_task_logical(void);
#endif
