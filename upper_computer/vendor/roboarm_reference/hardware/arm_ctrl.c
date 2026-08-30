#include "arm_ctrl.h"
#include "Motor.h"
#include "fdcan.h"
#include "arm_Calculate.h"
#include "Servo.h"


#define MOTOR_TEST_COUNT 7U
#define MOTOR_ENABLE_PERIOD_MS 100U
#define MOTOR_COMMAND_PERIOD_MS 10U
#define MOTOR_ENABLE_COUNT 7U
#define MOTOR_ENABLE_INTERVAL_MS 100U
/* 位置速度模式的速度指令，1.0 rad/s 低速测试，确保安全 */
#define MOTOR_TEST_VELOCITY_RAD_S 0.5f
#define MOTOR_SMOOTH_COEFF 0.02f
#define MOTOR_SMOOTH_VELOCITY_RAD_S 4.0f
#define MOTOR_TARGET_REACHED_TOLERANCE_RAD 0.05f
/* 达妙 Position 绝对位置的机械零位偏置 */
#define MOTOR_POSITION_OFFSET_RAD 12.5f



/* 定义全局独立变量，方便在 Debug 时逐一监控 */
float J1_torque, J2_torque, J3_torque, J4_torque, J5_torque, J6_torque, J7_torque;
float J1_angle,  J2_angle,  J3_angle,  J4_angle,  J5_angle,  J6_angle,  J7_angle;

/* 存储正解算出的末端 4x4 齐次变换矩阵 */
float End_Transform[16]; 
/* 存储正解算出的末端位置坐标，单位为 m，顺序为 X/Y/Z */
float End_Position[3];
/* 存储打包好的关节角，用于传给运动学函数 */
float current_joint_angles[ARM_CALCULATE_JOINT_COUNT];

float position;
float val_position;
float target_pos[MOTOR_TEST_COUNT] = {
  MOTOR_POSITION_OFFSET_RAD, MOTOR_POSITION_OFFSET_RAD, MOTOR_POSITION_OFFSET_RAD,
  MOTOR_POSITION_OFFSET_RAD, MOTOR_POSITION_OFFSET_RAD, MOTOR_POSITION_OFFSET_RAD,
  MOTOR_POSITION_OFFSET_RAD
};



/* 引入 Motor.c 中的反馈数据数组 */
extern volatile DM_Motor_Data_Typedef DM_Data_t[7];
/* Servo.c command interface: 1 = close palm, 2 = open palm. */
extern uint8_t servo_cmd;

/* 七自由度机械臂电机配置 */
/* CAN路由: J1/J2→CAN3, J3/J4→CAN1, J5/J6/J7→CAN2 */
static DM_Motor_Info_Typedef motor_test[MOTOR_TEST_COUNT] = {
  {.CANFrameInfo = {.CAN_ID = 0x01U, .Master_ID = 0x11U}}, /* J1: DM-8009P */
  {.CANFrameInfo = {.CAN_ID = 0x02U, .Master_ID = 0x12U}}, /* J2: DM-8009P */
  {.CANFrameInfo = {.CAN_ID = 0x03U, .Master_ID = 0x13U}}, /* J3: DM-4340P */
  {.CANFrameInfo = {.CAN_ID = 0x04U, .Master_ID = 0x14U}}, /* J4: DM-4040 */
  {.CANFrameInfo = {.CAN_ID = 0x05U, .Master_ID = 0x15U}}, /* J5: DM-4310 */
  {.CANFrameInfo = {.CAN_ID = 0x06U, .Master_ID = 0x16U}}, /* J6: DM-4310 */
  {.CANFrameInfo = {.CAN_ID = 0x07U, .Master_ID = 0x17U}}  /* J7: DM-4310 */
};

/* 各关节动作1目标角度（正向） */
static float motor_action1_angle[MOTOR_TEST_COUNT] = {
  0.121500,
  -0.614748,
  0.269894,
  1.202983,
  0.456054,
  -0.100519,
  -0.474365
};

/* 各关节动作2目标角度（反向/不同位置） */

static float motor_action2_angle[MOTOR_TEST_COUNT] = {
    -0.367933,
    -0.083734,
    0.364500,
    0.476273,
    0.074197,
    -0.182155,
    -0.614367
};

/* 各关节动作1目标角度（正向） */
static float motor_action3_angle[MOTOR_TEST_COUNT] = {
    -0.226788,
    -0.886358,
    0.341992,
    0.685702,
    -0.215343,
    0.031472,
    -0.631533
};

/* 各关节动作2目标角度（反向/不同位置） */

static float motor_action4_angle[MOTOR_TEST_COUNT] = {
    0.066186,
    -0.282483,
    -0.148966,
    0.240902,
    0.876440,
    0.004005,
    -0.134470
};


/* 各关节运动延时时间（单位：毫秒） */
/* 所有关节同时启动（延时均为0ms） */
static uint32_t motor_delay_ms[MOTOR_TEST_COUNT] = {
  0U,     /* J1: 立即开始 */
  20U,     /* J2: 立即开始 */
  40U,     /* J3: 立即开始 */
  60U,     /* J4: 立即开始 */
  80U,     /* J5: 立即开始 */
  100U,     /* J6: 立即开始 */
  120U      /* J7: 立即开始 */
};

/* 动作状态枚举 */
typedef enum {
  ACTION_IDLE,       /* 空闲状态 */
  ACTION_1,          /* 执行动作1 */
  ACTION_1_WAIT,     /* 动作1完成后等待 */
  ACTION_2,          /* 执行动作2 */
  ACTION_2_WAIT,     /* 动作2完成后等待 */
  ACTION_3,          /* 执行动作3 */
  ACTION_3_WAIT,     /* 动作3完成后等待 */
  ACTION_4,          /* 执行动作4 */
  ACTION_4_WAIT,     /* 动作4完成后等待 */
  ACTION_RETURN,     /* 直接返回初始位置 */
  ACTION_RETURN_WAIT /* 返回完成后等待，准备下一轮 */
} MotorActionState;

/* 当前动作状态 */
static MotorActionState motor_action_state = ACTION_IDLE;

/* 当前动作阶段的开始时间 */
static uint32_t motor_action_start_time_ms;

/* 各关节上次发送命令的时间戳 */
static uint32_t motor_last_command_ms[MOTOR_TEST_COUNT];
//static float motor_current_angle[MOTOR_TEST_COUNT] = {
//  MOTOR_POSITION_OFFSET_RAD, MOTOR_POSITION_OFFSET_RAD, MOTOR_POSITION_OFFSET_RAD,
//  MOTOR_POSITION_OFFSET_RAD, MOTOR_POSITION_OFFSET_RAD, MOTOR_POSITION_OFFSET_RAD,
//  MOTOR_POSITION_OFFSET_RAD
//};

/* 各关节位置速度模式 Position 绝对角度限幅（相对机械零位限幅加 12.5rad） */
static const float motor_position_min[MOTOR_TEST_COUNT] = {
  9.5f,    /* J1: -3.0rad */
  9.5f,    /* J2: -3.0rad */
  11.0f,   /* J3: -1.5rad */
  12.5f,   /* J4: 0.0rad */
  11.0f,   /* J5: -1.5rad */
  12.25f,  /* J6: -0.25rad */
  11.0f    /* J7: -1.5rad */
};

static const float motor_position_max[MOTOR_TEST_COUNT] = {
  15.0f,  /* J1: 2.5rad */
  12.7f,  /* J2: 0.2rad */
  14.0f,  /* J3: 1.5rad */
  14.8f,  /* J4: 2.3rad */
  14.0f,  /* J5: 1.5rad */
  13.3f,  /* J6: 0.8rad */
  14.0f   /* J7: 1.5rad */
};

/* 系统上电后的基准时间，用于计算各关节的延时启动 */
static uint32_t motor_start_time_ms;

typedef struct
{
  /* 关节所在 CAN 总线的发送帧和电机 CAN_ID */
  FDCAN_TxFrame_TypeDef *frame;
  uint16_t can_id;
} MotorEnableItem;

/* 达妙电机使能列表*/
static MotorEnableItem motor_enable_list[MOTOR_ENABLE_COUNT] = {
  {&FDCAN3TxFrame, 0x01U}, /* J1 */
  {&FDCAN3TxFrame, 0x02U}, /* J2 */
  {&FDCAN1TxFrame, 0x03U}, /* J3 */
  {&FDCAN1TxFrame, 0x04U}, /* J4 */
  {&FDCAN2TxFrame, 0x05U}, /* J5 */
  {&FDCAN2TxFrame, 0x06U}, /* J6 */
  {&FDCAN2TxFrame, 0x07U}  /* J7 */
};
static uint8_t motor_enable_index;
static uint32_t motor_enable_next_ms;

/* 各动作阶段的持续时间（单位：毫秒） */
#define ACTION_1_DURATION_MS    1700U  /* 动作1持续时间 */
#define ACTION_1_WAIT_MS        1000U  /* 动作1完成后等待 */
#define ACTION_2_WAIT_MS        1000U  /* 动作2完成后等待 */
#define ACTION_3_DURATION_MS    1700U  /* 动作3持续时间 */
#define ACTION_3_WAIT_MS        600U  /* 动作3完成后等待 */
#define ACTION_4_WAIT_MS        600U  /* 动作4完成后等待 */
//#define ACTION_5_DURATION_MS    3000U  /* 动作5持续时间 */
//#define ACTION_5_WAIT_MS        1000U  /* 动作5完成后等待 */
//#define ACTION_6_DURATION_MS    3000U  /* 动作6持续时间 */
//#define ACTION_6_WAIT_MS        1000U  /* 动作6完成后等待 */
//#define ACTION_7_DURATION_MS    3000U  /* 动作7持续时间 */
//#define ACTION_7_WAIT_MS        1000U  /* 动作7完成后等待 */
//#define ACTION_8_DURATION_MS    3000U  /* 动作8持续时间 */
//#define ACTION_8_WAIT_MS        1000U  /* 动作8完成后等待 */
//#define ACTION_9_DURATION_MS    3000U  /* 动作9持续时间 */
//#define ACTION_9_WAIT_MS        1000U  /* 动作9完成后等待 */
//#define ACTION_10_DURATION_MS   3000U  /* 动作10持续时间 */
//#define ACTION_10_WAIT_MS       1000U  /* 动作10完成后等待 */
#define ACTION_RETURN_DURATION_MS 3000U /* 返回动作持续时间 */
#define ACTION_RETURN_WAIT_MS   2000U  /* 返回完成后等待 */

/** 获取指定关节所在的 CAN 发送帧。 */
/* CAN routing: J1/J2 -> CAN3, J3/J4 -> CAN1, J5/J6/J7 -> CAN2. */
static FDCAN_TxFrame_TypeDef *motor_test_frame(uint8_t index)
{
  if ((index == 0U) || (index == 1U))
    return &FDCAN3TxFrame;  /* J1, J2 → CAN3 */
  else if ((index == 2U) || (index == 3U))
    return &FDCAN1TxFrame;  /* J3, J4 → CAN1 */
  else
    return &FDCAN2TxFrame;  /* J5, J6, J7 -> CAN2 */
}

void Arm_Ctrl_MotorEnableInit(void)
{
  uint8_t index;

  /* 初始使能：逐个发送，电机之间间隔1ms */
  for (index = 0U; index < MOTOR_ENABLE_COUNT; index++)
  {
    DM_Motor_Command(motor_enable_list[index].frame,
                     motor_enable_list[index].can_id, Motor_Enable);
    HAL_Delay(1U);
  }

  motor_enable_index = 0U;
  motor_enable_next_ms = HAL_GetTick();
}

void Arm_Ctrl_MotorEnableLoop(void)
{
  uint32_t now = HAL_GetTick();

  /* 循环使能：每次只发送一个电机，避免总线瞬时拥塞 */
  if ((int32_t)(now - motor_enable_next_ms) >= 0)
  {
    DM_Motor_Command(motor_enable_list[motor_enable_index].frame,
                     motor_enable_list[motor_enable_index].can_id, Motor_Enable);
    motor_enable_index++;
    if (motor_enable_index >= MOTOR_ENABLE_COUNT)
      motor_enable_index = 0U;
    motor_enable_next_ms = now + MOTOR_ENABLE_INTERVAL_MS;
  }
}

/* 初始位置（所有关节为0） */
static float motor_initial_angle[MOTOR_TEST_COUNT] = {0.0f};

/**
 * 获取当前动作阶段的目标角度数组
 */
static float* motor_get_target_angle(void)
{
  switch (motor_action_state)
  {
    case ACTION_1:
    case ACTION_1_WAIT:
      return motor_action1_angle;  /* 动作1目标角度 */

    case ACTION_2:
    case ACTION_2_WAIT:
      return motor_action2_angle;  /* 动作2目标角度 */

    case ACTION_3:
    case ACTION_3_WAIT:
      return motor_action3_angle;  /* 动作3目标角度 */

    case ACTION_4:
    case ACTION_4_WAIT:
      return motor_action4_angle;  /* 动作4目标角度 */

//    case ACTION_5:
//    case ACTION_5_WAIT:
//      return motor_action5_angle;  /* 动作5目标角度 */

//    case ACTION_6:
//    case ACTION_6_WAIT:
//      return motor_action6_angle;  /* 动作6目标角度 */

//    case ACTION_7:
//    case ACTION_7_WAIT:
//      return motor_action7_angle;  /* 动作7目标角度 */

//    case ACTION_8:
//    case ACTION_8_WAIT:
//      return motor_action8_angle;  /* 动作8目标角度 */

//    case ACTION_9:
//    case ACTION_9_WAIT:
//      return motor_action9_angle;  /* 动作9目标角度 */

//    case ACTION_10:
//    case ACTION_10_WAIT:
//      return motor_action10_angle; /* 动作10目标角度 */

    case ACTION_RETURN:
    case ACTION_RETURN_WAIT:
      return motor_initial_angle;  /* 返回到初始位置 */

    case ACTION_IDLE:
    default:
      return motor_initial_angle;  /* 空闲状态返回初始位置 */
  }
}

static float motor_limit_position(uint8_t index, float position)
{
  /* 将发送目标钳制在对应关节的机械安全范围内 */
  if (position < motor_position_min[index])
    return motor_position_min[index];
  if (position > motor_position_max[index])
    return motor_position_max[index];
  return position;
}

/* All joint feedback values must be within this tolerance of the action pose. */
static uint8_t motor_has_reached_target(const float *target_angle)
{
  uint8_t index;

  for (index = 0U; index < MOTOR_TEST_COUNT; index++)
  {
    float target_position = target_angle[index] + MOTOR_POSITION_OFFSET_RAD;
    float position_error;

    target_position = motor_limit_position(index, target_position);
    position_error = DM_Data_t[index].Position - target_position;
    if (position_error < 0.0f)
      position_error = -position_error;

    if (position_error > MOTOR_TARGET_REACHED_TOLERANCE_RAD)
      return 0U;
  }

  return 1U;
}

void Arm_Ctrl_Init(void)
{
  motor_action_state = ACTION_IDLE;
  motor_action_start_time_ms = 0U;
  motor_start_time_ms = HAL_GetTick();
  motor_enable_index = 0U;
  motor_enable_next_ms = motor_start_time_ms;
}

/**
 * @brief  机械臂计算层单步逻辑更新函数 (逐一赋值版)
 */
 void Arm_Calculate_task_logical(void)
{
    /* 1. 逐一获取电机的扭矩/电流反馈 */
    /* 注意：根据你的 DM_Motor_Data_Typedef 结构体，这里取 Current 或 Velocity */
    J1_torque = DM_Data_t[0].Torque; 
    J2_torque = DM_Data_t[1].Torque;
    J3_torque = DM_Data_t[2].Torque;
    J4_torque = DM_Data_t[3].Torque;
    J5_torque = DM_Data_t[4].Torque;
    J6_torque = DM_Data_t[5].Torque;
    J7_torque = DM_Data_t[6].Torque;

    /* 2. 逐一获取电机的物理角度，并严格减去 12.5rad 的达妙机械零位偏置 */
    J1_angle = DM_Data_t[0].Position - MOTOR_POSITION_OFFSET_RAD;
    J2_angle = DM_Data_t[1].Position - MOTOR_POSITION_OFFSET_RAD;
    J3_angle = DM_Data_t[2].Position - MOTOR_POSITION_OFFSET_RAD;
    J4_angle = DM_Data_t[3].Position - MOTOR_POSITION_OFFSET_RAD;
    J5_angle = DM_Data_t[4].Position - MOTOR_POSITION_OFFSET_RAD;
    J6_angle = DM_Data_t[5].Position - MOTOR_POSITION_OFFSET_RAD;
    J7_angle = DM_Data_t[6].Position - MOTOR_POSITION_OFFSET_RAD;

    /* 将散装的变量打包成数组，因为运动学算法接口需要数组形参 */
    current_joint_angles[0] = J1_angle;
    current_joint_angles[1] = J2_angle;
    current_joint_angles[2] = J3_angle;
    current_joint_angles[3] = J4_angle;
    current_joint_angles[4] = J5_angle;
    current_joint_angles[5] = J6_angle;
    current_joint_angles[6] = J7_angle;

    /* 3. 调用正运动学解算，计算结果保存在 End_Transform 矩阵中 */
    Arm_Calculate_ForwardKinematics(current_joint_angles, End_Transform);

    /* 提取末端位置，便于直接在调试器中观察 */
    End_Position[0] = End_Transform[3];   // X，单位 m
    End_Position[1] = End_Transform[7];   // Y，单位 m
    End_Position[2] = End_Transform[11];  // Z，单位 m
}

/**
 * 主循环电机测试：七自由度机械臂各关节动作状态机。
 * 执行流程：动作1 → 等待 → 动作2 → 等待 → 直接返回初始位置 → 等待 → 重复
 */
/* Active sequence: ACTION_1 -> ACTION_2 -> ACTION_3 -> ACTION_4
 *                  -> ACTION_RETURN -> ACTION_1. */
void Arm_Ctrl_Step(void)
{
  uint8_t index;
//  float position;
//  float target_pos;
  uint32_t now = HAL_GetTick();

  /* 10ms周期：重发使能帧（保持电机使能状态） */
  Arm_Ctrl_MotorEnableLoop();

  switch (motor_action_state)
  {
    case ACTION_IDLE:
      /* 初始化完成后，立即开始动作1 */
      motor_action_state = ACTION_1;
      motor_action_start_time_ms = now;
      break;

    case ACTION_1:
      /* 动作1：持续指定时间 */
      if ((now - motor_action_start_time_ms) >= ACTION_1_DURATION_MS)
      {
        motor_action_state = ACTION_1_WAIT;
        motor_action_start_time_ms = now;
      }
      break;

    case ACTION_1_WAIT:
      /* 动作1完成后等待 */
      if ((now - motor_action_start_time_ms) >= ACTION_1_WAIT_MS)
      {
        motor_action_state = ACTION_2;
        motor_action_start_time_ms = now;
      }
      break;

    case ACTION_2:
      /* 动作2：持续指定时间 */
      if (motor_has_reached_target(motor_action2_angle) != 0U)
      {
        /* Action 2 target reached: request a non-blocking palm close. */
        servo_cmd = 1U;
        motor_action_state = ACTION_2_WAIT;
        motor_action_start_time_ms = now;
      }
      break;

    case ACTION_2_WAIT:
      /* 动作2完成后等待，准备执行动作3 */
      if ((now - motor_action_start_time_ms) >= ACTION_2_WAIT_MS)
      {
        motor_action_state = ACTION_3;
        motor_action_start_time_ms = now;
      }
      break;

    case ACTION_3:
      if ((now - motor_action_start_time_ms) >= ACTION_3_DURATION_MS)
      {
        motor_action_state = ACTION_3_WAIT;
        motor_action_start_time_ms = now;
      }
      break;

    case ACTION_3_WAIT:
      if ((now - motor_action_start_time_ms) >= ACTION_3_WAIT_MS)
      {
        motor_action_state = ACTION_4;
        motor_action_start_time_ms = now;
      }
      break;

    case ACTION_4:
      if (motor_has_reached_target(motor_action4_angle) != 0U)
      {
        /* Action 4 target reached: request a non-blocking palm open. */
        servo_cmd = 2U;
        motor_action_state = ACTION_4_WAIT;
        motor_action_start_time_ms = now;
      }
      break;

    case ACTION_4_WAIT:
      if ((now - motor_action_start_time_ms) >= ACTION_4_WAIT_MS)
      {
        motor_action_state = ACTION_RETURN;
        motor_action_start_time_ms = now;
      }
      break;

//    case ACTION_5:
//      if ((now - motor_action_start_time_ms) >= ACTION_5_DURATION_MS)
//      {
//        motor_action_state = ACTION_5_WAIT;
//        motor_action_start_time_ms = now;
//      }
//      break;

//    case ACTION_5_WAIT:
//      if ((now - motor_action_start_time_ms) >= ACTION_5_WAIT_MS)
//      {
//        motor_action_state = ACTION_6;
//        motor_action_start_time_ms = now;
//      }
//      break;

//    case ACTION_6:
//      if ((now - motor_action_start_time_ms) >= ACTION_6_DURATION_MS)
//      {
//        motor_action_state = ACTION_6_WAIT;
//        motor_action_start_time_ms = now;
//      }
//      break;

//    case ACTION_6_WAIT:
//      if ((now - motor_action_start_time_ms) >= ACTION_6_WAIT_MS)
//      {
//        motor_action_state = ACTION_7;
//        motor_action_start_time_ms = now;
//      }
//      break;

//    case ACTION_7:
//      if ((now - motor_action_start_time_ms) >= ACTION_7_DURATION_MS)
//      {
//        motor_action_state = ACTION_7_WAIT;
//        motor_action_start_time_ms = now;
//      }
//      break;

//    case ACTION_7_WAIT:
//      if ((now - motor_action_start_time_ms) >= ACTION_7_WAIT_MS)
//      {
//        motor_action_state = ACTION_8;
//        motor_action_start_time_ms = now;
//      }
//      break;

//    case ACTION_8:
//      if ((now - motor_action_start_time_ms) >= ACTION_8_DURATION_MS)
//      {
//        motor_action_state = ACTION_8_WAIT;
//        motor_action_start_time_ms = now;
//      }
//      break;

//    case ACTION_8_WAIT:
//      if ((now - motor_action_start_time_ms) >= ACTION_8_WAIT_MS)
//      {
//        motor_action_state = ACTION_9;
//        motor_action_start_time_ms = now;
//      }
//      break;

//    case ACTION_9:
//      if ((now - motor_action_start_time_ms) >= ACTION_9_DURATION_MS)
//      {
//        motor_action_state = ACTION_9_WAIT;
//        motor_action_start_time_ms = now;
//      }
//      break;

//    case ACTION_9_WAIT:
//      if ((now - motor_action_start_time_ms) >= ACTION_9_WAIT_MS)
//      {
//        motor_action_state = ACTION_10;
//        motor_action_start_time_ms = now;
//      }
//      break;

//    case ACTION_10:
//      if ((now - motor_action_start_time_ms) >= ACTION_10_DURATION_MS)
//      {
//        motor_action_state = ACTION_10_WAIT;
//        motor_action_start_time_ms = now;
//      }
//      break;

//    case ACTION_10_WAIT:
//      if ((now - motor_action_start_time_ms) >= ACTION_10_WAIT_MS)
//      {
//        motor_action_state = ACTION_RETURN;
//        motor_action_start_time_ms = now;
//      }
//      break;

    case ACTION_RETURN:
      /* 直接返回初始位置 */
      if ((now - motor_action_start_time_ms) >= ACTION_RETURN_DURATION_MS)
      {
        motor_action_state = ACTION_RETURN_WAIT;
        motor_action_start_time_ms = now;
      }
      break;

    case ACTION_RETURN_WAIT:
      /* 返回完成后等待，准备下一轮 */
      if ((now - motor_action_start_time_ms) >= ACTION_RETURN_WAIT_MS)
      {
        motor_action_state = ACTION_1;  /* 重新开始动作1 */
        motor_action_start_time_ms = now;
      }
      break;

    default:
      motor_action_state = ACTION_IDLE;
      break;
  }

  /* 获取当前动作阶段的目标角度 */
  float* target_angle = motor_get_target_angle();

/* 按延时时间依次发送各关节位置速度控制帧 */
  for (index = 0U; index < MOTOR_TEST_COUNT; index++)
  {
    /* 检查该关节是否达到延时启动时间 */
    if ((now - motor_start_time_ms) >= motor_delay_ms[index])
    {
      /* 10ms周期：发送位置速度控制帧（100Hz控制频率） */
      if ((now - motor_last_command_ms[index]) >= MOTOR_COMMAND_PERIOD_MS)
      {
        motor_last_command_ms[index] = now;
        
        /* 1. 算出最终目标的绝对位置（基准0 + 机械偏置12.5） */
        position = target_angle[index] + MOTOR_POSITION_OFFSET_RAD;
        /* 2. 目标位置先限幅，避免插补轨迹朝机械限位外运动 */
        position = motor_limit_position(index, position);
        
        /* --- 以下为丝滑优化核心算法 --- */
        
        /* 3. 计算理想步长 */
        float diff = position - target_pos[index];
        float step = diff * MOTOR_SMOOTH_COEFF;
        
        /* 4. 【新增】限制最大步长，柔化启动瞬间 (限制软件最高插补速度约为 1.5rad/s) */
        float max_step = 0.015f; 
        if (step > max_step) step = max_step;
        else if (step < -max_step) step = -max_step;
        
        target_pos[index] += step;
        
        /* 5. 【新增】动态速度前馈：让电机的限速恰好等于本次软件插补的瞬时速度 */
        /* 公式：步长 * 100Hz = 当前真实角速度。外加 0.2rad/s 作为跟踪余量防滞后 */
        float dynamic_vel = fabs(step) * 100.0f + 0.2f; 
        /* 安全兜底限幅，不超过宏定义的最高限速 */
        if (dynamic_vel > MOTOR_SMOOTH_VELOCITY_RAD_S) {
            dynamic_vel = MOTOR_SMOOTH_VELOCITY_RAD_S;
        }
        
        /* 6. 插补后的实际发送值再次限幅，并按照要求的减去 OFFSET */
        val_position = motor_limit_position(index, target_pos[index]) - MOTOR_POSITION_OFFSET_RAD;        
        
        /* 7. 下发 CAN 指令给电机 (注意：这里填入的是动态算出的 dynamic_vel，而不是固定的宏) */
        DM_Motor_CAN_TxMessage(motor_test_frame(index), &motor_test[index],
                               Position_Velocity_Mode, val_position, dynamic_vel,
                               0.0f, 0.0f, 0.0f);
      }
    }
  }
}

void arm_task_logical(void)
{
    /* 七轴位置速度控制主循环 */
    Arm_Ctrl_Step();

    /* 调用舵机状态机更新函数 */
    Servo_Control_Step();
}
