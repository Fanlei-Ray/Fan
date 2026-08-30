#include "Servo.h"

#include "main.h"
#include "tim.h"

#define SERVO_CLOSE_DURATION_TICKS 120U  /* 1200 ms at a 10 ms task period */
#define SERVO_OPEN_DURATION_TICKS   40U  /* 600 ms at a 10 ms task period */

/* ================= 舵机控制专用变量 ================= */
uint8_t servo_cmd = 0;           // 外部指令接口：1=请求合拢，2=请求张开

/* 物理互锁标志：假设上电初始状态是张开的，赋值为 1。如果是合拢的，请改为 0 */
uint8_t claw_is_open = 1; 

uint8_t servo_step = 0;   // 运行阶段：0=空闲, 1=转动中, 2=停止缓冲中
uint16_t servo_timer = 0; // 计数器 (每次+1代表10ms)
/* ==================================================== */

/**
 * @brief 舵机非阻塞控制逻辑 (需在 10ms 周期的任务中循环调用)
 */
void Servo_Control_Step(void)
{
    /* 1. 指令触发与互锁判断 (仅在舵机空闲时接收指令) */
    if (servo_step == 0) 
    {
        /* 【互锁逻辑】：只有收到合拢指令(1) 且 当前是张开的(1) 才能合拢 */
        if (servo_cmd == 1 && claw_is_open == 1) 
        { 
            Servo_SetPwm(1, 1000); Servo_SetPwm(2, 1000); 
            Servo_SetPwm(3, 1000); Servo_SetPwm(4, 1000);
            
            claw_is_open = 0;  // 动作一触发，立即改变状态标记为“已合拢”
            servo_step = 1;    // 进入转动阶段
            servo_timer = 0;
        } 
        /* 【互锁逻辑】：只有收到张开指令(2) 且 当前是合拢的(0) 才能张开 */
        else if (servo_cmd == 2 && claw_is_open == 0) 
        {
            Servo_SetPwm(1, 2000); Servo_SetPwm(2, 2000); 
            Servo_SetPwm(3, 2000); Servo_SetPwm(4, 2000);
            
            claw_is_open = 1;  // 动作一触发，立即改变状态标记为“已张开”
            servo_step = 1;    // 进入转动阶段
            servo_timer = 0;
        }
        
        /* 无论指令是否合法（比如在合拢状态下又发了合拢指令），一律清空，防止指令堆积 */
        servo_cmd = 0; 
    }

    /* 2. 共用延时执行逻辑 (张和合的动作时长和停止命令完全一样) */
    else if (servo_step == 1) 
    {
        servo_timer++;
        /* claw_is_open is updated when the motion starts, so it identifies
           whether the active motion is opening or closing. */
        if (servo_timer >= (claw_is_open ? SERVO_OPEN_DURATION_TICKS : SERVO_CLOSE_DURATION_TICKS)) {
            Servo_SetPwm(1, 1500); Servo_SetPwm(2, 1500); 
            Servo_SetPwm(3, 1500); Servo_SetPwm(4, 1500); // 发送停止指令
            
            servo_step = 2;        // 进入缓冲阶段
            servo_timer = 0;
        }
    } 
    else if (servo_step == 2) 
    {
        servo_timer++;
        if (servo_timer >= 10) {   // 缓冲时间到达 100ms
            servo_step = 0;        // 彻底完成，回到空闲状态，准备接收下一次反向指令
        }
    }
}

void Servo_SetPwm(uint8_t servo, uint16_t pulse_us)
{
  switch (servo)
  {
    case 1: __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_1, pulse_us); break;
    case 2: __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_3, pulse_us); break;
    case 3: __HAL_TIM_SET_COMPARE(&htim1, TIM_CHANNEL_1, pulse_us); break;
    case 4: __HAL_TIM_SET_COMPARE(&htim1, TIM_CHANNEL_3, pulse_us); break;
    default: break;
  }
}

void Servo_StopAll(void)
{
  Servo_SetPwm(1, 1500);
  Servo_SetPwm(2, 1500);
  Servo_SetPwm(3, 1500);
  Servo_SetPwm(4, 1500);
}

void Servo_EnablePower(void)
{
  HAL_GPIO_WritePin(PWM_5V_EN_GPIO_Port, PWM_5V_EN_Pin, GPIO_PIN_SET);
}

void Servo_Init(void)
{
  HAL_TIM_PWM_Start(&htim2, TIM_CHANNEL_1);
  HAL_TIM_PWM_Start(&htim2, TIM_CHANNEL_3);
  HAL_TIM_PWM_Start(&htim1, TIM_CHANNEL_1);
  HAL_TIM_PWM_Start(&htim1, TIM_CHANNEL_3);
  Servo_StopAll();
}

//void Servo_he(void)
//{
//    Servo_SetPwm(1, 1000);   /* 正转 */
//    Servo_SetPwm(2, 1000);   /* 正转 */
//    Servo_SetPwm(3, 1000);   /* 正转 */
//    Servo_SetPwm(4, 1000);   /* 正转 */
//      
////      motor_test_step();
//      
//      HAL_Delay(1200U);

//    Servo_SetPwm(1, 1500);   /* 停止 */
//    Servo_SetPwm(2, 1500);   /* 停止 */
//    Servo_SetPwm(3, 1500);   /* 停止 */
//    Servo_SetPwm(4, 1500);   /* 停止 */
//     HAL_Delay(100U);
//}

//void Servo_kai(void)
//{
//    Servo_SetPwm(1, 2000);   /* 反转 */
//    Servo_SetPwm(2, 2000);   /* 反转 */
//    Servo_SetPwm(3, 2000);   /* 反转 */
//    Servo_SetPwm(4, 2000);   /* 反转 */
//      
////      motor_test_step();
//      
//      HAL_Delay(1200U);

//    Servo_SetPwm(1, 1500);   /* 停止 */
//    Servo_SetPwm(2, 1500);   /* 停止 */
//    Servo_SetPwm(3, 1500);   /* 停止 */
//    Servo_SetPwm(4, 1500);   /* 停止 */
//     HAL_Delay(100U);
//}
