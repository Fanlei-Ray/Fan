#ifndef SERVO_H
#define SERVO_H

#include <stdint.h>

/* 初始化 TIM1/TIM2 的四路 PWM 舵机输出 */
void Servo_Init(void);
/* 打开 PWM_5V_EN，为舵机电源端供电 */
void Servo_EnablePower(void);
/* 设置指定通道的 PWM 高电平宽度，单位为微秒 */
void Servo_SetPwm(uint8_t servo, uint16_t pulse_us);
/* 将四个连续旋转舵机置于 1500us 中位停止脉宽 */
void Servo_StopAll(void);
///* 舵机开合动作接口 */
//void Servo_kai(void);
//void Servo_he(void);
//brief 舵机非阻塞控制逻辑 (需在 10ms 周期的任务中循环调用)
void Servo_Control_Step(void);


#endif
