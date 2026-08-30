#include "Arm_task.h"
#include "FreeRTOS.h"
#include "task.h"
#include "cmsis_os.h"
#include "main.h"
#include "Servo.h"


void Arm_Task(void const *argument)
{      
  vTaskDelay(100);
   /* 初始化七轴控制状态，并执行上电使能流程 */
  Arm_Ctrl_Init();
  Arm_Ctrl_MotorEnableInit();
  vTaskDelay(100); // 延时 100ms，等待所有电机的 CAN 反馈帧到达并被 FIFO 回调解析

   while(1)
   {       
      arm_task_logical();
       
      vTaskDelay(10);
   }
}

