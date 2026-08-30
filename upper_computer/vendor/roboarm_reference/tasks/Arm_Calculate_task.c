#include "Arm_Calculate_task.h"
#include "FreeRTOS.h"
#include "task.h"
#include "cmsis_os.h"
#include "main.h"
#include "arm_ctrl.h"

void Arm_Calculate_Task(void const *argument)
{
   vTaskDelay(100);
   

   while(1)
   {        
    /* 一次性调用所有的提取、正解和状态机逻辑 */
      Arm_Calculate_task_logical();
       
      vTaskDelay(10);
   }
}

