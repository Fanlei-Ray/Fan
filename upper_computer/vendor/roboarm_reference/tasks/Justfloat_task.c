#include "Justfloat_task.h"
#include "FreeRTOS.h"
#include "task.h"
#include "Justfloat.h"
#include "arm_ctrl.h"
#include "cmsis_os2.h"
#include "Motor.h"

extern float J1_angle,  J2_angle,  J3_angle,  J4_angle,  J5_angle,  J6_angle,  J7_angle;

void Justfloat_Task(void *argument)
{
   while(1)
   {
       
        /* 将正解算出的末端位置作为 JustFloat 的前三个通道发送。 */
//        Justfloat_Buffer[0] = (float)End_Position[0];
//        Justfloat_Buffer[1] = (float)End_Position[1];
//        Justfloat_Buffer[2] = (float)End_Position[2];
       
        Justfloat_Buffer[0] = (float)J1_angle;
        Justfloat_Buffer[1] = (float)J2_angle;
        Justfloat_Buffer[2] = (float)J3_angle;
        Justfloat_Buffer[3] = (float)J4_angle;
        Justfloat_Buffer[4] = (float)J5_angle;
        Justfloat_Buffer[5] = (float)J6_angle;
        Justfloat_Buffer[6] = (float)J7_angle;
       
        Justfloat_Buffer[7] = (float)End_Position[0];
        Justfloat_Buffer[8] = (float)End_Position[1];
        Justfloat_Buffer[9] = (float)End_Position[2];

        JustFloat(Justfloat_Buffer, 10U);
       
      vTaskDelay(10);
   }
}
