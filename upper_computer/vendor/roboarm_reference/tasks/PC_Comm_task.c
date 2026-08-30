#include "PC_Comm_task.h"
#include "PC_comm.h"
#include "FreeRTOS.h"
#include "task.h"

void PC_Comm_Task(void *argument)
{
  PC_Comm_Init();

  while (1)
  {
    PC_Data_Send();
    vTaskDelay(10);
  }
}
