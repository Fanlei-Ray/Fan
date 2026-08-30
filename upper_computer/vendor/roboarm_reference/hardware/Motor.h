#ifndef Motor_H
#define Motor_H

#include "fdcan.h"
#include "stdbool.h"

typedef enum{

  Motor_Enable,
  Motor_Disable,
  Motor_Save_Zero_Position,
  DM_Motor_CMD_Type_Num,

}DM_Motor_CMD_e;										//电机状态枚举结构体

typedef enum{

  MIT_Mode,
  Position_Velocity_Mode,
  Velocity_Mode,
  DM_Motor_Mode_Type_Num,

}DM_Motor_Mode_e;										//电机模式枚举结构体


typedef struct 
{
  int16_t  State; 	
  uint16_t  P_int;
  uint16_t  V_int;
  uint16_t  T_int;
  float  Position;  
  float  Velocity;  
  float  Torque;  
  float  Temperature_MOS;   
  float  Temperature_Rotor;  
}DM_Motor_Data_Typedef;									//电机消息结构体

typedef struct
{
  uint32_t Master_ID;   								//反馈帧
  uint32_t CAN_ID;  									//接收帧
}Motor_CANFrameInfo_typedef;							//电机ID结构体

typedef struct
{
	uint16_t ID;										//···
	Motor_CANFrameInfo_typedef CANFrameInfo;			//电机ID项
	DM_Motor_Data_Typedef Data;  						//电机消息项
}DM_Motor_Info_Typedef;									//电机头结构体

typedef struct
{
	float  KP;
	float  KD;
	float  Position; 
	float  Velocity;  	
	float  Torque;  
	
}DM_Motor_Control_Typedef;								//电机控制参数结构体

extern DM_Motor_Control_Typedef DM_Motor_Control;

void HAL_FDCAN_RxFifo0Callback(FDCAN_HandleTypeDef *hfdcan, uint32_t RxFifo0ITs);
void HAL_FDCAN_RxFifo1Callback(FDCAN_HandleTypeDef *hfdcan, uint32_t RxFifo1ITs);
extern void DM_Motor_Command(FDCAN_TxFrame_TypeDef *TxFrame,uint16_t CAN_ID,uint8_t CMD);
void Enable_R(void);
void Enable_L(void);


extern void DM_Motor_CAN_TxMessage(FDCAN_TxFrame_TypeDef *TxFrame,DM_Motor_Info_Typedef *DM_Motor,uint8_t Mode,
	                                             float Postion, float Velocity, float KP, float KD, float Torque);

#endif
