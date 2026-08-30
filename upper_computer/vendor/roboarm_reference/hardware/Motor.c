#include "Motor.h"

#define P_MAX 12.5f
#define V_MAX 10.f
#define T_MAX 28.f


DM_Motor_Control_Typedef DM_Motor_Control;
DM_Motor_Data_Typedef DM_Data_t[7]; /* Updated in FDCAN ISR, read by control tasks. */

static float uint_to_float(int X_int, float X_min, float X_max, int Bits)
{
    float span = X_max - X_min;
    float offset = X_min;
	
    return ((float)X_int)*span/((float)((1<<Bits)-1)) + offset;
}

static int float_to_uint(float X_float, float X_min, float X_max, int bits)
{
    float span = X_max - X_min;
    float offset = X_min;
    return (int) ((X_float-offset)*((float)((1<<bits)-1))/span);
}


void DM_Motor_Command(FDCAN_TxFrame_TypeDef *TxFrame,uint16_t CAN_ID,uint8_t CMD)
{
	TxFrame->Header.Identifier = CAN_ID + 0x100U;
  	
	TxFrame->Data[0] = 0xFF;
	TxFrame->Data[1] = 0xFF;
 	TxFrame->Data[2] = 0xFF;
	TxFrame->Data[3] = 0xFF;
	TxFrame->Data[4] = 0xFF;
	TxFrame->Data[5] = 0xFF;
	TxFrame->Data[6] = 0xFF;
	
	switch(CMD)
	{
		 
		case Motor_Enable :
	        TxFrame->Data[7] = 0xFC; 
	    break;
      
		case Motor_Disable :
	        TxFrame->Data[7] = 0xFD; 
        break;
      
		case Motor_Save_Zero_Position :
	        TxFrame->Data[7] = 0xFE; 
		break;
			
		default:
	    break;   
	}
	
    HAL_FDCAN_AddMessageToTxFifoQ(TxFrame->hcan,&TxFrame->Header,TxFrame->Data);		//hcan��can�ߣ�ID��Header��

}



void DM_Motor_CAN_TxMessage(FDCAN_TxFrame_TypeDef *TxFrame,DM_Motor_Info_Typedef *DM_Motor,uint8_t Mode,float Postion, float Velocity, float KP, float KD, float Torque)
{

	if(Mode > Velocity_Mode) Mode = MIT_Mode;	
		 
	if(Mode == MIT_Mode)
	{
		 
		static uint16_t Postion_Tmp,Velocity_Tmp,Torque_Tmp,KP_Tmp,KD_Tmp;
			 
		Postion_Tmp  =  float_to_uint(Postion,-P_MAX,P_MAX,16) ;
		Velocity_Tmp =  float_to_uint(Velocity,-V_MAX,V_MAX,12);
		Torque_Tmp = float_to_uint(Torque,-T_MAX,T_MAX,12);
		
		KP_Tmp = float_to_uint(KP,0,500,12);
		KD_Tmp = float_to_uint(KD,0,5,12);

		TxFrame->Header.Identifier = DM_Motor->CANFrameInfo.CAN_ID;
			 
		TxFrame->Data[0] = (uint8_t)(Postion_Tmp>>8);
		TxFrame->Data[1] = (uint8_t)(Postion_Tmp);
		TxFrame->Data[2] = (uint8_t)(Velocity_Tmp>>4);
		TxFrame->Data[3] = (uint8_t)((Velocity_Tmp&0x0F)<<4) | (KP_Tmp>>8);
		TxFrame->Data[4] = (uint8_t)(KP_Tmp);
		TxFrame->Data[5] = (uint8_t)(KD_Tmp>>4);
		TxFrame->Data[6] = (uint8_t)((KD_Tmp&0x0F)<<4) | (Torque_Tmp>>8);
		TxFrame->Data[7] = (uint8_t)(Torque_Tmp);
			
		HAL_FDCAN_AddMessageToTxFifoQ(TxFrame->hcan,&TxFrame->Header,TxFrame->Data);
	 
	}
	else if(Mode == Position_Velocity_Mode)
	{
		
		KP = 0; KD = 0; Torque = 0;
		 
		uint8_t *Postion_Tmp,*Velocity_Tmp;
		   
		Postion_Tmp = (uint8_t *)&Postion; 
		Velocity_Tmp = (uint8_t *)&Velocity; 
		 
	    TxFrame->Header.Identifier = DM_Motor->CANFrameInfo.CAN_ID + 0x100;
			 
		TxFrame->Data[0] = *(Postion_Tmp);
		TxFrame->Data[1] = *(Postion_Tmp + 1);
		TxFrame->Data[2] = *(Postion_Tmp + 2);
		TxFrame->Data[3] = *(Postion_Tmp + 3);
		TxFrame->Data[4] = *(Velocity_Tmp);
		TxFrame->Data[5] = *(Velocity_Tmp + 1);
		TxFrame->Data[6] = *(Velocity_Tmp + 2);
		TxFrame->Data[7] = *(Velocity_Tmp + 3);
			
		HAL_FDCAN_AddMessageToTxFifoQ(TxFrame->hcan,&TxFrame->Header,TxFrame->Data);
	 
	}
	else if(Mode == Velocity_Mode)
	{
	 
		Postion = 0;KP = 0; KD = 0; Torque = 0;
		 
		uint8_t *Velocity_Tmp;
		   
		Velocity_Tmp = (uint8_t *)&Velocity; 
		 
	    TxFrame->Header.Identifier = DM_Motor->CANFrameInfo.CAN_ID + 0x200;
			 
		TxFrame->Data[0] = *(Velocity_Tmp);
		TxFrame->Data[1] = *(Velocity_Tmp + 1);
		TxFrame->Data[2] = *(Velocity_Tmp + 2);
		TxFrame->Data[3] = *(Velocity_Tmp + 3);
		TxFrame->Data[4] = 0;
		TxFrame->Data[5] = 0;
		TxFrame->Data[6] = 0;
		TxFrame->Data[7] = 0;
			
		HAL_FDCAN_AddMessageToTxFifoQ(TxFrame->hcan,&TxFrame->Header,TxFrame->Data);
	 
	 }
	 
	 
}

/**
 * FDCAN FIFO0 接收回调
 * CAN1 (J3/J4): Master_ID 0x13/0x14 → DM_Data_t[2/3]
 * CAN3 (J1/J2): Master_ID 0x11/0x12 → DM_Data_t[0/1]
 */
void HAL_FDCAN_RxFifo0Callback(FDCAN_HandleTypeDef *hfdcan, uint32_t RxFifo0ITs)
{
    FDCAN_RxHeaderTypeDef rx_header;
    uint8_t rx_data[8];
    
    HAL_FDCAN_GetRxMessage(hfdcan, FDCAN_RX_FIFO0, &rx_header, rx_data);
    
    if (hfdcan->Instance == FDCAN3)
    {
        /* CAN3: J1/J2 → DM_Data_t[0/1] */
        switch(rx_header.Identifier)
        {
            case 0x11:  /* J1 */
            {
                DM_Data_t[0].State = rx_data[0] >> 4;
				DM_Data_t[0].P_int = (rx_data[1] << 8) | rx_data[2];
				DM_Data_t[0].V_int = (rx_data[3] << 4) |(rx_data[4] >> 4);
				DM_Data_t[0].T_int = ((rx_data[4]&0xF) << 8) | rx_data[5];
				DM_Data_t[0].Position = uint_to_float(DM_Data_t[0].P_int, -P_MAX, P_MAX, 16) + P_MAX;
				DM_Data_t[0].Velocity = uint_to_float(DM_Data_t[0].V_int, -V_MAX, V_MAX, 12);
				DM_Data_t[0].Torque = uint_to_float(DM_Data_t[0].T_int, -T_MAX, T_MAX, 12);
                break;
            }
            
            case 0x12:  /* J2 */
            {
                DM_Data_t[1].State = rx_data[0] >> 4;
				DM_Data_t[1].P_int = (rx_data[1] << 8) | rx_data[2];
				DM_Data_t[1].V_int = (rx_data[3] << 4) |(rx_data[4] >> 4);
				DM_Data_t[1].T_int = ((rx_data[4]&0xF) << 8) | rx_data[5];
				DM_Data_t[1].Position = uint_to_float(DM_Data_t[1].P_int, -P_MAX, P_MAX, 16) + P_MAX;
				DM_Data_t[1].Velocity = uint_to_float(DM_Data_t[1].V_int, -V_MAX, V_MAX, 12);
				DM_Data_t[1].Torque = uint_to_float(DM_Data_t[1].T_int, -T_MAX, T_MAX, 12);
                break;
            }
            
            default:
                break;
        }
    }
    else if (hfdcan->Instance == FDCAN1)
    {
        /* CAN1: J3/J4 → DM_Data_t[2/3] */
        switch(rx_header.Identifier)
        {
            case 0x13:  /* J3 */
            {
                DM_Data_t[2].State = rx_data[0] >> 4;
				DM_Data_t[2].P_int = (rx_data[1] << 8) | rx_data[2];
				DM_Data_t[2].V_int = (rx_data[3] << 4) |(rx_data[4] >> 4);
				DM_Data_t[2].T_int = ((rx_data[4]&0xF) << 8) | rx_data[5];
				DM_Data_t[2].Position = uint_to_float(DM_Data_t[2].P_int, -P_MAX, P_MAX, 16) + P_MAX;
				DM_Data_t[2].Velocity = uint_to_float(DM_Data_t[2].V_int, -V_MAX, V_MAX, 12);
				DM_Data_t[2].Torque = uint_to_float(DM_Data_t[2].T_int, -T_MAX, T_MAX, 12);
                break;
            }
            
            case 0x14:  /* J4 */
            {
                DM_Data_t[3].State = rx_data[0] >> 4;
				DM_Data_t[3].P_int = (rx_data[1] << 8) | rx_data[2];
				DM_Data_t[3].V_int = (rx_data[3] << 4) |(rx_data[4] >> 4);
				DM_Data_t[3].T_int = ((rx_data[4]&0xF) << 8) | rx_data[5];
				DM_Data_t[3].Position = uint_to_float(DM_Data_t[3].P_int, -P_MAX, P_MAX, 16) + P_MAX;
				DM_Data_t[3].Velocity = uint_to_float(DM_Data_t[3].V_int, -V_MAX, V_MAX, 12);
				DM_Data_t[3].Torque = uint_to_float(DM_Data_t[3].T_int, -T_MAX, T_MAX, 12);
                break;
            }

            default:
                break;
        }
    }
}

/**
 * FDCAN FIFO1 接收回调
 * CAN2 (J5/J6/J7): Master_ID 0x15/0x16/0x17 → DM_Data_t[4/5/6]
 */
void HAL_FDCAN_RxFifo1Callback(FDCAN_HandleTypeDef *hfdcan, uint32_t RxFifo1ITs)
{
    FDCAN_RxHeaderTypeDef rx_header;
    uint8_t rx_data[8];
    
    HAL_FDCAN_GetRxMessage(hfdcan, FDCAN_RX_FIFO1, &rx_header, rx_data);
    
    /* CAN2: J5/J6/J7 → DM_Data_t[4/5/6] */
    switch(rx_header.Identifier)
    {
        case 0x15:  /* J5 */
        {
            DM_Data_t[4].State = rx_data[0] >> 4;
			DM_Data_t[4].P_int = (rx_data[1] << 8) | rx_data[2];
			DM_Data_t[4].V_int = (rx_data[3] << 4) |(rx_data[4] >> 4);
			DM_Data_t[4].T_int = ((rx_data[4]&0xF) << 8) | rx_data[5];
			DM_Data_t[4].Position = uint_to_float(DM_Data_t[4].P_int, -P_MAX, P_MAX, 16) + P_MAX;
			DM_Data_t[4].Velocity = uint_to_float(DM_Data_t[4].V_int, -V_MAX, V_MAX, 12);
			DM_Data_t[4].Torque = uint_to_float(DM_Data_t[4].T_int, -T_MAX, T_MAX, 12);
            break;
        }
        
        case 0x16:  /* J6 */
        {
            DM_Data_t[5].State = rx_data[0] >> 4;
			DM_Data_t[5].P_int = (rx_data[1] << 8) | rx_data[2];
			DM_Data_t[5].V_int = (rx_data[3] << 4) |(rx_data[4] >> 4);
			DM_Data_t[5].T_int = ((rx_data[4]&0xF) << 8) | rx_data[5];
			DM_Data_t[5].Position = uint_to_float(DM_Data_t[5].P_int, -P_MAX, P_MAX, 16) + P_MAX;
			DM_Data_t[5].Velocity = uint_to_float(DM_Data_t[5].V_int, -V_MAX, V_MAX, 12);
			DM_Data_t[5].Torque = uint_to_float(DM_Data_t[5].T_int, -T_MAX, T_MAX, 12);
            break;
        }
        
        case 0x17:  /* J7 */
        {
            DM_Data_t[6].State = rx_data[0] >> 4;
			DM_Data_t[6].P_int = (rx_data[1] << 8) | rx_data[2];
			DM_Data_t[6].V_int = (rx_data[3] << 4) |(rx_data[4] >> 4);
			DM_Data_t[6].T_int = ((rx_data[4]&0xF) << 8) | rx_data[5];
			DM_Data_t[6].Position = uint_to_float(DM_Data_t[6].P_int, -P_MAX, P_MAX, 16) + P_MAX;
			DM_Data_t[6].Velocity = uint_to_float(DM_Data_t[6].V_int, -V_MAX, V_MAX, 12);
			DM_Data_t[6].Torque = uint_to_float(DM_Data_t[6].T_int, -T_MAX, T_MAX, 12);
            break;
        }
        
        default:
            break;
    }
}
