#ifndef _PC_COMM_H_
#define _PC_COMM_H_

#include "string.h"
#include "main.h"

// 缓冲区及收发长度定义
#define PC_BUFLENGTH     128 // 最大接收数据长度
#define PC_DATALENGTH    24  // 接收有效数据长度 (2字节帧头 + 20字节数据 + 2字节帧尾)
#define PC_SEND_SIZE     24  // 发送数据长度

// 上位机通信数据结构体
typedef struct
{
    uint8_t real_receive[PC_BUFLENGTH]; // 实际接收数据缓存区
    int16_t  State;
    uint16_t P_int;
    uint16_t V_int;
    uint16_t T_int;
    float    Position;
    float    Velocity;
    float    Torque;
} PC_Data_t;

extern PC_Data_t pc_data;
extern uint8_t pc_send_buf[PC_SEND_SIZE];

void PC_Comm_Init(void);
void USART10_Receive_IDLE(void);
void PC_Data_Send(void);

#endif
