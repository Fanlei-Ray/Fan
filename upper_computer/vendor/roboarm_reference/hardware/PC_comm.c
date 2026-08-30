#include "PC_comm.h"
#include "Motor.h"

// 根据您的实际底层配置，声明串口10和DMA的句柄
extern UART_HandleTypeDef huart10;
extern DMA_HandleTypeDef hdma_usart10_rx;

extern DM_Motor_Data_Typedef DM_Data_t[7];

PC_Data_t pc_data;
uint8_t pc_rx_buffer[PC_BUFLENGTH];
uint8_t pc_send_buf[PC_SEND_SIZE];

/**
  * @brief  上位机通信初始化
  * @param  None
  * @retval None
  */
void PC_Comm_Init(void)
{
    __HAL_UART_ENABLE_IT(&huart10, UART_IT_IDLE); // 使能串口10空闲中断
    HAL_UART_Receive_DMA(&huart10, pc_rx_buffer, PC_BUFLENGTH); // 打开串口DMA接收
}

/**
  * @brief  串口10接收空闲中断处理函数
  * @param  None
  * @retval None
  */
void USART10_Receive_IDLE(void)
{
    uint8_t data_length;
    
    HAL_UART_DMAStop(&huart10); // 停止DMA接收以便计算数据长度
    data_length = PC_BUFLENGTH - __HAL_DMA_GET_COUNTER(&hdma_usart10_rx);
    
    // 校验帧头 'V' 'G'
    if(pc_rx_buffer[0] == 'V' && pc_rx_buffer[1] == 'G')
    {
        memcpy(pc_data.real_receive, pc_rx_buffer, data_length); // 拷贝实际接收数据
        
        // 校验帧尾 'N' 'E'
        if(pc_data.real_receive[data_length-1] == 'E' && pc_data.real_receive[data_length-2] == 'N')
        {
            // 按字节顺序依次提取7个参数
            memcpy(&pc_data.State,    &pc_data.real_receive[2],  sizeof(int16_t));
            memcpy(&pc_data.P_int,    &pc_data.real_receive[4],  sizeof(uint16_t));
            memcpy(&pc_data.V_int,    &pc_data.real_receive[6],  sizeof(uint16_t));
            memcpy(&pc_data.T_int,    &pc_data.real_receive[8],  sizeof(uint16_t));
            memcpy(&pc_data.Position, &pc_data.real_receive[10], sizeof(float));
            memcpy(&pc_data.Velocity, &pc_data.real_receive[14], sizeof(float));
            memcpy(&pc_data.Torque,   &pc_data.real_receive[18], sizeof(float));
        }
    }
    
    memset(pc_rx_buffer, 0, PC_BUFLENGTH); // 清空接收缓冲区，准备下一次接收
    HAL_UART_Receive_DMA(&huart10, pc_rx_buffer, PC_BUFLENGTH); // 重新开启DMA接收
}

/**
  * @brief  将参数回传给上位机
  * @param  None
  * @retval None
  */
void PC_Data_Send(void)
{
    // 装载帧头 'G' 'V'
    pc_send_buf[0] = 'G';
    pc_send_buf[1] = 'V';
    
    // 按字节顺序依次拷贝7个参数到发送缓冲区
    memcpy(&pc_send_buf[2],  (void *)&DM_Data_t[0].State,             sizeof(int16_t));
    memcpy(&pc_send_buf[4],  (void *)&DM_Data_t[0].P_int,             sizeof(uint16_t));
    memcpy(&pc_send_buf[6],  (void *)&DM_Data_t[0].V_int,             sizeof(uint16_t));
    memcpy(&pc_send_buf[8],  (void *)&DM_Data_t[0].T_int,             sizeof(uint16_t));
    memcpy(&pc_send_buf[10], (void *)&DM_Data_t[0].Position,          sizeof(float));
    memcpy(&pc_send_buf[14], (void *)&DM_Data_t[0].Velocity,          sizeof(float));
    memcpy(&pc_send_buf[18], (void *)&DM_Data_t[0].Torque,            sizeof(float));
    
    // 装载帧尾 'E' 'N'
    pc_send_buf[22] = 'E';
    pc_send_buf[23] = 'N';

    // 触发DMA发送
    HAL_UART_Transmit_DMA(&huart10, pc_send_buf, PC_SEND_SIZE);
}
