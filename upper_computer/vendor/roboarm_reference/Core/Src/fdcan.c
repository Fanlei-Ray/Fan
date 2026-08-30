/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file    fdcan.c
  * @brief   This file provides code for the configuration
  *          of the FDCAN instances.
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "fdcan.h"

/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

FDCAN_HandleTypeDef hfdcan1;
FDCAN_HandleTypeDef hfdcan2;
FDCAN_HandleTypeDef hfdcan3;

/* FDCAN1 init function */
void MX_FDCAN1_Init(void)
{

  /* USER CODE BEGIN FDCAN1_Init 0 */

  /* USER CODE END FDCAN1_Init 0 */

  /* USER CODE BEGIN FDCAN1_Init 1 */

  /* USER CODE END FDCAN1_Init 1 */
  hfdcan1.Instance = FDCAN1;
  hfdcan1.Init.FrameFormat = FDCAN_FRAME_CLASSIC;
  hfdcan1.Init.Mode = FDCAN_MODE_NORMAL;
  hfdcan1.Init.AutoRetransmission = DISABLE;
  hfdcan1.Init.TransmitPause = DISABLE;
  hfdcan1.Init.ProtocolException = DISABLE;
  hfdcan1.Init.NominalPrescaler = 3;
  hfdcan1.Init.NominalSyncJumpWidth = 10;
  hfdcan1.Init.NominalTimeSeg1 = 29;
  hfdcan1.Init.NominalTimeSeg2 = 10;
  hfdcan1.Init.DataPrescaler = 3;
  hfdcan1.Init.DataSyncJumpWidth = 10;
  hfdcan1.Init.DataTimeSeg1 = 29;
  hfdcan1.Init.DataTimeSeg2 = 10;
  hfdcan1.Init.MessageRAMOffset = 0;
  hfdcan1.Init.StdFiltersNbr = 1;
  hfdcan1.Init.ExtFiltersNbr = 0;
  hfdcan1.Init.RxFifo0ElmtsNbr = 4;
  hfdcan1.Init.RxFifo0ElmtSize = FDCAN_DATA_BYTES_8;
  hfdcan1.Init.RxFifo1ElmtsNbr = 0;
  hfdcan1.Init.RxFifo1ElmtSize = FDCAN_DATA_BYTES_8;
  hfdcan1.Init.RxBuffersNbr = 0;
  hfdcan1.Init.RxBufferSize = FDCAN_DATA_BYTES_8;
  hfdcan1.Init.TxEventsNbr = 0;
  hfdcan1.Init.TxBuffersNbr = 0;
  hfdcan1.Init.TxFifoQueueElmtsNbr = 16;
  hfdcan1.Init.TxFifoQueueMode = FDCAN_TX_FIFO_OPERATION;
  hfdcan1.Init.TxElmtSize = FDCAN_DATA_BYTES_8;
  if (HAL_FDCAN_Init(&hfdcan1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN FDCAN1_Init 2 */

  /* USER CODE END FDCAN1_Init 2 */

}
/* FDCAN2 init function */
void MX_FDCAN2_Init(void)
{

  /* USER CODE BEGIN FDCAN2_Init 0 */

  /* USER CODE END FDCAN2_Init 0 */

  /* USER CODE BEGIN FDCAN2_Init 1 */

  /* USER CODE END FDCAN2_Init 1 */
  hfdcan2.Instance = FDCAN2;
  hfdcan2.Init.FrameFormat = FDCAN_FRAME_CLASSIC;
  hfdcan2.Init.Mode = FDCAN_MODE_NORMAL;
  hfdcan2.Init.AutoRetransmission = DISABLE;
  hfdcan2.Init.TransmitPause = DISABLE;
  hfdcan2.Init.ProtocolException = DISABLE;
  hfdcan2.Init.NominalPrescaler = 3;
  hfdcan2.Init.NominalSyncJumpWidth = 10;
  hfdcan2.Init.NominalTimeSeg1 = 29;
  hfdcan2.Init.NominalTimeSeg2 = 10;
  hfdcan2.Init.DataPrescaler = 3;
  hfdcan2.Init.DataSyncJumpWidth = 10;
  hfdcan2.Init.DataTimeSeg1 = 29;
  hfdcan2.Init.DataTimeSeg2 = 10;
  hfdcan2.Init.MessageRAMOffset = 128;
  hfdcan2.Init.StdFiltersNbr = 1;
  hfdcan2.Init.ExtFiltersNbr = 0;
  hfdcan2.Init.RxFifo0ElmtsNbr = 0;
  hfdcan2.Init.RxFifo0ElmtSize = FDCAN_DATA_BYTES_8;
  hfdcan2.Init.RxFifo1ElmtsNbr = 4;
  hfdcan2.Init.RxFifo1ElmtSize = FDCAN_DATA_BYTES_8;
  hfdcan2.Init.RxBuffersNbr = 0;
  hfdcan2.Init.RxBufferSize = FDCAN_DATA_BYTES_8;
  hfdcan2.Init.TxEventsNbr = 0;
  hfdcan2.Init.TxBuffersNbr = 0;
  hfdcan2.Init.TxFifoQueueElmtsNbr = 16;
  hfdcan2.Init.TxFifoQueueMode = FDCAN_TX_FIFO_OPERATION;
  hfdcan2.Init.TxElmtSize = FDCAN_DATA_BYTES_8;
  if (HAL_FDCAN_Init(&hfdcan2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN FDCAN2_Init 2 */

  /* USER CODE END FDCAN2_Init 2 */

}
/* FDCAN3 init function */
void MX_FDCAN3_Init(void)
{

  /* USER CODE BEGIN FDCAN3_Init 0 */

  /* USER CODE END FDCAN3_Init 0 */

  /* USER CODE BEGIN FDCAN3_Init 1 */

  /* USER CODE END FDCAN3_Init 1 */
  hfdcan3.Instance = FDCAN3;
  hfdcan3.Init.FrameFormat = FDCAN_FRAME_CLASSIC;
  hfdcan3.Init.Mode = FDCAN_MODE_NORMAL;
  hfdcan3.Init.AutoRetransmission = DISABLE;
  hfdcan3.Init.TransmitPause = DISABLE;
  hfdcan3.Init.ProtocolException = DISABLE;
  hfdcan3.Init.NominalPrescaler = 3;
  hfdcan3.Init.NominalSyncJumpWidth = 10;
  hfdcan3.Init.NominalTimeSeg1 = 29;
  hfdcan3.Init.NominalTimeSeg2 = 10;
  hfdcan3.Init.DataPrescaler = 3;
  hfdcan3.Init.DataSyncJumpWidth = 10;
  hfdcan3.Init.DataTimeSeg1 = 29;
  hfdcan3.Init.DataTimeSeg2 = 10;
  hfdcan3.Init.MessageRAMOffset = 256;
  hfdcan3.Init.StdFiltersNbr = 1;
  hfdcan3.Init.ExtFiltersNbr = 0;
  hfdcan3.Init.RxFifo0ElmtsNbr = 4;
  hfdcan3.Init.RxFifo0ElmtSize = FDCAN_DATA_BYTES_8;
  hfdcan3.Init.RxFifo1ElmtsNbr = 0;
  hfdcan3.Init.RxFifo1ElmtSize = FDCAN_DATA_BYTES_8;
  hfdcan3.Init.RxBuffersNbr = 0;
  hfdcan3.Init.RxBufferSize = FDCAN_DATA_BYTES_8;
  hfdcan3.Init.TxEventsNbr = 0;
  hfdcan3.Init.TxBuffersNbr = 0;
  hfdcan3.Init.TxFifoQueueElmtsNbr = 16;
  hfdcan3.Init.TxFifoQueueMode = FDCAN_TX_FIFO_OPERATION;
  hfdcan3.Init.TxElmtSize = FDCAN_DATA_BYTES_8;
  if (HAL_FDCAN_Init(&hfdcan3) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN FDCAN3_Init 2 */

  /* USER CODE END FDCAN3_Init 2 */

}

static uint32_t HAL_RCC_FDCAN_CLK_ENABLED=0;

void HAL_FDCAN_MspInit(FDCAN_HandleTypeDef* fdcanHandle)
{

  GPIO_InitTypeDef GPIO_InitStruct = {0};
  RCC_PeriphCLKInitTypeDef PeriphClkInitStruct = {0};
  if(fdcanHandle->Instance==FDCAN1)
  {
  /* USER CODE BEGIN FDCAN1_MspInit 0 */

  /* USER CODE END FDCAN1_MspInit 0 */

  /** Initializes the peripherals clock
  */
    PeriphClkInitStruct.PeriphClockSelection = RCC_PERIPHCLK_FDCAN;
    PeriphClkInitStruct.FdcanClockSelection = RCC_FDCANCLKSOURCE_PLL;
    if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInitStruct) != HAL_OK)
    {
      Error_Handler();
    }

    /* FDCAN1 clock enable */
    HAL_RCC_FDCAN_CLK_ENABLED++;
    if(HAL_RCC_FDCAN_CLK_ENABLED==1){
      __HAL_RCC_FDCAN_CLK_ENABLE();
    }

    __HAL_RCC_GPIOD_CLK_ENABLE();
    /**FDCAN1 GPIO Configuration
    PD0     ------> FDCAN1_RX
    PD1     ------> FDCAN1_TX
    */
    GPIO_InitStruct.Pin = GPIO_PIN_0|GPIO_PIN_1;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    GPIO_InitStruct.Alternate = GPIO_AF9_FDCAN1;
    HAL_GPIO_Init(GPIOD, &GPIO_InitStruct);

    /* FDCAN1 interrupt Init */
    HAL_NVIC_SetPriority(FDCAN1_IT0_IRQn, 5, 0);
    HAL_NVIC_EnableIRQ(FDCAN1_IT0_IRQn);
  /* USER CODE BEGIN FDCAN1_MspInit 1 */

  /* USER CODE END FDCAN1_MspInit 1 */
  }
  else if(fdcanHandle->Instance==FDCAN2)
  {
  /* USER CODE BEGIN FDCAN2_MspInit 0 */

  /* USER CODE END FDCAN2_MspInit 0 */

  /** Initializes the peripherals clock
  */
    PeriphClkInitStruct.PeriphClockSelection = RCC_PERIPHCLK_FDCAN;
    PeriphClkInitStruct.FdcanClockSelection = RCC_FDCANCLKSOURCE_PLL;
    if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInitStruct) != HAL_OK)
    {
      Error_Handler();
    }

    /* FDCAN2 clock enable */
    HAL_RCC_FDCAN_CLK_ENABLED++;
    if(HAL_RCC_FDCAN_CLK_ENABLED==1){
      __HAL_RCC_FDCAN_CLK_ENABLE();
    }

    __HAL_RCC_GPIOB_CLK_ENABLE();
    /**FDCAN2 GPIO Configuration
    PB5     ------> FDCAN2_RX
    PB6     ------> FDCAN2_TX
    */
    GPIO_InitStruct.Pin = GPIO_PIN_5|GPIO_PIN_6;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    GPIO_InitStruct.Alternate = GPIO_AF9_FDCAN2;
    HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

    /* FDCAN2 interrupt Init */
    HAL_NVIC_SetPriority(FDCAN2_IT1_IRQn, 5, 0);
    HAL_NVIC_EnableIRQ(FDCAN2_IT1_IRQn);
  /* USER CODE BEGIN FDCAN2_MspInit 1 */

  /* USER CODE END FDCAN2_MspInit 1 */
  }
  else if(fdcanHandle->Instance==FDCAN3)
  {
  /* USER CODE BEGIN FDCAN3_MspInit 0 */

  /* USER CODE END FDCAN3_MspInit 0 */

  /** Initializes the peripherals clock
  */
    PeriphClkInitStruct.PeriphClockSelection = RCC_PERIPHCLK_FDCAN;
    PeriphClkInitStruct.FdcanClockSelection = RCC_FDCANCLKSOURCE_PLL;
    if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInitStruct) != HAL_OK)
    {
      Error_Handler();
    }

    /* FDCAN3 clock enable */
    HAL_RCC_FDCAN_CLK_ENABLED++;
    if(HAL_RCC_FDCAN_CLK_ENABLED==1){
      __HAL_RCC_FDCAN_CLK_ENABLE();
    }

    __HAL_RCC_GPIOD_CLK_ENABLE();
    /**FDCAN3 GPIO Configuration
    PD12     ------> FDCAN3_RX
    PD13     ------> FDCAN3_TX
    */
    GPIO_InitStruct.Pin = GPIO_PIN_12|GPIO_PIN_13;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    GPIO_InitStruct.Alternate = GPIO_AF5_FDCAN3;
    HAL_GPIO_Init(GPIOD, &GPIO_InitStruct);

    /* FDCAN3 interrupt Init */
    HAL_NVIC_SetPriority(FDCAN3_IT0_IRQn, 5, 0);
    HAL_NVIC_EnableIRQ(FDCAN3_IT0_IRQn);
  /* USER CODE BEGIN FDCAN3_MspInit 1 */

  /* USER CODE END FDCAN3_MspInit 1 */
  }
}

void HAL_FDCAN_MspDeInit(FDCAN_HandleTypeDef* fdcanHandle)
{

  if(fdcanHandle->Instance==FDCAN1)
  {
  /* USER CODE BEGIN FDCAN1_MspDeInit 0 */

  /* USER CODE END FDCAN1_MspDeInit 0 */
    /* Peripheral clock disable */
    HAL_RCC_FDCAN_CLK_ENABLED--;
    if(HAL_RCC_FDCAN_CLK_ENABLED==0){
      __HAL_RCC_FDCAN_CLK_DISABLE();
    }

    /**FDCAN1 GPIO Configuration
    PD0     ------> FDCAN1_RX
    PD1     ------> FDCAN1_TX
    */
    HAL_GPIO_DeInit(GPIOD, GPIO_PIN_0|GPIO_PIN_1);

    /* FDCAN1 interrupt Deinit */
    HAL_NVIC_DisableIRQ(FDCAN1_IT0_IRQn);
  /* USER CODE BEGIN FDCAN1_MspDeInit 1 */

  /* USER CODE END FDCAN1_MspDeInit 1 */
  }
  else if(fdcanHandle->Instance==FDCAN2)
  {
  /* USER CODE BEGIN FDCAN2_MspDeInit 0 */

  /* USER CODE END FDCAN2_MspDeInit 0 */
    /* Peripheral clock disable */
    HAL_RCC_FDCAN_CLK_ENABLED--;
    if(HAL_RCC_FDCAN_CLK_ENABLED==0){
      __HAL_RCC_FDCAN_CLK_DISABLE();
    }

    /**FDCAN2 GPIO Configuration
    PB5     ------> FDCAN2_RX
    PB6     ------> FDCAN2_TX
    */
    HAL_GPIO_DeInit(GPIOB, GPIO_PIN_5|GPIO_PIN_6);

    /* FDCAN2 interrupt Deinit */
    HAL_NVIC_DisableIRQ(FDCAN2_IT1_IRQn);
  /* USER CODE BEGIN FDCAN2_MspDeInit 1 */

  /* USER CODE END FDCAN2_MspDeInit 1 */
  }
  else if(fdcanHandle->Instance==FDCAN3)
  {
  /* USER CODE BEGIN FDCAN3_MspDeInit 0 */

  /* USER CODE END FDCAN3_MspDeInit 0 */
    /* Peripheral clock disable */
    HAL_RCC_FDCAN_CLK_ENABLED--;
    if(HAL_RCC_FDCAN_CLK_ENABLED==0){
      __HAL_RCC_FDCAN_CLK_DISABLE();
    }

    /**FDCAN3 GPIO Configuration
    PD12     ------> FDCAN3_RX
    PD13     ------> FDCAN3_TX
    */
    HAL_GPIO_DeInit(GPIOD, GPIO_PIN_12|GPIO_PIN_13);

    /* FDCAN3 interrupt Deinit */
    HAL_NVIC_DisableIRQ(FDCAN3_IT0_IRQn);
  /* USER CODE BEGIN FDCAN3_MspDeInit 1 */

  /* USER CODE END FDCAN3_MspDeInit 1 */
  }
}

/* USER CODE BEGIN 1 */
FDCAN_TxFrame_TypeDef FDCAN1TxFrame = 
{
	.hcan = &hfdcan1,
	.Header.IdType = FDCAN_STANDARD_ID, 
	.Header.TxFrameType = FDCAN_DATA_FRAME,
	.Header.DataLength = 8,
	.Header.ErrorStateIndicator =  FDCAN_ESI_ACTIVE,
	.Header.BitRateSwitch = FDCAN_BRS_OFF,
	.Header.FDFormat =  FDCAN_CLASSIC_CAN,           
	.Header.TxEventFifoControl =  FDCAN_NO_TX_EVENTS,  
	.Header.MessageMarker = 0,
};

FDCAN_TxFrame_TypeDef FDCAN2TxFrame = 
{
	.hcan = &hfdcan2,
	.Header.IdType = FDCAN_STANDARD_ID, 
	.Header.TxFrameType = FDCAN_DATA_FRAME,
	.Header.DataLength = 8,
	.Header.ErrorStateIndicator =  FDCAN_ESI_ACTIVE,
	.Header.BitRateSwitch = FDCAN_BRS_OFF,
	.Header.FDFormat =  FDCAN_CLASSIC_CAN,           
	.Header.TxEventFifoControl =  FDCAN_NO_TX_EVENTS,  
	.Header.MessageMarker = 0,
};

FDCAN_TxFrame_TypeDef FDCAN3TxFrame = 
{
	.hcan = &hfdcan3,
	.Header.IdType = FDCAN_STANDARD_ID, 
	.Header.TxFrameType = FDCAN_DATA_FRAME,
	.Header.DataLength = 8,
	.Header.ErrorStateIndicator =  FDCAN_ESI_ACTIVE,
	.Header.BitRateSwitch = FDCAN_BRS_OFF,
	.Header.FDFormat =  FDCAN_CLASSIC_CAN,           
	.Header.TxEventFifoControl =  FDCAN_NO_TX_EVENTS,  
	.Header.MessageMarker = 0,
};

void BSP_FDCAN1_Init(void)
{

	FDCAN_FilterTypeDef FDCAN1_FilterConfig;
		
	FDCAN1_FilterConfig.IdType = FDCAN_STANDARD_ID; // ���˱�׼ID������CANֻ�б�׼ID
	FDCAN1_FilterConfig.FilterIndex = 0;           //��������ţ��ü�·CAN����������0��1��2....
	FDCAN1_FilterConfig.FilterType = FDCAN_FILTER_MASK; //������Maskģʽ �غ�������ID1��ID2������
	FDCAN1_FilterConfig.FilterConfig = FDCAN_FILTER_TO_RXFIFO0;//ѡ���ĸ�FIFO�����գ�������CubeMX����������FIFO1�͸ĳ�FDCAN_FILTER_TO_RXFIFO1
	FDCAN1_FilterConfig.FilterID1 = 0x00000000; // ������У�ֻҪID2����0x00000000�Ͳ�����˵��κ�ID
	FDCAN1_FilterConfig.FilterID2 = 0x00000000; //��������
	  
	HAL_FDCAN_ConfigFilter(&hfdcan1, &FDCAN1_FilterConfig); //���������õ�CAN1
			
	HAL_FDCAN_ConfigGlobalFilter(&hfdcan1, FDCAN_REJECT, FDCAN_REJECT, FDCAN_FILTER_REMOTE, FDCAN_FILTER_REMOTE); //����CAN1��ȫ�ֹ��ˣ����ǿ���������
	 
	HAL_FDCAN_ActivateNotification(&hfdcan1, FDCAN_IT_RX_FIFO0_NEW_MESSAGE, 0);//��FIFO0���������ݽ����жϣ�
	  
	HAL_FDCAN_Start(&hfdcan1);//ʹ��CAN1
 	
	
}

void BSP_FDCAN2_Init(void)
{

	FDCAN_FilterTypeDef FDCAN2_FilterConfig;
		
	FDCAN2_FilterConfig.IdType = FDCAN_STANDARD_ID; // 过滤标准ID，经典CAN只有标准ID
	FDCAN2_FilterConfig.FilterIndex = 0;           // 每个FDCAN实例的过滤器独立编号，从0开始
	FDCAN2_FilterConfig.FilterType = FDCAN_FILTER_MASK; //过滤器Mask模式 关乎到下面ID1、ID2的配置
	FDCAN2_FilterConfig.FilterConfig = FDCAN_FILTER_TO_RXFIFO1;//选择哪个FIFO区接收，根据你CubeMX的配置来，FIFO1就改成FDCAN_FILTER_TO_RXFIFO1
	FDCAN2_FilterConfig.FilterID1 = 0x00000000; // 这个都行，只要ID2配置0x00000000就不会过滤调任何ID
	FDCAN2_FilterConfig.FilterID2 = 0x00000000; //理由如上
	  
	HAL_FDCAN_ConfigFilter(&hfdcan2, &FDCAN2_FilterConfig); //将上述配置到CAN2
			
	HAL_FDCAN_ConfigGlobalFilter(&hfdcan2, FDCAN_REJECT, FDCAN_REJECT, FDCAN_FILTER_REMOTE, FDCAN_FILTER_REMOTE); //开启CAN2的全局过滤，就是开启过滤器
	 
	/* FIFO1 notifications must be routed to the IRQ line enabled for FDCAN2. */
	if (HAL_FDCAN_ConfigInterruptLines(&hfdcan2,
	                                   FDCAN_IT_RX_FIFO1_NEW_MESSAGE,
	                                   FDCAN_INTERRUPT_LINE1) != HAL_OK)
	{
	  Error_Handler();
	}

	HAL_FDCAN_ActivateNotification(&hfdcan2, FDCAN_IT_RX_FIFO1_NEW_MESSAGE, 0);//打开FIFO0区的新数据接收中断，
	  
	HAL_FDCAN_Start(&hfdcan2);//使能CAN2
 	
 	
}

void BSP_FDCAN3_Init(void)
{

	FDCAN_FilterTypeDef FDCAN3_FilterConfig;
		
	FDCAN3_FilterConfig.IdType = FDCAN_STANDARD_ID; // 过滤标准ID，经典CAN只有标准ID
	FDCAN3_FilterConfig.FilterIndex = 0;           // 每个FDCAN实例的过滤器独立编号，从0开始
	FDCAN3_FilterConfig.FilterType = FDCAN_FILTER_MASK; //过滤器Mask模式 关乎到下面ID1、ID2的配置
	FDCAN3_FilterConfig.FilterConfig = FDCAN_FILTER_TO_RXFIFO0;//选择哪个FIFO区接收
	FDCAN3_FilterConfig.FilterID1 = 0x00000000; // 这个都行，只要ID2配置0x00000000就不会过滤调任何ID
	FDCAN3_FilterConfig.FilterID2 = 0x00000000; //理由如上
	  
	HAL_FDCAN_ConfigFilter(&hfdcan3, &FDCAN3_FilterConfig); //将上述配置到CAN3
			
	HAL_FDCAN_ConfigGlobalFilter(&hfdcan3, FDCAN_REJECT, FDCAN_REJECT, FDCAN_FILTER_REMOTE, FDCAN_FILTER_REMOTE); //开启CAN3的全局过滤，就是开启过滤器
	 
	HAL_FDCAN_ActivateNotification(&hfdcan3, FDCAN_IT_RX_FIFO0_NEW_MESSAGE, 0);//打开FIFO0区的新数据接收中断
	  
	HAL_FDCAN_Start(&hfdcan3);//使能CAN3
 	
 	
}
/* USER CODE END 1 */

