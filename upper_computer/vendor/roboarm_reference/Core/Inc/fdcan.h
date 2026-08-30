/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file    fdcan.h
  * @brief   This file contains all the function prototypes for
  *          the fdcan.c file
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
/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __FDCAN_H__
#define __FDCAN_H__

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

extern FDCAN_HandleTypeDef hfdcan1;

extern FDCAN_HandleTypeDef hfdcan2;

extern FDCAN_HandleTypeDef hfdcan3;

/* USER CODE BEGIN Private defines */
typedef struct 
{
	FDCAN_HandleTypeDef *hcan;
	
    FDCAN_TxHeaderTypeDef Header;
	
    uint8_t	Data[8];
}FDCAN_TxFrame_TypeDef;

extern  FDCAN_TxFrame_TypeDef   FDCAN1TxFrame;
extern  FDCAN_TxFrame_TypeDef   FDCAN2TxFrame;
extern  FDCAN_TxFrame_TypeDef   FDCAN3TxFrame;
/* USER CODE END Private defines */

void MX_FDCAN1_Init(void);
void MX_FDCAN2_Init(void);
void MX_FDCAN3_Init(void);

/* USER CODE BEGIN Prototypes */
void BSP_FDCAN1_Init(void);
void BSP_FDCAN2_Init(void);
void BSP_FDCAN3_Init(void);

/* USER CODE END Prototypes */

#ifdef __cplusplus
}
#endif

#endif /* __FDCAN_H__ */

