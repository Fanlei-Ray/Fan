#ifndef __JUSTFLOAT_H
#define __JUSTFLOAT_H

#include "main.h"
#include "gpio.h"
#include <math.h>
#include "string.h"
#include "usart.h"
#include <stdint.h>

#define CHANNAL_MAX (20U)
#define JUSTFLOAT_TAIL_SIZE (4U)
#define JUSTFLOAT_FRAME_MAX_SIZE ((CHANNAL_MAX * sizeof(float)) + JUSTFLOAT_TAIL_SIZE)

extern uint8_t rx_data;
extern uint8_t movemode;

typedef struct
{
  uint8_t L1, L2, L3, L4, L5, L6;
  uint8_t S1, S2, S3, S4, S5, S6;
} LocationData;

void JustFloat(const float *dat, uint8_t channel_count);

extern float Justfloat_Buffer[CHANNAL_MAX];

#endif
