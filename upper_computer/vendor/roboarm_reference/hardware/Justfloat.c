#include "justfloat.h"

#define JUSTFLOAT_UART (huart7)

uint8_t rx_data;
uint8_t movemode;

float Justfloat_Buffer[CHANNAL_MAX] = {0.0f};

/* JustFloat: [N 个小端 float][00 00 80 7F]。 */
void JustFloat(const float *dat, uint8_t channel_count)
{
    static const uint8_t tail[JUSTFLOAT_TAIL_SIZE] = {
        0x00U, 0x00U, 0x80U, 0x7FU
    };
    uint8_t frame[JUSTFLOAT_FRAME_MAX_SIZE];
    uint16_t payload_size;

    if ((dat == NULL) || (channel_count == 0U))
    {
        return;
    }

    if (channel_count > CHANNAL_MAX)
    {
        channel_count = CHANNAL_MAX;
    }

    payload_size = (uint16_t)(sizeof(float) * channel_count);
    memcpy(frame, dat, payload_size);
    memcpy(&frame[payload_size], tail, JUSTFLOAT_TAIL_SIZE);

    HAL_UART_Transmit(&JUSTFLOAT_UART, frame,
                      (uint16_t)(payload_size + JUSTFLOAT_TAIL_SIZE), HAL_MAX_DELAY);
}
