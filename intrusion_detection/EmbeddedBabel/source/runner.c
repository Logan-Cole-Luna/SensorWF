#include "runner.h"
#include "stm32h7_can_ids_float32.h"
#include "stm32h7_can_ids_scaler.h"

void nn_runner(void)
{
    static float features[IDS_N_FEATURES] = {0};
    ids_scale_features(features);
    (void)ids_predict(features);
}

void HAL_CAN_RxFifo0MsgPendingCallback(CAN_HandleTypeDef *hcan)
{
    CAN_RxHeaderTypeDef hdr;
    uint8_t data[8];

    if (HAL_CAN_GetRxMessage(hcan, CAN_RX_FIFO0, &hdr, data) == HAL_OK)
    {
    	(void)hdr;
        (void)data;

    }
}


