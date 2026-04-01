/**
 * stm32h7_can_ids_benchmark.h  —  bare-metal, no FreeRTOS
 */
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Call from main() after HAL_Init() and SystemClock_Config().
 * Blocks forever (benchmark loop). Does not return.
 * No osKernelStart() or FreeRTOS required.
 *
 * Minimal main():
 *
 *   int main(void) {
 *       HAL_Init();
 *       SystemClock_Config();
 *       MX_USART3_UART_Init();
 *       ids_benchmark_run();   // never returns
 *   }
 */
void ids_benchmark_run(void);

#ifdef __cplusplus
}
#endif
