/**
 * stm32f373_can_ids_benchmark.h  —  bare-metal, no FreeRTOS
 * Target: STM32F373C8T  (Cortex-M4F, 72 MHz, 64 KB Flash, 16 KB RAM)
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
 *       SystemClock_Config();   // configure PLL to 72 MHz
 *       MX_USART2_UART_Init();  // USART2 on PA2/PA3
 *       ids_benchmark_run();    // never returns
 *   }
 */
void ids_benchmark_run(void);

#ifdef __cplusplus
}
#endif
