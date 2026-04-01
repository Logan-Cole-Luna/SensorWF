/**
 * stm32h7_can_ids_benchmark.h
 */
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Call once from main() before osKernelStart().
 * Creates the benchmark FreeRTOS task (static allocation, no heap).
 */
void ids_benchmark_task_init(void);

#ifdef __cplusplus
}
#endif
