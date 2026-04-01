/**
 * stm32h7_can_ids_benchmark.c
 * ============================
 * CAN IDS inference benchmark task for STM32H7.
 *
 * What this measures (IDS model only, isolated from normal operations):
 *   - Cycles per inference  (DWT cycle counter, 1-cycle resolution)
 *   - Wall-clock time per inference (derived from HCLK)
 *   - Stack high-water mark (FreeRTOS)
 *   - Heap used by this task (FreeRTOS heap4)
 *   - Estimated energy per inference (mJ) from datasheet current + Vdd
 *   - ML metrics: accuracy, recall, precision, F1, FPR
 *     (labels fed in from host over UART alongside features)
 *
 * Isolation strategy:
 *   The benchmark task calls vTaskSuspendAll() around every single
 *   inference so no scheduler tick or ISR can inflate the cycle count.
 *   Normal CAN Rx, telemetry, and ADCS tasks are suspended for the
 *   duration of a measurement burst, then resumed.
 *
 * Protocol (UART, 115200 8N1, host drives):
 *   Host  →  MCU :  'S'<n_samples:uint16_le>
 *   For each sample:
 *     Host  →  MCU :  15× float32 (60 bytes, raw features PRE-scaling)
 *                     + 1× uint8  (ground-truth label: 0=normal, 1=attack)
 *   MCU   →  Host :  per-sample record (see ids_sample_result_t, 16 bytes)
 *   MCU   →  Host :  final summary (see ids_bench_summary_t, 64 bytes)
 *
 * Build:
 *   Add this file to your STM32CubeIDE project alongside the three headers.
 *   Call ids_benchmark_task_init() from main() before osKernelStart().
 *   Enable DWT in SystemInit or before this task starts.
 *
 * Headers required (all auto-generated, copy to same folder):
 *   stm32h7_can_ids_float32.h   — tree arrays + ids_predict()
 *   stm32h7_can_ids_scaler.h    — scaler params + ids_scale_features()
 */

#include "stm32h7_can_ids_benchmark.h"

#include "stm32h7_can_ids_float32.h"   /* ids_predict(), IDS_N_FEATURES */
#include "stm32h7_can_ids_scaler.h"    /* ids_scale_features()           */

#include "FreeRTOS.h"
#include "task.h"
#include "cmsis_os.h"

#include "usart.h"          /* huart3 or whichever UART is wired to debug */
#include <string.h>
#include <stdio.h>
#include <math.h>

/* ── Configuration ─────────────────────────────────────────────────────────── */

#define IDS_UART            huart3          /* change to match your board       */
#define IDS_UART_TIMEOUT_MS 5000
#define IDS_TASK_STACK_SIZE 1024            /* words (4 KB) — tree + locals     */
#define IDS_TASK_PRIORITY   osPriorityLow   /* lower than all mission tasks      */
#define IDS_MAX_SAMPLES     8192

/* STM32H7 at 480 MHz, active-run current from datasheet Table 26 (~120 mA)    */
/* Adjust if your board runs at a different HCLK or Vdd                         */
#define IDS_HCLK_HZ         480000000UL
#define IDS_VDD_MV          3300U           /* mV                                */
#define IDS_RUN_CURRENT_UA  120000U         /* µA, whole-chip active run current */

/* ── DWT helpers ───────────────────────────────────────────────────────────── */

static inline void dwt_enable(void) {
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT  = 0;
    DWT->CTRL   |= DWT_CTRL_CYCCNTENA_Msk;
}

static inline uint32_t dwt_cycles(void) {
    return DWT->CYCCNT;
}

/* ── Wire types ─────────────────────────────────────────────────────────────── */

/* Sent to host after every inference */
typedef struct __attribute__((packed)) {
    uint8_t  prediction;      /* 0 or 1              */
    uint8_t  ground_truth;    /* 0 or 1 (echo back)  */
    uint32_t cycles;          /* DWT cycles           */
    uint32_t stack_hwm_words; /* FreeRTOS watermark   */
    uint32_t heap_free_bytes; /* xPortGetFreeHeapSize */
    uint32_t reserved;
} ids_sample_result_t;        /* 16 bytes             */

/* Sent once at the end */
typedef struct __attribute__((packed)) {
    uint32_t n_samples;
    uint32_t n_correct;       /* TP + TN              */
    uint32_t tp, tn, fp, fn;
    uint32_t cycles_min;
    uint32_t cycles_max;
    uint32_t cycles_sum_hi;   /* high 32 bits of 64-bit sum */
    uint32_t cycles_sum_lo;
    uint32_t stack_hwm_min_words;
    float    energy_per_inf_nj; /* nanojoules           */
    float    inference_us_mean;
    float    accuracy;
    float    precision;
    float    recall;
    float    f1;
    float    fpr;
} ids_bench_summary_t;        /* 64 bytes             */

/* ── Task state ─────────────────────────────────────────────────────────────── */

static TaskHandle_t s_bench_task_handle = NULL;
static StaticTask_t s_bench_task_tcb;
static StackType_t  s_bench_task_stack[IDS_TASK_STACK_SIZE];

/* ── UART helpers ───────────────────────────────────────────────────────────── */

static HAL_StatusTypeDef uart_recv(void *buf, uint16_t len) {
    return HAL_UART_Receive(&IDS_UART, (uint8_t *)buf, len, IDS_UART_TIMEOUT_MS);
}

static HAL_StatusTypeDef uart_send(const void *buf, uint16_t len) {
    return HAL_UART_Transmit(&IDS_UART, (const uint8_t *)buf, len, IDS_UART_TIMEOUT_MS);
}

static void uart_log(const char *msg) {
    uart_send(msg, (uint16_t)strlen(msg));
}

/* ── Benchmark task ─────────────────────────────────────────────────────────── */

static void ids_benchmark_task(void *arg) {
    (void)arg;

    char logbuf[128];
    dwt_enable();
    uart_log("[IDS] Benchmark task ready. Waiting for host...\r\n");

    for (;;) {
        /* ── Wait for start command ──────────────────────────────────── */
        uint8_t cmd = 0;
        while (cmd != 'S') {
            uart_recv(&cmd, 1);
        }

        uint16_t n_samples = 0;
        if (uart_recv(&n_samples, 2) != HAL_OK || n_samples == 0 ||
            n_samples > IDS_MAX_SAMPLES) {
            uart_log("[IDS] Bad sample count\r\n");
            continue;
        }

        snprintf(logbuf, sizeof(logbuf),
                 "[IDS] Starting %u-sample benchmark\r\n", n_samples);
        uart_log(logbuf);

        /* ── Suspend non-IDS tasks for clean power/cycle measurement ─── */
        vTaskSuspendAll();

        /* ── Per-run accumulators ─────────────────────────────────────── */
        uint32_t tp = 0, tn = 0, fp = 0, fn = 0;
        uint32_t cycles_min = UINT32_MAX, cycles_max = 0;
        uint64_t cycles_sum = 0;
        uint32_t hwm_min    = UINT32_MAX;

        for (uint16_t i = 0; i < n_samples; i++) {
            /* Receive raw feature vector (15 × float32) + ground-truth label */
            float    features[IDS_N_FEATURES];
            uint8_t  gt_label = 0;

            /* Re-enable scheduler ONLY for UART Rx (DMA/ISR driven) */
            xTaskResumeAll();
            HAL_StatusTypeDef rx_status = uart_recv(features, sizeof(features));
            if (rx_status == HAL_OK) rx_status = uart_recv(&gt_label, 1);
            vTaskSuspendAll();

            if (rx_status != HAL_OK) {
                uart_log("[IDS] UART Rx error\r\n");
                break;
            }

            /* ── Scale features (in-place) ─────────────────────────── */
            ids_scale_features(features);

            /* ── Isolated inference: DWT start ─────────────────────── */
            uint32_t t0 = dwt_cycles();
            int prediction = ids_predict(features);
            uint32_t elapsed = dwt_cycles() - t0;

            /* ── Metrics ────────────────────────────────────────────── */
            if (elapsed < cycles_min) cycles_min = elapsed;
            if (elapsed > cycles_max) cycles_max = elapsed;
            cycles_sum += elapsed;

            uint32_t hwm = uxTaskGetStackHighWaterMark(NULL);
            if (hwm < hwm_min) hwm_min = hwm;

            /* Confusion matrix */
            if      (prediction == 1 && gt_label == 1) tp++;
            else if (prediction == 0 && gt_label == 0) tn++;
            else if (prediction == 1 && gt_label == 0) fp++;
            else                                        fn++;

            /* ── Send per-sample result ──────────────────────────────── */
            ids_sample_result_t result = {
                .prediction      = (uint8_t)prediction,
                .ground_truth    = gt_label,
                .cycles          = elapsed,
                .stack_hwm_words = hwm,
                .heap_free_bytes = xPortGetFreeHeapSize(),
                .reserved        = 0,
            };
            xTaskResumeAll();
            uart_send(&result, sizeof(result));
            vTaskSuspendAll();
        }

        xTaskResumeAll();

        /* ── Compute summary ──────────────────────────────────────────── */
        uint32_t n_correct = tp + tn;
        float accuracy  = (float)n_correct / n_samples;
        float precision = (tp + fp) > 0 ? (float)tp / (tp + fp) : 0.0f;
        float recall    = (tp + fn) > 0 ? (float)tp / (tp + fn) : 0.0f;
        float f1        = (precision + recall) > 0.0f
                          ? 2.0f * precision * recall / (precision + recall)
                          : 0.0f;
        float fpr       = (fp + tn) > 0 ? (float)fp / (fp + tn) : 0.0f;

        float cycles_mean     = (float)(cycles_sum / n_samples);
        float inf_us          = cycles_mean / (IDS_HCLK_HZ / 1e6f);
        /* Energy = V × I × t  (whole-chip active, isolated to IDS duty only) */
        float energy_nj       = ((float)IDS_VDD_MV / 1000.0f)
                                * ((float)IDS_RUN_CURRENT_UA / 1e6f)
                                * (inf_us / 1e6f) * 1e9f;

        ids_bench_summary_t summary = {
            .n_samples           = n_samples,
            .n_correct           = n_correct,
            .tp = tp, .tn = tn, .fp = fp, .fn = fn,
            .cycles_min          = cycles_min,
            .cycles_max          = cycles_max,
            .cycles_sum_hi       = (uint32_t)(cycles_sum >> 32),
            .cycles_sum_lo       = (uint32_t)(cycles_sum & 0xFFFFFFFF),
            .stack_hwm_min_words = hwm_min,
            .energy_per_inf_nj   = energy_nj,
            .inference_us_mean   = inf_us,
            .accuracy            = accuracy,
            .precision           = precision,
            .recall              = recall,
            .f1                  = f1,
            .fpr                 = fpr,
        };
        uart_send(&summary, sizeof(summary));

        snprintf(logbuf, sizeof(logbuf),
                 "[IDS] Done. Acc=%.4f Rec=%.4f FPR=%.4f Inf=%.2fus E=%.2fnJ\r\n",
                 accuracy, recall, fpr, inf_us, energy_nj);
        uart_log(logbuf);
    }
}

/* ── Public init ────────────────────────────────────────────────────────────── */

void ids_benchmark_task_init(void) {
    s_bench_task_handle = xTaskCreateStatic(
        ids_benchmark_task,
        "IDS_Bench",
        IDS_TASK_STACK_SIZE,
        NULL,
        IDS_TASK_PRIORITY,
        s_bench_task_stack,
        &s_bench_task_tcb
    );
    configASSERT(s_bench_task_handle != NULL);
}
