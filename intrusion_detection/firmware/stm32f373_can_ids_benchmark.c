/**
 * stm32f373_can_ids_benchmark.c  —  bare-metal, no FreeRTOS
 * ===========================================================
 * Measures TinyDecisionTree inference on STM32F373C8T, isolated from all
 * interrupts and board operations during the timing window.
 *
 * Target: STM32F373C8T  (Cortex-M4F, 72 MHz, 64 KB Flash, 16 KB RAM)
 *
 * Isolation strategy (bare-metal):
 *   __disable_irq() is called immediately before ids_predict() and
 *   __enable_irq() immediately after. This masks every interrupt —
 *   CAN Rx, SysTick, ADCS, UART DMA — so the DWT cycle count reflects
 *   only the tree traversal, with zero ISR noise.
 *   UART Rx between samples runs with interrupts enabled; only the
 *   ~50-cycle inference window is masked.
 *
 * RAM budget (16 KB total):
 *   Canary probe  : 256 words = 1 KB  (static, .bss)
 *   Feature buffer: 15 × float = 60 B (stack, per-sample)
 *   Log buffer    : 128 B             (stack)
 *   Remaining for board stack/data: ~14.8 KB
 *
 * What is measured:
 *   - DWT cycle count per inference (1-cycle resolution at 72 MHz)
 *   - Stack depth via canary pattern
 *   - Estimated energy per inference from datasheet Idd × Vdd × t
 *     (STM32F373 datasheet Table 28: 27 mA typ at 72 MHz, 3.3V)
 *   - ML scores: accuracy, recall, precision, F1, FPR
 *
 * Wire protocol (UART 115200 8N1, host drives):
 *   Host → MCU  : 'S' + uint16_le  (start + sample count)
 *   Per sample  : 15× float32 (60 bytes) + 1× uint8 ground-truth label
 *   MCU → Host  : ids_sample_result_t  (16 bytes per sample)
 *   MCU → Host  : ids_bench_summary_t  (68 bytes, once at end)
 *
 * Required headers (copy to same folder, all auto-generated):
 *   stm32h7_can_ids_float32.h   — tree arrays + ids_predict()
 *   stm32h7_can_ids_scaler.h    — scaler params + ids_scale_features()
 *
 * Build:
 *   STM32CubeIDE: add .c to project, include path already covers Core/Inc.
 *   Call ids_benchmark_run() from main() after HAL_Init() and clock config.
 *   No osKernelStart() required.
 */

#include "stm32f373_can_ids_benchmark.h"

#include "stm32h7_can_ids_float32.h"
#include "stm32h7_can_ids_scaler.h"

#include "main.h"
#include <string.h>
#include <stdio.h>
#include <stdint.h>

extern UART_HandleTypeDef huart2;

/* ── Configuration ──────────────────────────────────────────────────────────── */

#define IDS_UART             huart2          /* STM32F373C8T: USART2 on PA2/PA3  */
#define IDS_UART_TIMEOUT_MS  5000
#define IDS_MAX_SAMPLES      8192

#define IDS_HCLK_HZ          72000000UL     /* STM32F373 max clock               */
#define IDS_VDD_MV           3300U
/* STM32F373 datasheet Table 28: IDD typ = 29.2 mA @ 72 MHz, 3.6V, peripherals
   off. Derated to 3.3V ≈ 27 mA. Use 27000 µA as conservative estimate.        */
#define IDS_RUN_CURRENT_UA   27000U

/* Stack canary — 256 words = 1 KB out of 16 KB total RAM                       */
#define CANARY_VALUE         0xDEADBEEFU
#define CANARY_WORDS         256

/* ── DWT ────────────────────────────────────────────────────────────────────── */

static inline void dwt_enable(void) {
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT       = 0;
    DWT->CTRL        |= DWT_CTRL_CYCCNTENA_Msk;
}

static inline uint32_t dwt_get(void) {
    return DWT->CYCCNT;
}

/* ── Stack canary ───────────────────────────────────────────────────────────── */

/*
 * Paint a 1 KB region on the stack with a sentinel value before the
 * benchmark loop, then scan it afterward to find the high-water mark.
 * This gives the same information as FreeRTOS uxTaskGetStackHighWaterMark()
 * without any RTOS dependency.
 */
static uint32_t s_canary[CANARY_WORDS];

static void canary_paint(void) {
    for (int i = 0; i < CANARY_WORDS; i++) s_canary[i] = CANARY_VALUE;
    /* Compiler barrier — prevent the paint from being optimised away */
    __asm volatile("" ::: "memory");
}

/* Returns number of bytes consumed (canary words that were overwritten). */
static uint32_t canary_used_bytes(void) {
    int i = 0;
    while (i < CANARY_WORDS && s_canary[i] == CANARY_VALUE) i++;
    return (uint32_t)(CANARY_WORDS - i) * 4;
}

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

/* ── Wire types ─────────────────────────────────────────────────────────────── */

typedef struct __attribute__((packed)) {
    uint8_t  prediction;
    uint8_t  ground_truth;
    uint32_t cycles;
    uint32_t stack_used_bytes;   /* canary measurement                          */
    uint32_t reserved0;
    uint32_t reserved1;
} ids_sample_result_t;           /* 16 bytes — same layout as FreeRTOS version  */

typedef struct __attribute__((packed)) {
    uint32_t n_samples;
    uint32_t n_correct;
    uint32_t tp, tn, fp, fn;
    uint32_t cycles_min;
    uint32_t cycles_max;
    uint32_t cycles_sum_hi;
    uint32_t cycles_sum_lo;
    uint32_t stack_used_bytes_max;
    float    energy_per_inf_nj;
    float    inference_us_mean;
    float    accuracy;
    float    precision;
    float    recall;
    float    f1;
    float    fpr;
} ids_bench_summary_t;           /* 68 bytes */

/* ── Main benchmark loop ────────────────────────────────────────────────────── */

void ids_benchmark_run(void) {
    char logbuf[128];
    dwt_enable();
    uart_log("[IDS] Bare-metal benchmark ready. Waiting for host...\r\n");

    for (;;) {
        /* Wait for start command 'S' */
        uint8_t cmd = 0;
        while (cmd != 'S') {
            uart_recv(&cmd, 1);
        }

        uint16_t n_samples = 0;
        if (uart_recv(&n_samples, 2) != HAL_OK ||
            n_samples == 0 || n_samples > IDS_MAX_SAMPLES) {
            uart_log("[IDS] Bad sample count\r\n");
            continue;
        }

        snprintf(logbuf, sizeof(logbuf),
                 "[IDS] Starting %u-sample benchmark\r\n", n_samples);
        uart_log(logbuf);

        /* Paint canary before the loop */
        canary_paint();

        uint32_t tp = 0, tn = 0, fp = 0, fn = 0;
        uint32_t cycles_min         = UINT32_MAX;
        uint32_t cycles_max         = 0;
        uint64_t cycles_sum         = 0;
        uint32_t stack_used_max     = 0;

        for (uint16_t i = 0; i < n_samples; i++) {
            /* ── Receive feature vector + label (interrupts enabled) ─── */
            float   features[IDS_N_FEATURES];
            uint8_t gt_label = 0;

            HAL_StatusTypeDef rx = uart_recv(features, sizeof(features));
            if (rx == HAL_OK) rx  = uart_recv(&gt_label, 1);
            if (rx != HAL_OK) {
                uart_log("[IDS] UART Rx error\r\n");
                break;
            }

            /* ── Scale features in-place ────────────────────────────── */
            ids_scale_features(features);

            /* ── Isolated inference window ──────────────────────────── */
            /*    Mask ALL interrupts: SysTick, CAN Rx, DMA, ADCS, etc. */
            __disable_irq();
            __DSB();                         /* drain store buffer        */
            __ISB();                         /* flush pipeline            */

            uint32_t t0      = dwt_get();
            int prediction   = ids_predict(features);
            uint32_t elapsed = dwt_get() - t0;

            __enable_irq();
            /* ── End of isolated window ─────────────────────────────── */

            if (elapsed < cycles_min) cycles_min = elapsed;
            if (elapsed > cycles_max) cycles_max = elapsed;
            cycles_sum += elapsed;

            uint32_t stack_used = canary_used_bytes();
            if (stack_used > stack_used_max) stack_used_max = stack_used;

            if      (prediction == 1 && gt_label == 1) tp++;
            else if (prediction == 0 && gt_label == 0) tn++;
            else if (prediction == 1 && gt_label == 0) fp++;
            else                                        fn++;

            ids_sample_result_t result = {
                .prediction        = (uint8_t)prediction,
                .ground_truth      = gt_label,
                .cycles            = elapsed,
                .stack_used_bytes  = stack_used,
                .reserved0         = 0,
                .reserved1         = 0,
            };
            uart_send(&result, sizeof(result));
        }

        /* ── Summary ────────────────────────────────────────────────── */
        uint32_t n_correct = tp + tn;
        float accuracy  = (float)n_correct / n_samples;
        float precision = (tp + fp) > 0 ? (float)tp / (tp + fp) : 0.0f;
        float recall    = (tp + fn) > 0 ? (float)tp / (tp + fn) : 0.0f;
        float f1        = (precision + recall) > 0.0f
                          ? 2.0f * precision * recall / (precision + recall) : 0.0f;
        float fpr       = (fp + tn) > 0 ? (float)fp / (fp + tn) : 0.0f;

        float cycles_mean = (float)(cycles_sum / n_samples);
        float inf_us      = cycles_mean / (IDS_HCLK_HZ / 1e6f);
        float energy_nj   = ((float)IDS_VDD_MV / 1000.0f)
                            * ((float)IDS_RUN_CURRENT_UA / 1e6f)
                            * (inf_us / 1e6f) * 1e9f;

        ids_bench_summary_t summary = {
            .n_samples            = n_samples,
            .n_correct            = n_correct,
            .tp = tp, .tn = tn, .fp = fp, .fn = fn,
            .cycles_min           = cycles_min,
            .cycles_max           = cycles_max,
            .cycles_sum_hi        = (uint32_t)(cycles_sum >> 32),
            .cycles_sum_lo        = (uint32_t)(cycles_sum & 0xFFFFFFFF),
            .stack_used_bytes_max = stack_used_max,
            .energy_per_inf_nj    = energy_nj,
            .inference_us_mean    = inf_us,
            .accuracy             = accuracy,
            .precision            = precision,
            .recall               = recall,
            .f1                   = f1,
            .fpr                  = fpr,
        };
        uart_send(&summary, sizeof(summary));

        snprintf(logbuf, sizeof(logbuf),
                 "[IDS] Done. Acc=%.4f Rec=%.4f FPR=%.4f Inf=%.2fus E=%.2fnJ\r\n",
                 accuracy, recall, fpr, inf_us, energy_nj);
        uart_log(logbuf);
    }
}
