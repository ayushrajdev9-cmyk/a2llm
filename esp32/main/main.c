/* A2LM-ESP console app (ESP-IDF).
 *
 * Boots, prints model stats, reads a prompt line from UART (or uses the
 * default when the input times out), then streams generated text.
 *
 * Build & flash (ESP-IDF v5.x):
 *     idf.py set-target esp32s3        # or esp32 / esp32c3
 *     idf.py build flash monitor
 */

#include <stdio.h>
#include <string.h>

#include "driver/uart.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "sdkconfig.h"

#include "a2lm.h"

#define UART_PORT   UART_NUM_0
#define BUF_SIZE    1024
#define DEFAULT_PROMPT "ROMEO:"
#define MAX_NEW      120
#define TEMPERATURE  0.9f
#define TOP_K        40

static const char *TAG = "a2lm-esp";

static void print_stats(void) {
    ESP_LOGI(TAG, "A2LM-ESP | params=%d | int8 weights=%d bytes (%.1f KB flash) | "
                  "ctx=%d dim=%d layers=%d heads=%d",
             A2LM_TOTAL_PARAMS, A2LM_INT8_WEIGHT_BYTES,
             (float)A2LM_INT8_WEIGHT_BYTES / 1024.0f,
             A2LM_CTX, A2LM_DIM, A2LM_LAYERS, A2LM_HEADS);
    ESP_LOGI(TAG, "free heap: %lu bytes", (unsigned long)esp_get_free_heap_size());
}

int app_main(void) {
    uint8_t prompt_buf[BUF_SIZE];
    uint8_t out_buf[MAX_NEW];
    uint8_t line[BUF_SIZE];
    size_t n_line = 0;

    uart_config_t cfg = {
        .baud_rate = 115200,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT,
    };
    ESP_ERROR_CHECK(uart_driver_install(UART_PORT, BUF_SIZE * 2, 0, 0, NULL, 0));
    ESP_ERROR_CHECK(uart_param_config(UART_PORT, &cfg));
    ESP_ERROR_CHECK(uart_set_pin(UART_PORT, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE,
                                 UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE));

    vTaskDelay(pdMS_TO_TICKS(500));   /* let the host open the monitor */
    print_stats();

    while (1) {
        uart_write_bytes(UART_PORT, (const char *)"\nA2LM> ", 7);
        n_line = 0;
        int64_t t0 = esp_timer_get_time();
        while (n_line < BUF_SIZE - 1) {
            int n = uart_read_bytes(UART_PORT, line + n_line, 1,
                                    pdMS_TO_TICKS(20));
            if (n > 0) {
                n_line += (size_t)n;
                if (line[n_line - 1] == '\n' || line[n_line - 1] == '\r') break;
            } else if (n_line == 0 && esp_timer_get_time() - t0 > 5000000) {
                break;   /* 5 s idle: run the default prompt */
            }
        }
        size_t n_prompt = n_line;
        if (n_prompt == 0 || n_prompt > A2LM_CTX) {
            const char *def = DEFAULT_PROMPT;
            n_prompt = strlen(def);
            memcpy(prompt_buf, def, n_prompt);
        } else {
            /* trim trailing newline */
            while (n_prompt > 0 &&
                   (line[n_prompt - 1] == '\n' || line[n_prompt - 1] == '\r'))
                n_prompt--;
            memcpy(prompt_buf, line, n_prompt);
        }

        uart_write_bytes(UART_PORT, "generating: ", 12);
        uart_write_bytes(UART_PORT, (const char *)prompt_buf, n_prompt);
        uart_write_bytes(UART_PORT, "\n", 1);

        uint32_t rng = (uint32_t)esp_timer_get_time() | 1u;
        int64_t t_gen0 = esp_timer_get_time();
        size_t n_out = a2lm_generate(prompt_buf, n_prompt, MAX_NEW,
                                     TEMPERATURE, TOP_K, &rng, out_buf);
        int64_t dt_us = esp_timer_get_time() - t_gen0;

        uart_write_bytes(UART_PORT, "---\n", 4);
        uart_write_bytes(UART_PORT, (const char *)out_buf, n_out);
        uart_write_bytes(UART_PORT, "\n---\n", 5);
        ESP_LOGI(TAG, "%u tokens in %.1f ms (%.0f tok/s), free heap %lu B",
                 (unsigned)n_out, (double)dt_us / 1000.0,
                 (double)n_out * 1e6 / (double)dt_us,
                 (unsigned long)esp_get_free_heap_size());
    }
}