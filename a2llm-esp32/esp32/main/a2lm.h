/* A2LM-ESP: tiny C inference engine for A2LM-0.01 (nano).
 *
 * Runs a decoder-only Transformer with int8-quantized weights
 * (see a2lm/quantize.py) and float32 activations. Zero dependencies,
 * static memory only — fits an ESP32 (or any 32-bit MCU with ~40 KB RAM).
 *
 * The math mirrors the PyTorch reference exactly:
 *   - causal scaled dot-product attention
 *   - pre-norm residual blocks with LayerNorm (eps = 1e-5)
 *   - GELU (tanh approximation, matching nn.GELU(approximate="tanh"))
 *   - tied token embedding / LM head
 *   - temperature + top-k sampling, xorshift32 PRNG
 */

#ifndef A2LM_H
#define A2LM_H

#include <stddef.h>
#include <stdint.h>

#include "weights.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Model layout (must match weights.h / quantize.py). */
#define A2LM_EPS 1e-5f

/* Run the model on `tokens[0..n)` (n <= A2LM_CTX) and fill logits[256]
 * for the LAST position (next-token distribution, pre-softmax). */
void a2lm_logits(const uint8_t *tokens, size_t n, float *logits);

/* Autoregressive generation.
 *   prompt   : input bytes (byte tokenizer: 1 token per byte)
 *   n_prompt : prompt length (<= A2LM_CTX)
 *   max_new  : max tokens to generate
 *   temp     : sampling temperature (<=0 -> greedy argmax)
 *   top_k    : sample from top-k only (0 or 1 -> greedy)
 *   rng      : in/out PRNG state (xorshift32), may be NULL (internal seed)
 *   out      : caller buffer, receives generated token bytes (max_new)
 * Returns number of tokens generated. */
size_t a2lm_generate(const uint8_t *prompt, size_t n_prompt,
                     size_t max_new, float temp, int top_k,
                     uint32_t *rng, uint8_t *out);

#ifdef __cplusplus
}
#endif

#endif /* A2LM_H */