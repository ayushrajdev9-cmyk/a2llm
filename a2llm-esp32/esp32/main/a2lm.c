/* A2LM-ESP inference engine — see a2lm.h. Pure C89-ish C, no libc deps
 * beyond math.h, all buffers static (no malloc). */

#include "a2lm.h"
#include "weights_data.h"   /* int8 weight arrays (single TU) */

#include <math.h>
#include <string.h>

/* ----------------------------- weights table ---------------------------- */

typedef struct {
    const int8_t *w;   /* rows*cols, row-major, int8  */
    const float  *s;   /* per-row scales, `rows`      */
    int rows, cols;
} A2LMMat;

typedef struct {
    const float *g, *b;
    int n;
} A2LMNorm;

/* per-layer tensors come from the generated layer table in weights.h */

static const A2LMMat TOKEN_EMB = { A2LM_TOKEN_EMB, A2LM_TOKEN_EMB_SCALE, A2LM_VOCAB, A2LM_DIM };
static const A2LMMat POS_EMB   = { A2LM_POS_EMB,   A2LM_POS_EMB_SCALE,   A2LM_CTX,   A2LM_DIM };

/* ------------------------------ workspace ------------------------------- */
/* All static: ~30 KB total, zero heap usage. */

static float BUF_H[A2LM_CTX * A2LM_DIM];      /* hidden states (context)      */
static float BUF_LN[A2LM_CTX * A2LM_DIM];     /* layernorm scratch            */
static float BUF_QKV[A2LM_CTX * 3 * A2LM_DIM];/* qkv projections              */
static float BUF_SCORE[A2LM_CTX * A2LM_CTX];  /* attention scores             */
static float BUF_ATT[A2LM_CTX * A2LM_DIM];    /* attention output             */
static float BUF_FFN[A2LM_FFN];               /* ffn hidden (one token)       */
static float BUF_DIM[A2LM_DIM];               /* misc single-token scratch    */
static float BUF_LOGITS[A2LM_VOCAB];

/* ------------------------------ primitives ------------------------------ */

/* y[i] = s[i] * (sum_j w[i][j] * x[j])   (int8 weights, float activations) */
static void matvec_int8(const A2LMMat *m, const float *x, float *y) {
    int i, j;
    for (i = 0; i < m->rows; i++) {
        float acc = 0.0f;
        const int8_t *row = m->w + (size_t)i * m->cols;
        for (j = 0; j < m->cols; j++) acc += (float)row[j] * x[j];
        y[i] = acc * m->s[i];
    }
}

/* matvec with bias: y[i] = s[i]*sum + b[i] */
static void matvec_int8_b(const A2LMMat *m, const float *x, const float *b, float *y) {
    matvec_int8(m, x, y);
    for (int i = 0; i < m->rows; i++) y[i] += b[i];
}

/* embed one row (token or position) into `out` */
static void embed_row(const A2LMMat *m, int row, float *out) {
    const int8_t *r = m->w + (size_t)row * m->cols;
    float s = m->s[row];
    for (int j = 0; j < m->cols; j++) out[j] = (float)r[j] * s;
}

/* x = (x - mean) / sqrt(var + eps) * g + b   (eps = A2LM_EPS, matches PyTorch) */
static void layernorm(float *x, const A2LMNorm *n, float *out) {
    float mean = 0.0f, var = 0.0f;
    int i;
    for (i = 0; i < n->n; i++) mean += x[i];
    mean /= (float)n->n;
    for (i = 0; i < n->n; i++) { float d = x[i] - mean; var += d * d; }
    var /= (float)n->n;
    for (i = 0; i < n->n; i++)
        out[i] = (x[i] - mean) / sqrtf(var + A2LM_EPS) * n->g[i] + n->b[i];
}

/* GELU, tanh approximation (identical to torch nn.GELU(approximate="tanh")) */
static float gelu(float x) {
    const float c = 0.7978845608f;            /* sqrt(2/pi) */
    return 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x * x * x)));
}

/* softmax over the last row of score[0..n) — the row being decoded */
static void softmax_row(float *v, size_t n) {
    float maxv = v[0], sum = 0.0f;
    for (size_t i = 1; i < n; i++) if (v[i] > maxv) maxv = v[i];
    for (size_t i = 0; i < n; i++) { v[i] = expf(v[i] - maxv); sum += v[i]; }
    for (size_t i = 0; i < n; i++) v[i] /= sum;
}

/* xorshift32 — tiny, good enough for sampling */
static uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *state = x;
    return x;
}

/* ------------------------------- forward -------------------------------- */

static void forward(const uint8_t *tokens, size_t n, float *logits) {
    size_t t, i;
    const int D = A2LM_DIM, H = A2LM_HEADS, HD = A2LM_HEAD_DIM;

    /* 1. token + positional embeddings -> BUF_H */
    for (t = 0; t < n; t++) {
        float *h = BUF_H + t * D;
        embed_row(&TOKEN_EMB, tokens[t], h);
        float *p = BUF_DIM;
        embed_row(&POS_EMB, (int)t, p);
        for (i = 0; i < (size_t)D; i++) h[i] += p[i];
    }

    /* 2. transformer blocks */
    for (int l = 0; l < A2LM_LAYERS; l++) {
        const A2LMLayerDef *L = &A2LM_LAYER_TABLE[l];
        const A2LMMat qkv = { L->qkv, L->qkv_s, 3 * D, D };
        const A2LMMat out = { L->out, L->out_s, D, D };
        const A2LMNorm ln1 = { L->ln1_g, L->ln1_b, D };
        const A2LMNorm ln2 = { L->ln2_g, L->ln2_b, D };
        const A2LMMat up  = { L->up,  L->up_s,  A2LM_FFN, D };
        const A2LMMat dn  = { L->down, L->down_s, D, A2LM_FFN };

        /* (a) pre-norm + qkv for every position */
        for (t = 0; t < n; t++)
            layernorm(BUF_H + t * D, &ln1, BUF_LN + t * D);
        for (t = 0; t < n; t++)
            matvec_int8(&qkv, BUF_LN + t * D, BUF_QKV + t * 3 * D);

        /* (b) causal multi-head attention */
        for (int h = 0; h < H; h++) {
            /* scores */
            for (size_t i = 0; i < n; i++) {
                const float *qi = BUF_QKV + i * 3 * D + (size_t)h * HD;
                for (size_t j = 0; j < n; j++) {
                    const float *kj = BUF_QKV + j * 3 * D + D + (size_t)h * HD;
                    float acc = 0.0f;
                    for (int k = 0; k < HD; k++) acc += qi[k] * kj[k];
                    BUF_SCORE[i * A2LM_CTX + j] = acc / sqrtf((float)HD);
                }
            }
            /* causal mask: future positions -> -inf */
            for (size_t i = 0; i < n; i++)
                for (size_t j = i + 1; j < n; j++)
                    BUF_SCORE[i * A2LM_CTX + j] = -1e30f;
            /* softmax + weighted sum of values */
            for (size_t i = 0; i < n; i++) {
                float *srow = BUF_SCORE + i * A2LM_CTX;
                softmax_row(srow, n);
                float *oi = BUF_ATT + i * D;
                for (int k = 0; k < HD; k++) oi[h * HD + k] = 0.0f;
                for (size_t j = 0; j < n; j++) {
                    const float *vj = BUF_QKV + j * 3 * D + 2 * D + (size_t)h * HD;
                    float w = srow[j];
                    for (int k = 0; k < HD; k++) oi[h * HD + k] += w * vj[k];
                }
            }
        }
        /* (c) output projection + residual */
        for (t = 0; t < n; t++) {
            matvec_int8(&out, BUF_ATT + t * D, BUF_LN + t * D);  /* reuse LN buf */
            float *h = BUF_H + t * D;
            for (i = 0; i < (size_t)D; i++) h[i] += BUF_LN[t * D + i];
        }

        /* (d) feed-forward + residual (token by token) */
        for (t = 0; t < n; t++) {
            layernorm(BUF_H + t * D, &ln2, BUF_LN + t * D);
            matvec_int8_b(&up, BUF_LN + t * D, L->up_b, BUF_FFN);
            for (i = 0; i < (size_t)A2LM_FFN; i++) BUF_FFN[i] = gelu(BUF_FFN[i]);
            matvec_int8_b(&dn, BUF_FFN, L->down_b, BUF_DIM);
            float *h = BUF_H + t * D;
            for (i = 0; i < (size_t)D; i++) h[i] += BUF_DIM[i];
        }
    }

    /* 3. final layernorm + tied head on the LAST position */
    layernorm(BUF_H + (n - 1) * D, &(A2LMNorm){ A2LM_LN_F_G, A2LM_LN_F_B, D }, BUF_DIM);
    for (int v = 0; v < A2LM_VOCAB; v++) {
        const int8_t *row = A2LM_TOKEN_EMB + (size_t)v * D;
        float acc = 0.0f;
        for (i = 0; i < (size_t)D; i++) acc += (float)row[i] * BUF_DIM[i];
        logits[v] = acc * A2LM_TOKEN_EMB_SCALE[v];
    }
}

/* ------------------------------- sampling ------------------------------- */

static int sample(const float *logits, float temp, int top_k, uint32_t *rng) {
    float scaled[A2LM_VOCAB];
    float maxv = logits[0];
    int v, k;

    if (temp <= 0.0f || top_k <= 1) {          /* greedy */
        int best = 0;
        for (v = 1; v < A2LM_VOCAB; v++)
            if (logits[v] > logits[best]) best = v;
        return best;
    }

    for (v = 0; v < A2LM_VOCAB; v++) scaled[v] = logits[v] / temp;
    if (top_k > 0 && top_k < A2LM_VOCAB) {
        /* find the top_k-th largest logit */
        float thr = -1e30f;
        for (k = 0; k < top_k; k++) {
            float cur = -1e30f;
            for (v = 0; v < A2LM_VOCAB; v++)
                if (scaled[v] > cur && scaled[v] < thr + 1e30f) cur = scaled[v];
            thr = cur;
        }
        for (v = 0; v < A2LM_VOCAB; v++)
            if (scaled[v] < thr) scaled[v] = -1e30f;
    }

    maxv = scaled[0];
    for (v = 1; v < A2LM_VOCAB; v++) if (scaled[v] > maxv) maxv = scaled[v];
    float sum = 0.0f;
    for (v = 0; v < A2LM_VOCAB; v++) { scaled[v] = expf(scaled[v] - maxv); sum += scaled[v]; }

    float r = (float)(xorshift32(rng) & 0xFFFFFF) / 16777216.0f * sum;
    float c = 0.0f;
    for (v = 0; v < A2LM_VOCAB; v++) {
        c += scaled[v];
        if (r <= c) return v;
    }
    return A2LM_VOCAB - 1;
}

/* -------------------------------- public -------------------------------- */

void a2lm_logits(const uint8_t *tokens, size_t n, float *logits) {
    if (n == 0) n = 1;                         /* need at least one token */
    if (n > A2LM_CTX) n = A2LM_CTX;
    forward(tokens, n, logits);
}

size_t a2lm_generate(const uint8_t *prompt, size_t n_prompt,
                     size_t max_new, float temp, int top_k,
                     uint32_t *rng, uint8_t *out) {
    uint8_t ctx[A2LM_CTX];
    size_t n = n_prompt > A2LM_CTX ? A2LM_CTX : n_prompt;
    size_t generated = 0;
    uint32_t local_rng = 0x9E3779B9u;

    if (n == 0) { ctx[0] = 0; n = 1; }         /* empty prompt: start from NUL */
    memcpy(ctx, prompt, n);
    if (rng == NULL) rng = &local_rng;

    for (size_t t = 0; t < max_new; t++) {
        forward(ctx, n, BUF_LOGITS);
        int tok = sample(BUF_LOGITS, temp, top_k, rng);
        if (n < A2LM_CTX) { ctx[n++] = (uint8_t)tok; }
        else { memmove(ctx, ctx + 1, A2LM_CTX - 1); ctx[A2LM_CTX - 1] = (uint8_t)tok; }
        out[generated++] = (uint8_t)tok;
    }
    return generated;
}