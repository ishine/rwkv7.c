#include "rwkv7.h"    

// operators
#if !defined(AVX) && !defined(NEON) && !defined(BLAS)
void vec_add(float *xout, const float *a, const float *b, int len) {
    for (int i = 0; i < len; i++) { xout[i] = a[i] + b[i]; }
}

void vec_sub(float *xout, const float *a, const float *b, int len) {
    for (int i = 0; i < len; i++) { xout[i] = a[i] - b[i]; }
}

void vec_hadamard(float *xout, const float *a, const float *b, int len) {
    for (int i = 0; i < len; i++) { xout[i] = a[i] * b[i]; }
}

void vec_bias(float *xout, const float *a, float b, int len) {
    for (int i = 0; i < len; i++) { xout[i] = a[i] + b; }
}

void vec_scale(float *xout, const float *a, float b, int len) {
    for (int i = 0; i < len; i++) { xout[i] = a[i] * b; }
}

static inline float vec_dot_product(const float *a, const float *b, int len) {
    float ret = 0.0;
    for (int i = 0; i < len; i++) { ret += a[i] * b[i]; }
    return ret;
}

void vec_out_product(float *xout, const float *a, const float *b, int len) {
    for (int i = 0; i < len; i++) {
        for (int j = 0; j < len; j++) { xout[i * len + j] = a[i] * b[j]; }
    }
}

float vec_sum(const float *x, int len) {
    float ret = 0.0f;
    for (int i = 0; i < len; i++) { ret += x[i]; }
    return ret;
}

void lerp(float *xout, const float *a, const float *b, const float *mu, int len, int seq_len) {
    // xout = b + mu * (a - b)
    for (int i = 0; i < seq_len; i++) {
        const float *_a = a + i * len;
        const float *_b = b + i * len;
        float *_xout = xout + i * len;
        for (int j = 0; j < len; j++) {
            _xout[j] = _b[j] + mu[j] * (_a[j] - _b[j]);
        }
    }
}
#endif

#ifndef BLAS
void mat_mul_vec(float *xout, const float *x, const float *w, int x_len, int xout_len, int seq_len) {
    // W (d,n) @ x (n,) -> xout (d,) 
    int d = xout_len;
    int n = x_len;
    for (int i = 0; i < seq_len; i++) {
        const float *_x = x + i * n;
        float *_xout = xout + i * d;
        for (int j = 0; j < d; j++) {
            _xout[j] = vec_dot_product(w + j * n, _x, n);
        }
    }
}
#endif

void layer_norm(float *xout, const float *x, const float *weight, const float *bias, int len, int seq_len, float eps) {
    // actually group_norm now
    for (int i = 0; i < seq_len; i++) {
        const float *_x = x + i * len;
        float *_xout = xout + i * len;
        float x_mean = vec_sum(_x, len) / len;

        float x_centered[len];
        VECBIAS(x_centered, _x, -x_mean);
        float x_var = vec_dot_product(x_centered, x_centered, len) / len;

        vec_scale(_xout, x_centered, 1.0f/sqrt(x_var + eps), len);
        vec_hadamard(_xout, _xout, weight, len);
        vec_add(_xout, _xout, bias, len);
    }
}

void softmax(float *xout, const float *x, float temp, int len) {
    float max = 0.0;
    for (int i = 0; i < len; i++) {
        if (x[i] > max) { max = x[i]; }
    }
    float sum = 0.0f;
    for (int i = 0; i < len; i++) {
        xout[i] = exp((x[i] - max) / temp);
        sum += xout[i];
    }
    for (int i = 0; i < len; i++) { xout[i] /= sum; }
}

void lora(float *xout, const float *x, const float *weight_1, const float *weight_2, int x_len, int lora_rank, int seq_len, lora_act func) {
    float tmp[lora_rank * seq_len];
    mat_mul_vec(tmp, x, weight_1, x_len, lora_rank, seq_len);
    switch (func) {
        case LORA_NONE: break;
        case LORA_TANH: VECTANH(tmp, lora_rank * seq_len); break;
        case LORA_SIGM: VECSIGM(tmp, lora_rank * seq_len); break;
        default: ERR(1, "unknown lora activation function");
    }
    mat_mul_vec(xout, tmp, weight_2, lora_rank, x_len, seq_len);
}

// utils for tokenizer
// temporarily use the implementation from llama2.c
int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex *)a)->str, ((TokenIndex *)b)->str);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    TokenIndex tok = { .str = str }; // acts as the key to search for
    TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(rwkv_tokenizer *t, char *text, int *tokens, int *n_tokens) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    ERR(text == NULL, "cannot encode NULL text");

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    char* str_buffer = malloc((t->max_token_length*2 +1 +2) * sizeof(char));
    int str_len = 0;

    // start at 0 tokens
    *n_tokens = 0;

    if (text[0] == '\0') {
        tokens[(*n_tokens)++] = 1;
    }

    // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
    // Code point ↔ UTF-8 conversion
    // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
    // U+0000	U+007F	    0xxxxxxx
    // U+0080	U+07FF	    110xxxxx	10xxxxxx
    // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
    // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

    // process the raw (UTF-8) byte sequence of the input string
    for (char *c = text; *c != '\0'; c++) {

        // reset buffer if the current byte is ASCII or a leading byte
        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        // 0x80 is 10000000
        // in UTF-8, all continuation bytes start with "10" in first two bits
        // so in English this is: "if this byte is not a continuation byte"
        if ((*c & 0xC0) != 0x80) {
            // this byte must be either a leading byte (11...) or an ASCII char (0x...)
            // => reset our location, as we're starting a new UTF-8 codepoint
            str_len = 0;
        }

        // append the current byte to the buffer
        str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
        str_buffer[str_len] = '\0';

        // while the next character is a continuation byte, continue appending
        // but if there are too many of them, just stop to avoid overruning str_buffer size.
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens[(*n_tokens)++] = id;
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (int i=0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (1) {
        int best_id = -1;
        int best_idx = -1;

        for (int i=0; i < (*n_tokens-1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1) {
                // this merge pair exists in vocab! record its score and position
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--; // token length decreased
    }

    free(str_buffer);
}

// sampler
int compare_probs(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

int sample_logits(float* logits, rwkv_config *c, rwkv_sampler *s) {
    int next = 0;
    // apply repetition penalty to logits
    for (int i = 0; i < c->vocab_size; i++) {
        logits[i] -= s->presence_penalty + s->occurrence[i] * s->frequency_penalty;
    }
    if (s->temperature != 0.0f) {
        softmax(logits, logits, s->temperature, c->vocab_size);
    }

    ProbIndex sorted_probs[c->vocab_size];
    for (int i = 0; i < c->vocab_size; i++) {
        sorted_probs[i].index = i;
        sorted_probs[i].prob = logits[i];
    }
    qsort(sorted_probs, c->vocab_size, sizeof(ProbIndex), compare_probs);

    if (s->temperature != 0.0f) {
        // calculate cutoff length
        int cutoff_len = 0;
        float cumulative_probs = 0;
        for (; cutoff_len < c->vocab_size; cutoff_len++) {
            cumulative_probs += sorted_probs[cutoff_len].prob;
            if (s->top_p <= cumulative_probs) {
                cutoff_len++;
                break;
            }
        }
        // roll a die!
        float die = (float)rand() / (float)RAND_MAX * cumulative_probs;
        cumulative_probs = 0.0f;
        for (int i = 0; i < cutoff_len; i++) {
            cumulative_probs += sorted_probs[i].prob;
            if (die <= cumulative_probs) {
                next = sorted_probs[i].index;
                break;
            }
        }
    }
    else {
        next = sorted_probs[0].index;
    }

    // maintain repetition penalty
    for (int i = 0; i < c->vocab_size; i++) { s->occurrence[i] *= s->presence_penalty; }
    s->occurrence[next] += 1;

    return next;
}

// RWKV block
void wkv_kernel(
    float *y,
    block_weights *bw, rwkv_config *c,
    float *state,
    const float *r, const float *w, const float *k, const float *v, float *kk, const float *a
) {
    // multi-head
    for (int i = 0; i < c->n_head; i++) {
        float *head_state   = state + i * c->head_size * c->head_size;
        float *head_kk      = kk    + i * c->head_size;
        float *head_y       = y     + i * c->head_size;
        const float *head_r = r     + i * c->head_size;
        const float *head_w = w     + i * c->head_size;
        const float *head_k = k     + i * c->head_size;
        const float *head_v = v     + i * c->head_size;
        const float *head_a = a     + i * c->head_size;

        const float *ln_w   = bw->att_ln_x_weight   + i * c->head_size;
        const float *ln_b   = bw->att_ln_x_bias     + i * c->head_size;
        const float *r_k    = bw->att_r_k           + i * c->head_size;

        // kk /= np.maximum(np.linalg.norm(kk, axis=1,keepdims=1), 1e-12)
        do {
            float kk_norm = vec_dot_product(head_kk, head_kk, c->head_size);
            kk_norm = sqrt(kk_norm);
            vec_scale(head_kk, head_kk, 1.0f/MAXIMUM(kk_norm, 1e-12f), c->head_size);
        } while(0); // kk /= np.maximum(np.linalg.norm(kk, axis=1,keepdims=1), 1e-12)

        // RWKV v7 paper: S = S @ (diag(w) - kk @ (kk * a).mT) + v * k.mT
        // - multiply S into the parentheses, 
        // - to avoid non-contiguous memory access of column vectors in matrix computation
        // - S = S * w.mT - S @ kk * (kk * a).mT + v * k.mT
        do {
            float state_mul_kk[c->head_size];
            mat_mul_vec(state_mul_kk, head_kk, head_state, c->head_size, c->head_size, 1); // S @ kk

            float kk_mul_a[c->head_size];
            HADAMARD(kk_mul_a, head_kk, head_a);                                        // kk * a

            float tmp[c->head_size * c->head_size];
            vec_out_product(tmp, state_mul_kk, kk_mul_a, c->head_size);                 // S @ kk * (kk*a).mT

            float v_mul_k[c->head_size * c->head_size];
            vec_out_product(v_mul_k, head_v, head_k, c->head_size);                     // v * k.mT

            for (int j = 0; j < c->head_size; j++) {
                float *state_row = head_state + j * c->head_size;
                vec_hadamard(state_row, state_row, head_w, c->head_size);               // S = S * w.mT
            }

            vec_sub(head_state, head_state, tmp, c->head_size * c->head_size);          // S -= S @ kk * (kk * a).mT
            vec_add(head_state, head_state, v_mul_k, c->head_size * c->head_size);      // S += v * k.mT
        } while(0); // S = S * w.mT - S @ kk * (kk * a).mT + v * k.mT

        // y = S @ r
        mat_mul_vec(head_y, head_r, head_state, c->head_size, c->head_size, 1);

        // y = group_norm(y, ln_w, ln_b)
        layer_norm(head_y, head_y, ln_w, ln_b, c->head_size, 1, 64e-5f);

        // y += ((r * k * r_k).sum(axis=1,keepdims=1) * v).flatten()
        do {
            float k_mul_r_k[c->head_size];
            HADAMARD(k_mul_r_k, head_k, r_k);
            float y_sum_ = vec_dot_product(head_r, k_mul_r_k, c->head_size);
            float head_u[c->head_size];
            VECSCALE(head_u, head_v, y_sum_);
            vec_add(head_y, head_y, head_u, c->head_size);
        } while(0); // y += ((r * k * r_k).sum(axis=1,keepdims=1) * v).flatten()
    }   // multi-head

}

void time_mixing(float *dx, const float *x, float *v0, float *last_x, float *state, block_weights *bw, rwkv_config *c, int seq_len) {
    float x_shift[c->n_embd * seq_len];
    memcpy(x_shift, last_x, sizeof(float) * c->n_embd);
    memcpy(x_shift + c->n_embd, x, sizeof(float) * c->n_embd * (seq_len - 1));

    float x_lerp[c->n_embd * seq_len];
    float *wkv_input = malloc(sizeof(float) * c->n_embd * seq_len * 6); // r, w, k, v, kk, a

    // r = Wr @ xr
    float *r = wkv_input + 0 * c->n_embd * seq_len;
    lerp(x_lerp, x_shift, x, bw->att_x_r, c->n_embd, seq_len);
    mat_mul_vec(r, x_lerp, bw->att_receptance_weight, c->n_embd, c->n_embd, seq_len);

    // w = np.exp(-sigmoid(np.tanh(xw @ Ww1) @ Ww2 + w_bias)/np.e**0.5)
    float *w = wkv_input + 1 * c->n_embd * seq_len;
    lerp(x_lerp, x_shift, x, bw->att_x_w, c->n_embd, seq_len);
    do {
        float w_sigmoid_[c->n_embd * seq_len];
        lora(w_sigmoid_, x_lerp, bw->att_w1_T, bw->att_w2_T, c->n_embd, c->w_lora_r, seq_len, LORA_TANH);   // np.tanh(xw @ Ww1) @ Ww2
        VECADD_SEQ(w_sigmoid_, w_sigmoid_, bw->att_w0, c->n_embd, seq_len);                                 // np.tanh(xw @ Ww1) @ Ww2 + w_bias
        VECSIGM(w_sigmoid_, c->n_embd * seq_len);                                                           // sigmoid(...)
        for (int i = 0; i < c->n_embd * seq_len; i++) { w[i] = exp(-w_sigmoid_[i] / SQRT_E_VALUE); }        // exp(...)
    } while(0); // w = np.exp(-sigmoid(np.tanh(xw @ Ww1) @ Ww2 + w_bias)/np.e**0.5)

    // k = Wk @ xk
    float *k = wkv_input + 2 * c->n_embd * seq_len;
    lerp(x_lerp, x_shift, x, bw->att_x_k, c->n_embd, seq_len);
    mat_mul_vec(k, x_lerp, bw->att_key_weight, c->n_embd, c->n_embd, seq_len);

    // v = Wv @ xv
    float *v = wkv_input + 3 * c->n_embd * seq_len;
    lerp(x_lerp, x_shift, x, bw->att_x_v, c->n_embd, seq_len);
    mat_mul_vec(v, x_lerp, bw->att_value_weight, c->n_embd, c->n_embd, seq_len);

    if (IS_NAN(v0[0])) {
        memcpy(v0, v, sizeof(float) * c->n_embd * seq_len);
    }
    else {
        // v += (v0 - v) * sigmoid(xv @ Wv1 @ Wv2 + v_bias)
        float v_sigmoid_[c->n_embd * seq_len];
        lora(v_sigmoid_, x_lerp, bw->att_v1_T, bw->att_v2_T, c->n_embd, c->v_lora_r, seq_len, LORA_NONE);   // xv @ Wv1 @ Wv2
        VECADD_SEQ(v_sigmoid_, v_sigmoid_, bw->att_v0, c->n_embd, seq_len);                                 // xv @ Wv1 @ Wv2 + v_bias
        VECSIGM(v_sigmoid_, c->n_embd * seq_len);                                                           // sigmoid(...)
        lerp(v, v0, v, v_sigmoid_, c->n_embd * seq_len, 1);                                                 // (v0 - v) * sigmoid(...)
    }

    // kk = k * k_k
    float *kk = wkv_input + 5 * c->n_embd * seq_len;
    HADAMARD_SEQ(kk, k, bw->att_k_k, c->n_embd, seq_len);

    // a = sigmoid(xa @ Wa1 @ Wa2 + a_bias)
    float *a = wkv_input + 4 * c->n_embd * seq_len;
    lerp(x_lerp, x_shift, x, bw->att_x_a, c->n_embd, seq_len);
    lora(a, x_lerp, bw->att_a1_T, bw->att_a2_T, c->n_embd, c->a_lora_r, seq_len, LORA_NONE);    // xa @ Wa1 @ Wa2
    VECADD_SEQ(a, a, bw->att_a0, c->n_embd, seq_len);                                           // xa @ Wa1 @ Wa2 + a_bias
    VECSIGM(a, c->n_embd * seq_len);                                                            // sigmoid(...)

    // g = sigmoid(xg @ Wg1) @ Wg2
    float g[c->n_embd * seq_len];
    lerp(x_lerp, x_shift, x, bw->att_x_g, c->n_embd, seq_len);
    lora(g, x_lerp, bw->att_g1_T, bw->att_g2_T, c->n_embd, c->g_lora_r, seq_len, LORA_SIGM);
    
    // k = k * lerp(a, 1, k_a)
    // -> k += k * (a-1) * k_a
    do {
        float a_minus_1[c->n_embd * seq_len];
        VECBIAS(a_minus_1, a, -1.0f);
        HADAMARD_SEQ(a_minus_1, a_minus_1, bw->att_k_a, c->n_embd, seq_len);
        HADAMARD(a_minus_1, k, a_minus_1);
        vec_add(k, k, a_minus_1, c->n_embd * seq_len);
    } while(0);

    // wkv kernel
    float y[c->n_embd * seq_len];
    for (int i = 0; i < seq_len; i++) {
        float *_y = y + i * c->n_embd;
        float *_r = r + i * c->n_embd;
        float *_w = w + i * c->n_embd;
        float *_k = k + i * c->n_embd;
        float *_v = v + i * c->n_embd;
        float *_kk = kk + i * c->n_embd;
        float *_a = a + i * c->n_embd;
        wkv_kernel(_y, bw, c, state, _r, _w, _k, _v, _kk, _a);
    }
    free(wkv_input);

    // dx = Wo @ (y * g)
    do {
        HADAMARD(y, y, g);
        mat_mul_vec(dx, y, bw->att_output_weight, c->n_embd, c->n_embd, seq_len);
    } while(0); // dx = Wo @ (y * g)

    // last_x = x
    memcpy(last_x, x + (seq_len - 1) * c->n_embd, sizeof(float) * c->n_embd);
}

void channel_mixing(float *dx, const float *x, float *last_x, const float *s_emb_x_T, block_weights *bw, rwkv_config *c, int seq_len) {
    float k[c->n_embd * 4 * seq_len];
    float x_shift[c->n_embd * seq_len];
    memcpy(x_shift, last_x, sizeof(float) * c->n_embd);
    memcpy(x_shift + c->n_embd, x, sizeof(float) * c->n_embd * (seq_len - 1));
    float xk[c->n_embd * seq_len];
    lerp(xk, x_shift, x, bw->ffn_x_k, c->n_embd, seq_len);

    mat_mul_vec(k, xk, bw->ffn_key_weight, c->n_embd, c->n_embd * 4, seq_len);
    for (int i = 0; i < ARRLEN(k); i++) { k[i] = SQUARE(RELU(k[i])); }
    
    if (c->de) {
        float s[c->n_embd * 4 * seq_len];
        float ss[c->s_lora_r * seq_len], ss_[c->s_lora_r * seq_len];
        mat_mul_vec(ss, x, bw->ffn_s1_T, c->n_embd, c->s_lora_r, seq_len);
        for (int i = 0; i < seq_len; i++) {
            mat_mul_vec(
                ss_ + i * c->s_lora_r, 
                ss + i * c->s_lora_r, 
                s_emb_x_T + i * c->s_lora_r * 32,
                c->s_lora_r, 32, 1
            );
        }
        mat_mul_vec(s, ss_, bw->ffn_s2_T, c->s_lora_r, c->n_embd * 4, seq_len);
        VECADD_SEQ(s, s, bw->ffn_s0, c->n_embd, seq_len);
        HADAMARD(k, k, s);
    }

    float *v = dx;
    mat_mul_vec(v, k, bw->ffn_value_weight, c->n_embd * 4, c->n_embd, seq_len);
    
    memcpy(last_x, x + (seq_len - 1) * c->n_embd, sizeof(float) * c->n_embd);
}

void forward(
    float *logits,                      // output
    rwkv_config *c, rwkv_weights *w,    // model
    float *last_x, float *wkv_state,    // hidden states
    int *token_list, int seq_len        // input
) {
    float x[c->n_embd * seq_len], _x[c->n_embd * seq_len];
    for (int i = 0; i < seq_len; i++) {
        memcpy(x + i * c->n_embd, w->emb_weight + token_list[i] * c->n_embd, sizeof(float) * c->n_embd);
    }
    memcpy(_x, x, sizeof(float) * ARRLEN(x));

    float x_[c->n_embd * seq_len];
    float v0[c->n_embd * seq_len]; v0[0] = NAN;
    float dx[c->n_embd * seq_len];

    for (int i = 0; i < c->n_layer; i++) {
        layer_norm(x_, x, w->blocks[i].ln1_weight, w->blocks[i].ln1_bias, c->n_embd, seq_len, 1e-5f);

        int last_x_offset = IDX(i, 0, 0, 2, c->n_embd);
        int state_offset = i * c->n_head * c->head_size * c->head_size;
        time_mixing(dx, x_, v0, last_x + last_x_offset, wkv_state + state_offset, w->blocks + i, c, seq_len);
        VECADD(x, x, dx);

        layer_norm(x_, x, w->blocks[i].ln2_weight, w->blocks[i].ln2_bias, c->n_embd, seq_len, 1e-5f);
        float s_emb_x_T[c->s_lora_r * 32 * seq_len];
        if (c->de) {
            mat_mul_vec(s_emb_x_T, _x, w->blocks[i].ffn_s_emb_x_weight, c->n_embd, c->s_lora_r * 32, seq_len);
            for (int j = 0; j < seq_len; j++) {
                float *_s_emb_x_T = s_emb_x_T + j * c->s_lora_r * 32;
                vec_add(_s_emb_x_T, _s_emb_x_T, w->blocks[i].ffn_s_emb + token_list[j] * c->s_lora_r * 32, c->s_lora_r * 32);
                mat_transpose(_s_emb_x_T, c->s_lora_r, 32);
            }
        }
        last_x_offset = IDX(i, 1, 0, 2, c->n_embd);
        channel_mixing(dx, x_, last_x + last_x_offset, s_emb_x_T, w->blocks + i, c, seq_len);
        VECADD(x, x, dx);
    }

    if (!c->seq_full_output) {
        memcpy(x, x + (seq_len - 1) * c->n_embd, sizeof(float) * c->n_embd);
        seq_len = 1;
    }
    layer_norm(x, x, w->ln_out_weight, w->ln_out_bias, c->n_embd, seq_len, 1e-5f);
    mat_mul_vec(logits, x, w->head_weight, c->n_embd, c->vocab_size, seq_len);

}

void run(
    float *logits,                                                      // output
    rwkv_config *c, rwkv_weights *w, rwkv_tokenizer *t, rwkv_sampler *s,// model
    float *last_x, float *wkv_state,                                    // hidden states
    int *token_list, int prefilling_tokens                              // input
) {
    long start, end;

    // prefilling
    SYSTIME_MS(start);
    for (int i = 0; i < prefilling_tokens; i += c->chunk_size) {
        int this_chunk_size = MINIMUM(c->chunk_size, prefilling_tokens - i);
        forward(logits, c, w, last_x, wkv_state, token_list + i, this_chunk_size);
    }
    SYSTIME_MS(end);
    c->prefilling_time = end - start;
    if (c->chat_mode) {
        for (int i = 0; i < prefilling_tokens; i++) {
            const char *token_str = t->vocab[token_list[i]];
            printf("%s", token_str);
            fflush(stdout);
        }
    }

    // decoding
    SYSTIME_MS(start);
    for (c->decoding_tokens = 0; c->decoding_tokens < c->max_dec_len; c->decoding_tokens++) {
        int next_token = sample_logits(logits, c, s);
        if ((next_token == 0) && !c->bench_mode) { printf("\n---Meet EOS!---\n"); break; }

        forward(logits, c, w, last_x, wkv_state, &next_token, 1);
        if (!c->bench_mode) {
            const char *token_str = t->vocab[next_token];
            // if (strncmp(token_str, "\n\n", 2) == 0) { break; }

            printf("%s", token_str);
            fflush(stdout);
        }
    }
    SYSTIME_MS(end);
    c->decoding_time = end - start;
}

// load and free
void mat_transpose(float *mat, int rows, int cols) {
    float *tmp = malloc(rows * cols * sizeof(float));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            tmp[j * rows + i] = mat[i * cols + j];
        }
    }
    memcpy(mat, tmp, rows * cols * sizeof(float));
    free(tmp);
}

void load_model(const char *model_path, rwkv_config *c, rwkv_weights *w) {
    FILE *fp = fopen(model_path, "r");
    ERR(fp == NULL, "failed to open model file");

    // get file size
    fseek(fp, 0, SEEK_END);
    size_t file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    // load header
    #pragma pack(push, 1)   // prevent compiler memory align
    struct {
        uint64_t magic_number;
        int32_t quant;
        int32_t head_size;
        int32_t n_embd;
        int32_t n_layer;
        int32_t vocab_size;
        int32_t w_lora_r;
        int32_t a_lora_r;
        int32_t g_lora_r;
        int32_t v_lora_r;
        int32_t de;
        int32_t dea;
        int32_t s_lora_r;
    } header;
    #pragma pack(pop)
    ERR(fread(&header, sizeof(header), 1, fp) != 1, "failed to get model header");

    ERR(header.magic_number != 0x00632E37766B7772, "invalid model magic number");
    ERR(header.quant != 0, "quantized model is not supported");

    // set model config
    c->head_size    = header.head_size;
    c->vocab_size   = header.vocab_size;
    c->n_embd       = header.n_embd;
    c->n_layer      = header.n_layer;
    c->n_head       = header.n_embd / c->head_size;
    c->w_lora_r     = header.w_lora_r;
    c->a_lora_r     = header.a_lora_r;
    c->g_lora_r     = header.g_lora_r;
    c->v_lora_r     = header.v_lora_r;
    c->de           = header.de;
    c->dea          = header.dea;
    c->s_lora_r     = header.s_lora_r;
    // load weights
    w->blocks = malloc(c->n_layer * sizeof(block_weights));

    size_t raw_weights_size = file_size - sizeof(header);
    w->raw = malloc(raw_weights_size);
    ERR(fread(w->raw, sizeof(uint8_t), raw_weights_size, fp) != raw_weights_size, "failed to load model weights");
    fclose(fp);

    float *ptr = w->raw;
    w->emb_weight                   = ptr; ptr += c->vocab_size * c->n_embd     ;
    w->blocks_0_ln0_weight          = ptr; ptr += c->n_embd                     ;
    w->blocks_0_ln0_bias            = ptr; ptr += c->n_embd                     ;
    for (int i = 0; i < c->n_layer; i++) {
        block_weights *b = w->blocks + i;
        b->ln1_weight               = ptr; ptr += c->n_embd                     ;
        b->ln1_bias                 = ptr; ptr += c->n_embd                     ;
        b->ln2_weight               = ptr; ptr += c->n_embd                     ;
        b->ln2_bias                 = ptr; ptr += c->n_embd                     ;
        b->att_x_r                  = ptr; ptr += c->n_embd                     ;
        b->att_x_w                  = ptr; ptr += c->n_embd                     ;
        b->att_x_k                  = ptr; ptr += c->n_embd                     ;
        b->att_x_v                  = ptr; ptr += c->n_embd                     ;
        b->att_x_a                  = ptr; ptr += c->n_embd                     ;
        b->att_x_g                  = ptr; ptr += c->n_embd                     ;
        b->att_w0                   = ptr; ptr += c->n_embd                     ;
        b->att_r_k                  = ptr; ptr += c->n_head     * c->head_size  ;
        b->att_w1_T                 = ptr; ptr += c->n_embd     * c->w_lora_r   ;
        b->att_w2_T                 = ptr; ptr += c->w_lora_r   * c->n_embd     ;
        b->att_a1_T                 = ptr; ptr += c->n_embd     * c->a_lora_r   ;
        b->att_a2_T                 = ptr; ptr += c->a_lora_r   * c->n_embd     ;
        b->att_a0                   = ptr; ptr += c->n_embd                     ;
        b->att_g1_T                 = ptr; ptr += c->n_embd     * c->g_lora_r   ;
        b->att_g2_T                 = ptr; ptr += c->g_lora_r   * c->n_embd     ;
        if (i != 0) {
            b->att_v2_T             = ptr; ptr += c->v_lora_r   * c->n_embd     ;
            b->att_v1_T             = ptr; ptr += c->n_embd     * c->v_lora_r   ;
            b->att_v0               = ptr; ptr += c->n_embd                     ;
        }
        b->att_k_k                  = ptr; ptr += c->n_embd                     ;
        b->att_k_a                  = ptr; ptr += c->n_embd                     ;
        b->att_receptance_weight    = ptr; ptr += c->n_embd     * c->n_embd     ;
        b->att_key_weight           = ptr; ptr += c->n_embd     * c->n_embd     ;
        b->att_value_weight         = ptr; ptr += c->n_embd     * c->n_embd     ;
        b->att_output_weight        = ptr; ptr += c->n_embd     * c->n_embd     ;
        b->att_ln_x_weight          = ptr; ptr += c->n_embd                     ;
        b->att_ln_x_bias            = ptr; ptr += c->n_embd                     ;
        b->ffn_x_k                  = ptr; ptr += c->n_embd                     ;
        b->ffn_key_weight           = ptr; ptr += c->n_embd     * c->n_embd * 4 ;
        b->ffn_value_weight         = ptr; ptr += c->n_embd * 4 * c->n_embd     ;
        if (c->de) {
            b->ffn_s1_T             = ptr; ptr += c->n_embd     * c->s_lora_r   ;
            b->ffn_s2_T             = ptr; ptr += c->s_lora_r   * c->n_embd * 4 ;
            b->ffn_s0               = ptr; ptr += c->n_embd * 4                 ;
            b->ffn_s_emb_x_weight   = ptr; ptr += c->s_lora_r * 32 * c->n_embd  ;
        }
    }
    w->ln_out_weight                = ptr; ptr += c->n_embd                     ;
    w->ln_out_bias                  = ptr; ptr += c->n_embd                     ;
    w->head_weight                  = ptr; ptr += c->n_embd     * c->vocab_size ;

    ERR((ptr - w->raw) * sizeof(float) != raw_weights_size, "failed to map model weights");

    // merge ln0 into emb
    layer_norm((float *)w->emb_weight, w->emb_weight, w->blocks_0_ln0_weight, w->blocks_0_ln0_bias, c->n_embd, c->vocab_size, 1e-5f);

    // transpose all lora weight matrices
    for (int i = 0; i < c->n_layer; i++) {
        block_weights *b = w->blocks + i;
        if (i != 0) {
            mat_transpose((float *)b->att_v1_T, c->n_embd, c->v_lora_r);
            mat_transpose((float *)b->att_v2_T, c->v_lora_r, c->n_embd);
        }
        mat_transpose((float *)b->att_w1_T, c->n_embd, c->w_lora_r);
        mat_transpose((float *)b->att_w2_T, c->w_lora_r, c->n_embd);
        mat_transpose((float *)b->att_a1_T, c->n_embd, c->a_lora_r);
        mat_transpose((float *)b->att_a2_T, c->a_lora_r, c->n_embd);
        mat_transpose((float *)b->att_g1_T, c->n_embd, c->g_lora_r);
        mat_transpose((float *)b->att_g2_T, c->g_lora_r, c->n_embd);
    }

    if (c->de) {
        // load extra weights
        char *extra_path = malloc(strlen(model_path) + 6 + 1);
        sprintf(extra_path, "%s.extra", model_path);
        int fd = open(extra_path, O_RDONLY);
        ERR(fd == -1, "failed to open extra file");
        struct stat sb;
        ERR(fstat(fd, &sb) == -1, "fstat failed");
        w->extra_size = sb.st_size;
        w->extra_raw = mmap(NULL, w->extra_size, PROT_READ, MAP_PRIVATE, fd, 0);
        ERR(w->extra_raw == MAP_FAILED, "mmap failed");
        close(fd);
        free(extra_path);

        for (int i = 0; i < c->n_layer; i++) {
            block_weights *b = w->blocks + i;
            mat_transpose((float *)b->ffn_s1_T, c->n_embd, c->s_lora_r);
            mat_transpose((float *)b->ffn_s2_T, c->s_lora_r, c->n_embd * 4);
            b->ffn_s_emb = w->extra_raw + i * c->s_lora_r * 32 * c->vocab_size;
        }
    }
    else { w->extra_raw = NULL; }
}

void free_model(rwkv_weights *w) {
    free(w->blocks);
    free(w->raw);
    if (w->extra_raw) { munmap((void *)w->extra_raw, w->extra_size); }
}

void load_tokenizer(rwkv_tokenizer *t, int vocab_size) {
    t->vocab = _vocab;
    t->vocab_size = vocab_size;
    t->max_token_length = 0;

    t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex));
    int sorted_vocab_idx = 0;
    for (int i = 0; i <= t->vocab_size; i++) {
        if (t->vocab[i]) {
            t->sorted_vocab[sorted_vocab_idx].str = t->vocab[i];
            t->sorted_vocab[sorted_vocab_idx].id = i;
            sorted_vocab_idx++;

            t->max_token_length = MAXIMUM(strlen(t->vocab[i]), t->max_token_length);
        }
    }
    ERR(sorted_vocab_idx != t->vocab_size, "failed to load vocabulary");

    qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
}

void free_tokenizer(rwkv_tokenizer *t) {
    free(t->sorted_vocab);
}

// main
void print_usage(char *argv[]) {
    fprintf(stderr, "Usage: %s [options] model_path\n", argv[0]);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  --chat                       enable chat mode\n");
    fprintf(stderr, "  --reasoner                   enable reasoner mode\n");
    fprintf(stderr, "  --bench                      enable benchmark mode\n");
    fprintf(stderr, "  -i, --input <input message>  model inference input\n");
    fprintf(stderr, "  --temperature <float>        sample temperature\n");
    fprintf(stderr, "  --top-p <float>              sample top-p\n");
    fprintf(stderr, "  --presence_penalty <float>   presence penalty\n");
    fprintf(stderr, "  --frequency_penalty <float>  frequency penalty\n");
    fprintf(stderr, "  --seed <int>                 random seed\n");
    fprintf(stderr, "  -l --max_dec_len <int>       max decoding length\n");
    fprintf(stderr, "  --chunk_size <int>           chunk size for prefilling (default: 64)\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {
    rwkv_config config = {
        .chat_mode = false, .reasoner_mode = false, .bench_mode = false,
        .max_dec_len = 10240, .chunk_size = 64, .seq_full_output = false
    };
    rwkv_weights weights;
    rwkv_tokenizer tokenizer;
    rwkv_sampler sampler = { .temperature = 1.0, .top_p = 0.7, .presence_penalty = 0.1, .frequency_penalty = 0.2 };
    unsigned int seed = time(NULL);
    const char *msg = NULL;
    const char *model_path = NULL;

    if (argc < 2) { print_usage(argv); }
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--chat") == 0) 
            { config.chat_mode = true; }
        else if (strcmp(argv[i], "--reasoner") == 0)
            { config.chat_mode = true; config.reasoner_mode = true; }
        else if (strcmp(argv[i], "--bench") == 0)
            { config.bench_mode = true; }
        else if ((strcmp(argv[i], "-i") == 0) || (strcmp(argv[i], "--input") == 0))
            { msg = argv[i + 1]; i++; }
        else if (strcmp(argv[i], "--temperature") == 0)
            { sampler.temperature = atof(argv[i + 1]); i++; }
        else if (strcmp(argv[i], "--top-p") == 0) 
            { sampler.top_p = atof(argv[i + 1]); i++; }
        else if (strcmp(argv[i], "--seed") == 0) 
            { seed = atoi(argv[i + 1]); i++; }
        else if (strcmp(argv[i], "--presence_penalty") == 0)
            { sampler.presence_penalty = atof(argv[i + 1]); i++; }
        else if (strcmp(argv[i], "--frequency_penalty") == 0)
            { sampler.frequency_penalty = atof(argv[i + 1]); i++; }
        else if ((strcmp(argv[i], "--max_dec_len") == 0) || (strcmp(argv[i], "-l") == 0))
            { config.max_dec_len = atoi(argv[i + 1]) <= 0 ? config.max_dec_len : atoi(argv[i + 1]); i++; }
        else if (strcmp(argv[i], "--chunk_size") == 0)
            { config.chunk_size = atoi(argv[i + 1]) <= 0 ? 1 : atoi(argv[i + 1]); i++; }
        else { model_path = argv[i]; }
    }
    if (model_path == NULL) { print_usage(argv); }
    if ((msg == NULL) && (!config.bench_mode)) { print_usage(argv); }
    if (config.bench_mode) {
        config.max_dec_len = 128;
        sampler.temperature = 0.0;
    }
    if (sampler.temperature         < 0.0) { sampler.temperature        = 0.0; }
    if (sampler.top_p               < 0.0) { sampler.top_p              = 0.0; }
    if (sampler.presence_penalty    < 0.0) { sampler.presence_penalty   = 0.0; }
    if (sampler.frequency_penalty   < 0.0) { sampler.frequency_penalty  = 0.0; }
    srand(seed);

    printf("Hello, RWKV! seed: %u\n\n", seed);
    printf("Loading model...\n");
    load_model(model_path, &config, &weights);
    printf("Model loaded!\n\n");

    printf("Loading tokenizer...\n");
    load_tokenizer(&tokenizer, VOCAB_SIZE);
    printf("Tokenizer loaded!\n\n");

    sampler.occurrence = calloc(config.vocab_size, sizeof(int));

    int *token_list = NULL;
    int prefilling_tokens = 0;
    if (!config.bench_mode) {
        const char *prompt_tmpl = "%s";
        if (config.chat_mode && !config.reasoner_mode)
            { prompt_tmpl = "User: %s\n\nAssistant: "; }
        else if (config.chat_mode && config.reasoner_mode)
            { prompt_tmpl = "User: %s\n\nAssistant: <think>"; }

        int context_len = snprintf(NULL, 0, prompt_tmpl, msg);
        char *context = malloc(context_len + 1);
        sprintf(context, prompt_tmpl, msg);
        token_list = malloc(sizeof(int) * context_len);
        encode(&tokenizer, context, token_list, &prefilling_tokens);
        free(context);
    }
    else {
        prefilling_tokens = 512;
        token_list = malloc(sizeof(int) * prefilling_tokens);
        for (int i = 0 ; i < prefilling_tokens; i++) {
            // token_list[i] = (int)((double)rand() / RAND_MAX * config.vocab_size);
            token_list[i] = rand() % config.vocab_size;
        }
    }

    // run
    float logits[config.vocab_size];
    float *last_x, *wkv_state;  // init with zero
    last_x = calloc(config.n_layer * 2 * config.n_embd, sizeof(float));
    wkv_state = calloc(config.n_layer * config.n_head * config.head_size * config.head_size, sizeof(float));
    if (!config.bench_mode) {
        run(
            logits,
            &config, &weights, &tokenizer, &sampler,
            last_x, wkv_state,
            token_list, prefilling_tokens
        );
        float prefilling_speed = (double)prefilling_tokens / config.prefilling_time * 1000;
        float decoding_speed = (double)config.decoding_tokens / config.decoding_time * 1000;
        printf("\n---------\n");
        printf("Prefill: %d tokens, %ld ms, %f token/s\n",
            prefilling_tokens, config.prefilling_time, prefilling_speed);
        printf("Decode: %d tokens, %ld ms, %f token/s\n",
            config.decoding_tokens, config.decoding_time, decoding_speed);
    }
    else {
        printf("Start benchmark: p%dg%d\n", prefilling_tokens, config.max_dec_len);
        int heat = 1, loop = 5;
        float p_speed[loop], d_speed[loop];
        for (int i = 0; i < heat + loop; i++) {
            memset(last_x, 0, config.n_layer * 2 * config.n_embd * sizeof(float));
            memset(wkv_state, 0, config.n_layer * config.n_head * config.head_size * config.head_size * sizeof(float));
            run(
                logits,
                &config, &weights, &tokenizer, &sampler,
                last_x, wkv_state,
                token_list, prefilling_tokens
            );
            if (i >= heat) {
                p_speed[i - heat] = (double)prefilling_tokens / config.prefilling_time * 1000;
                d_speed[i - heat] = (double)config.decoding_tokens / config.decoding_time * 1000;
                printf("%d: loop\n", i);
            }
            else { printf("%d: heat\n", i); }
        }
        float p_speed_mean = vec_sum(p_speed, loop) / loop;
        float d_speed_mean = vec_sum(d_speed, loop) / loop;
        float p_speed_var = 0, d_speed_var = 0;
        for (int i = 0; i < loop; i++) {
            p_speed_var = MAXIMUM(p_speed_var, fabs(p_speed[i] - p_speed_mean));
            d_speed_var = MAXIMUM(d_speed_var, fabs(d_speed[i] - d_speed_mean));
        }

        printf("\n---------\n");
        printf("p%dg%d\n", prefilling_tokens, config.max_dec_len);
        printf("Prefill: %f(±%f) token/s\n", p_speed_mean, p_speed_var);
        printf("Decode: %f(±%f) token/s\n", d_speed_mean, d_speed_var);
    }

    free(token_list);
    free(last_x);
    free(wkv_state);
    free_model(&weights);
    free_tokenizer(&tokenizer);
    free(sampler.occurrence);
}
