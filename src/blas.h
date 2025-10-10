#ifndef __BLAS_H__
#define __BLAS_H__
#include <cblas.h>

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
    // cblas_scopy(len, a, 1, xout, 1)
    // cblas_sscal(len, b, a, 1);
    for (int i = 0; i < len; i++) { xout[i] = a[i] * b; }
}

static inline float vec_dot_product(const float *a, const float *b, int len) {
    return cblas_sdot(len, a, 1, b, 1);
}

void vec_out_product(float *xout, const float *a, const float *b, int len) {
    // int M = len, K = 1, N = len;
    // cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, a, K, b, N, 0.0, xout, N);
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

void mat_mul_vec(float *xout, const float *x, const float *w, int x_len, int xout_len, int seq_len) {
    if (seq_len == 1) {
        int M = xout_len, N = x_len;
        cblas_sgemv(CblasRowMajor, CblasNoTrans, M, N, 1.0, w, N, x, 1, 0.0, xout, 1);
    }
    else {
        int M = seq_len, K = x_len, N = xout_len;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0, x, K, w, K, 0.0, xout, N);
    }
}

#endif
