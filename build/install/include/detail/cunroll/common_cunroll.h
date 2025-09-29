#ifndef COMMON_CUNROLL_H
#define COMMON_CUNROLL_H

#include <stdint.h>
#include <string.h>

#include <omp.h>
#include "immintrin.h"

#include "../common.h"
#include "../utils.h"

#define ANONYMOUSLIB_CSR5_OMEGA      4
#define ANONYMOUSLIB_CSR5_SIGMA      16
#define ANONYMOUSLIB_X86_CACHELINE   64

/********************
 * 1. SET
 ********************/
#define C_setzero_pd(reg)        \
    do {                         \
        (reg)[0] = 0.0;          \
        (reg)[1] = 0.0;          \
        (reg)[2] = 0.0;          \
        (reg)[3] = 0.0;          \
    } while (0)

#define C_setzero_si256_i32(reg) \
    do {                         \
        (reg)[0] = 0;            \
        (reg)[1] = 0;            \
        (reg)[2] = 0;            \
        (reg)[3] = 0;            \
        (reg)[4] = 0;            \
        (reg)[5] = 0;            \
        (reg)[6] = 0;            \
        (reg)[7] = 0;            \
    } while (0)

#define C_setzero_si256_i64(reg) \
    do {                         \
        (reg)[0] = 0;            \
        (reg)[1] = 0;            \
        (reg)[2] = 0;            \
        (reg)[3] = 0;            \
    } while (0)

#define C_set_pd(dst, e3, e2, e1, e0) \
    do {                              \
        (dst)[0] = (e0);              \
        (dst)[1] = (e1);              \
        (dst)[2] = (e2);              \
        (dst)[3] = (e3);              \
    } while (0)

#define C_set1_epi64x(reg, value) \
    do {                          \
        (reg)[0] = (value);       \
        (reg)[1] = (value);       \
        (reg)[2] = (value);       \
        (reg)[3] = (value);       \
    } while (0)

#define C_set_epi64x(dst, e3, e2, e1, e0) \
    do {                                  \
        (dst)[0] = (e0);                  \
        (dst)[1] = (e1);                  \
        (dst)[2] = (e2);                  \
        (dst)[3] = (e3);                  \
    } while (0)

/********************
 * 1.5. CAST
 ********************/
static inline void C_castps256_ps128(float dst128[4], const float src256[8]) {
    dst128[0] = src256[0];
    dst128[1] = src256[1];
    dst128[2] = src256[2];
    dst128[3] = src256[3];
}

/* 保留你原本的两个重载名（若用 C++ 编译器可重载；若用 C 编译请保留其中一个或改名） */
static inline void C_castsi256_pd(double dst[4], const uint64_t src[4]) {
    union { uint64_t i; double d; } u;
    u.i = src[0]; dst[0] = u.d;
    u.i = src[1]; dst[1] = u.d;
    u.i = src[2]; dst[2] = u.d;
    u.i = src[3]; dst[3] = u.d;
}
static inline void C_castsi256_pd(double dst[4], const int64_t src[4]) {
    union { int64_t i; double d; } u;
    u.i = src[0]; dst[0] = u.d;
    u.i = src[1]; dst[1] = u.d;
    u.i = src[2]; dst[2] = u.d;
    u.i = src[3]; dst[3] = u.d;
}

static inline void C_castpd_si256(int64_t dst[4], const double src[4]) {
    union { double d; int64_t i; } u;
    u.d = src[0]; dst[0] = u.i;
    u.d = src[1]; dst[1] = u.i;
    u.d = src[2]; dst[2] = u.i;
    u.d = src[3]; dst[3] = u.i;
}

static inline void castsi128_si256(int dst256[8], const int src128[4]) {
    dst256[0] = src128[0];
    dst256[1] = src128[1];
    dst256[2] = src128[2];
    dst256[3] = src128[3];
    dst256[4] = 0;
    dst256[5] = 0;
    dst256[6] = 0;
    dst256[7] = 0;
}

/********************
 * 2. LOAD / STORE
 ********************/
#define C_store_pd(mem, reg)     \
    do {                         \
        (mem)[0] = (reg)[0];     \
        (mem)[1] = (reg)[1];     \
        (mem)[2] = (reg)[2];     \
        (mem)[3] = (reg)[3];     \
    } while (0)

#define C_store_si128_i32(mem, reg) \
    do {                            \
        (mem)[0] = (reg)[0];        \
        (mem)[1] = (reg)[1];        \
        (mem)[2] = (reg)[2];        \
        (mem)[3] = (reg)[3];        \
    } while (0)

#define C_store_si128_i64(mem, reg) \
    do {                            \
        (mem)[0] = (reg)[0];        \
        (mem)[1] = (reg)[1];        \
    } while (0)

#define C_store_si256_i32(mem, reg) \
    do {                            \
        (mem)[0] = (reg)[0];        \
        (mem)[1] = (reg)[1];        \
        (mem)[2] = (reg)[2];        \
        (mem)[3] = (reg)[3];        \
        (mem)[4] = (reg)[4];        \
        (mem)[5] = (reg)[5];        \
        (mem)[6] = (reg)[6];        \
        (mem)[7] = (reg)[7];        \
    } while (0)

#define C_store_si256_i64(mem, reg) \
    do {                            \
        (mem)[0] = (reg)[0];        \
        (mem)[1] = (reg)[1];        \
        (mem)[2] = (reg)[2];        \
        (mem)[3] = (reg)[3];        \
    } while (0)

#define C_load_pd(reg, mem)      \
    do {                         \
        (reg)[0] = (mem)[0];     \
        (reg)[1] = (mem)[1];     \
        (reg)[2] = (mem)[2];     \
        (reg)[3] = (mem)[3];     \
    } while (0)

#define C_load_si128_i32(reg, mem) \
    do {                           \
        (reg)[0] = (mem)[0];       \
        (reg)[1] = (mem)[1];       \
        (reg)[2] = (mem)[2];       \
        (reg)[3] = (mem)[3];       \
    } while (0)

/********************
 * 3. GATHER
 ********************/
#define C_i32gather_epi32(reg, base_addr, idx, scale) \
    do {                                              \
        const char* _base = (const char*)(base_addr); \
        (reg)[0] = *(const int32_t*)(_base + (idx)[0] * (scale)); \
        (reg)[1] = *(const int32_t*)(_base + (idx)[1] * (scale)); \
        (reg)[2] = *(const int32_t*)(_base + (idx)[2] * (scale)); \
        (reg)[3] = *(const int32_t*)(_base + (idx)[3] * (scale)); \
    } while (0)

/********************
 * 4. ARITHMETIC
 ********************/
#define C_add_epi32(reg, a, b)  \
    do {                        \
        (reg)[0] = (a)[0] + (b)[0]; \
        (reg)[1] = (a)[1] + (b)[1]; \
        (reg)[2] = (a)[2] + (b)[2]; \
        (reg)[3] = (a)[3] + (b)[3]; \
        (reg)[4] = (a)[4] + (b)[4]; \
        (reg)[5] = (a)[5] + (b)[5]; \
        (reg)[6] = (a)[6] + (b)[6]; \
        (reg)[7] = (a)[7] + (b)[7]; \
    } while (0)

#define C_add_epi64(reg, a, b)  \
    do {                        \
        (reg)[0] = (a)[0] + (b)[0]; \
        (reg)[1] = (a)[1] + (b)[1]; \
        (reg)[2] = (a)[2] + (b)[2]; \
        (reg)[3] = (a)[3] + (b)[3]; \
    } while (0)

#define C_add_epi32_128(reg, a, b) \
    do {                           \
        (reg)[0] = (a)[0] + (b)[0]; \
        (reg)[1] = (a)[1] + (b)[1]; \
        (reg)[2] = (a)[2] + (b)[2]; \
        (reg)[3] = (a)[3] + (b)[3]; \
    } while (0)

#define C_add_pd(reg, a, b)     \
    do {                        \
        (reg)[0] = (a)[0] + (b)[0]; \
        (reg)[1] = (a)[1] + (b)[1]; \
        (reg)[2] = (a)[2] + (b)[2]; \
        (reg)[3] = (a)[3] + (b)[3]; \
    } while (0)

#define C_mul_pd_restrict(reg, a, b) \
    do {                             \
        (reg)[0] = (a)[0] * (b)[0];  \
        (reg)[1] = (a)[1] * (b)[1];  \
        (reg)[2] = (a)[2] * (b)[2];  \
        (reg)[3] = (a)[3] * (b)[3];  \
    } while (0)

#define C_sub_epi64(reg, a, b)  \
    do {                        \
        (reg)[0] = (a)[0] - (b)[0]; \
        (reg)[1] = (a)[1] - (b)[1]; \
        (reg)[2] = (a)[2] - (b)[2]; \
        (reg)[3] = (a)[3] - (b)[3]; \
    } while (0)

#define C_fmadd_pd(reg, a, b, c)    \
    do {                            \
        (reg)[0] = (a)[0] * (b)[0] + (c)[0]; \
        (reg)[1] = (a)[1] * (b)[1] + (c)[1]; \
        (reg)[2] = (a)[2] * (b)[2] + (c)[2]; \
        (reg)[3] = (a)[3] * (b)[3] + (c)[3]; \
    } while (0)

#define C_add_ps(dst, a, b) \
    do {                    \
        (dst)[0] = (a)[0] + (b)[0]; \
        (dst)[1] = (a)[1] + (b)[1]; \
        (dst)[2] = (a)[2] + (b)[2]; \
        (dst)[3] = (a)[3] + (b)[3]; \
        (dst)[4] = (a)[4] + (b)[4]; \
        (dst)[5] = (a)[5] + (b)[5]; \
        (dst)[6] = (a)[6] + (b)[6]; \
        (dst)[7] = (a)[7] + (b)[7]; \
    } while (0)

#define C_sub_pd(reg, a, b)     \
    do {                        \
        (reg)[0] = (a)[0] - (b)[0]; \
        (reg)[1] = (a)[1] - (b)[1]; \
        (reg)[2] = (a)[2] - (b)[2]; \
        (reg)[3] = (a)[3] - (b)[3]; \
    } while (0)

/********************
 * 5. LOGICAL
 ********************/
#define C_and_si256(reg, a, b)  \
    do {                        \
        (reg)[0] = (a)[0] & (b)[0]; \
        (reg)[1] = (a)[1] & (b)[1]; \
        (reg)[2] = (a)[2] & (b)[2]; \
        (reg)[3] = (a)[3] & (b)[3]; \
    } while (0)

#define C_or_si128(reg, a, b)   \
    do {                        \
        (reg)[0] = (a)[0] | (b)[0]; \
        (reg)[1] = (a)[1] | (b)[1]; \
        (reg)[2] = (a)[2] | (b)[2]; \
        (reg)[3] = (a)[3] | (b)[3]; \
    } while (0)

#define C_or_si256(dst, a, b)   \
    do {                        \
        (dst)[0] = (a)[0] | (b)[0]; \
        (dst)[1] = (a)[1] | (b)[1]; \
        (dst)[2] = (a)[2] | (b)[2]; \
        (dst)[3] = (a)[3] | (b)[3]; \
    } while (0)

static inline int C_testz_si256(const uint64_t a[4], const uint64_t b[4]) {
    /* 等价于循环：若 (a[i] & b[i]) 全为 0 则返回 1，否则 0 */
    return (((a[0] & b[0]) | (a[1] & b[1]) | (a[2] & b[2]) | (a[3] & b[3])) == 0);
}

static inline void C_and_pd(double dst[4], const double a[4], const double b[4]) {
    uint64_t ai, bi, di;
    memcpy(&ai, &a[0], sizeof(ai)); memcpy(&bi, &b[0], sizeof(bi)); di = ai & bi; memcpy(&dst[0], &di, sizeof(di));
    memcpy(&ai, &a[1], sizeof(ai)); memcpy(&bi, &b[1], sizeof(bi)); di = ai & bi; memcpy(&dst[1], &di, sizeof(di));
    memcpy(&ai, &a[2], sizeof(ai)); memcpy(&bi, &b[2], sizeof(bi)); di = ai & bi; memcpy(&dst[2], &di, sizeof(di));
    memcpy(&ai, &a[3], sizeof(ai)); memcpy(&bi, &b[3], sizeof(bi)); di = ai & bi; memcpy(&dst[3], &di, sizeof(di));
}

#define C_andnot_si256(dst, a, b) \
    do {                          \
        (dst)[0] = (~(a)[0]) & (b)[0]; \
        (dst)[1] = (~(a)[1]) & (b)[1]; \
        (dst)[2] = (~(a)[2]) & (b)[2]; \
        (dst)[3] = (~(a)[3]) & (b)[3]; \
    } while (0)

/********************
 * 6. SHIFT
 ********************/
#define C_srli_epi32(reg, a, imm)        \
    do {                                  \
        (reg)[0] = (int32_t)((uint32_t)(a)[0] >> (imm)); \
        (reg)[1] = (int32_t)((uint32_t)(a)[1] >> (imm)); \
        (reg)[2] = (int32_t)((uint32_t)(a)[2] >> (imm)); \
        (reg)[3] = (int32_t)((uint32_t)(a)[3] >> (imm)); \
    } while (0)

#define C_slli_epi32(reg, a, imm)        \
    do {                                  \
        (reg)[0] = (int32_t)((uint32_t)(a)[0] << (imm)); \
        (reg)[1] = (int32_t)((uint32_t)(a)[1] << (imm)); \
        (reg)[2] = (int32_t)((uint32_t)(a)[2] << (imm)); \
        (reg)[3] = (int32_t)((uint32_t)(a)[3] << (imm)); \
    } while (0)

/********************
 * 7. CONVERT
 ********************/
#define C_cvtepu32_epi64(reg, a) \
    do {                         \
        (reg)[0] = (uint64_t)(a)[0]; \
        (reg)[1] = (uint64_t)(a)[1]; \
        (reg)[2] = (uint64_t)(a)[2]; \
        (reg)[3] = (uint64_t)(a)[3]; \
    } while (0)

/********************
 * 8. COMPARE
 ********************/
#define C_cmpeq_epi64(reg, a, b)                            \
    do {                                                    \
        (reg)[0] = ((a)[0] == (b)[0]) ? 0xFFFFFFFFFFFFFFFFULL : 0ULL; \
        (reg)[1] = ((a)[1] == (b)[1]) ? 0xFFFFFFFFFFFFFFFFULL : 0ULL; \
        (reg)[2] = ((a)[2] == (b)[2]) ? 0xFFFFFFFFFFFFFFFFULL : 0ULL; \
        (reg)[3] = ((a)[3] == (b)[3]) ? 0xFFFFFFFFFFFFFFFFULL : 0ULL; \
    } while (0)

#define C_cmpgt_epi64(reg, a, b)                            \
    do {                                                    \
        (reg)[0] = ((a)[0] > (b)[0]) ? 0xFFFFFFFFFFFFFFFFULL : 0ULL; \
        (reg)[1] = ((a)[1] > (b)[1]) ? 0xFFFFFFFFFFFFFFFFULL : 0ULL; \
        (reg)[2] = ((a)[2] > (b)[2]) ? 0xFFFFFFFFFFFFFFFFULL : 0ULL; \
        (reg)[3] = ((a)[3] > (b)[3]) ? 0xFFFFFFFFFFFFFFFFULL : 0ULL; \
    } while (0)

/********************
 * 9. SWIZZLE
 ********************/
#define C_permutevar8x32_epi32(dst, src, index) \
    do {                                        \
        (dst)[0] = (src)[(index)[0]];           \
        (dst)[1] = (src)[(index)[1]];           \
        (dst)[2] = (src)[(index)[2]];           \
        (dst)[3] = (src)[(index)[3]];           \
        (dst)[4] = (src)[(index)[4]];           \
        (dst)[5] = (src)[(index)[5]];           \
        (dst)[6] = (src)[(index)[6]];           \
        (dst)[7] = (src)[(index)[7]];           \
    } while (0)

/*原样复制 */
#define C_permute_ps(dst, in, imm8) \
    do {                            \
        (void)(imm8);               \
        (dst)[0] = (in)[0];         \
        (dst)[1] = (in)[1];         \
        (dst)[2] = (in)[2];         \
        (dst)[3] = (in)[3];         \
        (dst)[4] = (in)[4];         \
        (dst)[5] = (in)[5];         \
        (dst)[6] = (in)[6];         \
        (dst)[7] = (in)[7];         \
    } while (0)

#define C_blend_pd(dst, a, b, imm8) \
    do {                            \
        (dst)[0] = (((unsigned)(imm8) >> 0) & 1) ? (b)[0] : (a)[0]; \
        (dst)[1] = (((unsigned)(imm8) >> 1) & 1) ? (b)[1] : (a)[1]; \
        (dst)[2] = (((unsigned)(imm8) >> 2) & 1) ? (b)[2] : (a)[2]; \
        (dst)[3] = (((unsigned)(imm8) >> 3) & 1) ? (b)[3] : (a)[3]; \
    } while (0)

static inline void C_blendv_pd(double dst[4], const double a[4], const double b[4], const int64_t mask[4]) {
    /* 依据 mask 的符号位（最高位）选择 */
    int64_t msb;
    msb = (mask[0] >> 63) & 1; dst[0] = msb ? b[0] : a[0];
    msb = (mask[1] >> 63) & 1; dst[1] = msb ? b[1] : a[1];
    msb = (mask[2] >> 63) & 1; dst[2] = msb ? b[2] : a[2];
    msb = (mask[3] >> 63) & 1; dst[3] = msb ? b[3] : a[3];
}


static inline void C_permute2f128_ps(float dst[8], const float a[8],
                                     const float b[8], unsigned imm8)
{
    int sel_low  =  imm8        & 0x3; /* 低 128-bit 来源选择 */
    int sel_high = (imm8 >> 4)  & 0x3; /* 高 128-bit 来源选择 */

    /* low 128-bit */
    if (imm8 & 0x08) {
        dst[0] = 0.0f; dst[1] = 0.0f; dst[2] = 0.0f; dst[3] = 0.0f;
    } else {
        const float *src = (sel_low < 2) ? a : b;
        int base = (sel_low & 1) ? 4 : 0;
        dst[0] = src[base + 0];
        dst[1] = src[base + 1];
        dst[2] = src[base + 2];
        dst[3] = src[base + 3];
    }

    /* high 128-bit */
    if (imm8 & 0x80) {
        dst[4] = 0.0f; dst[5] = 0.0f; dst[6] = 0.0f; dst[7] = 0.0f;
    } else {
        const float *src = (sel_high < 2) ? a : b;
        int base = (sel_high & 1) ? 4 : 0;
        dst[4] = src[base + 0];
        dst[5] = src[base + 1];
        dst[6] = src[base + 2];
        dst[7] = src[base + 3];
    }
}

static inline void C_permute4x64_pd(double dst[4], const double src[4], unsigned imm8)
{
    unsigned sel;
    sel = (imm8 >> 0) & 0x3; dst[0] = src[sel];
    sel = (imm8 >> 2) & 0x3; dst[1] = src[sel];
    sel = (imm8 >> 4) & 0x3; dst[2] = src[sel];
    sel = (imm8 >> 6) & 0x3; dst[3] = src[sel];
}

#define C_blend_ps(dst, a, b, imm8) \
    do {                            \
        (dst)[0] = (((unsigned)(imm8) >> 0) & 1) ? (b)[0] : (a)[0]; \
        (dst)[1] = (((unsigned)(imm8) >> 1) & 1) ? (b)[1] : (a)[1]; \
        (dst)[2] = (((unsigned)(imm8) >> 2) & 1) ? (b)[2] : (a)[2]; \
        (dst)[3] = (((unsigned)(imm8) >> 3) & 1) ? (b)[3] : (a)[3]; \
        (dst)[4] = (((unsigned)(imm8) >> 4) & 1) ? (b)[4] : (a)[4]; \
        (dst)[5] = (((unsigned)(imm8) >> 5) & 1) ? (b)[5] : (a)[5]; \
        (dst)[6] = (((unsigned)(imm8) >> 6) & 1) ? (b)[6] : (a)[6]; \
        (dst)[7] = (((unsigned)(imm8) >> 7) & 1) ? (b)[7] : (a)[7]; \
    } while (0)

#endif 