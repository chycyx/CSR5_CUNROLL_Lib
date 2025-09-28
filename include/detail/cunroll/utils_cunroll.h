#ifndef UTILS_AVX2_H
#define UTILS_AVX2_H

#include "common_cunroll.h"

struct anonymouslib_timer {
    timeval t1, t2;
    struct timezone tzone;

    void start() {
        gettimeofday(&t1, &tzone);
    }

    double stop() {
        gettimeofday(&t2, &tzone);
        double elapsedTime = 0;
        elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
        elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms
        return elapsedTime;
    }
};

template<typename iT>
iT binary_search_right_boundary_kernel(const iT *d_row_pointer,
                                       const iT  key_input,
                                       const iT  size)
{
    iT start = 0;
    iT stop  = size - 1;
    iT median;
    iT key_median;

    while (stop >= start)
    {
        median = (stop + start) / 2;
        key_median = d_row_pointer[median];
        if (key_input >= key_median)
            start = median + 1;
        else
            stop = median - 1;
    }
    return start;
}

// sum up 4 double-precision numbers
inline double hsum_avx(const double in[4])
{
    double sum = 0;
    for (int i = 0; i < 4; i++)
        sum += in[i];
    return sum;
}

// sum up 8 single-precision numbers
inline float hsum_avx(const float in[8])
{
    float sum = 0;
    for (int i = 0; i < 8; i++)
        sum += in[i];
    return sum;
}

// exclusive scan using a single thread
template<typename T>
void scan_single(T *s_scan, const int l)
{
    T old_val = s_scan[0];
    s_scan[0] = 0;
    for (int i = 1; i < l; i++)
    {
        T new_val = s_scan[i];
        s_scan[i] = old_val + s_scan[i-1];
        old_val = new_val;
    }
}

// inclusive prefix-sum scan (double)
inline void hscan_avx(const double in[4], double out[4])
{
    out[0] = in[0];
    for (int i = 1; i < 4; i++)
        out[i] = out[i-1] + in[i];
}

// inclusive prefix-sum scan (float)
inline void hscan_avx(const float in[8], float out[8])
{
    out[0] = in[0];
    for (int i = 1; i < 8; i++)
        out[i] = out[i-1] + in[i];
}
//inclusive prefix-sum scan 
inline __m256 hscan_avx(__m256 in256)
{
    __m256 t0, t1;
    //shift1_AVX + add
    t0 = _mm256_permute_ps(in256, _MM_SHUFFLE(2, 1, 0, 3));
    t1 = _mm256_permute2f128_ps(t0, t0, 41);
    in256 = _mm256_add_ps(in256, _mm256_blend_ps(t0, t1, 0x11));
    //shift2_AVX + add
    t0 = _mm256_permute_ps(in256, _MM_SHUFFLE(1, 0, 3, 2));
    t1 = _mm256_permute2f128_ps(t0, t0, 41);
    in256 = _mm256_add_ps(in256, _mm256_blend_ps(t0, t1, 0x33));
    //shift3_AVX + add
    in256 = _mm256_add_ps(in256, _mm256_permute2f128_ps(in256, in256, 41));
    return in256;
}
inline __m256d hscan_avx(__m256d in256d)
{
    __m256d t0, t1;
    t0 = _mm256_permute4x64_pd(in256d, 0x93);
    t1 = _mm256_add_pd(in256d, _mm256_blend_pd(t0, _mm256_set1_pd(0), 0x1));

    t0 = _mm256_permute4x64_pd(in256d, 0x4E);
    t1 = _mm256_add_pd(t1, _mm256_blend_pd(t0, _mm256_set1_pd(0), 0x3));

    t0 = _mm256_permute4x64_pd(in256d, 0x39);
    t1 = _mm256_add_pd(t1, _mm256_blend_pd(t0, _mm256_set1_pd(0), 0x7));

    return t1;
}

#endif // UTILS_AVX2_H