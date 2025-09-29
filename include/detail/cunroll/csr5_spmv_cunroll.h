
#ifndef CSR5_SPMV_CUNROLL_H
#define CSR5_SPMV_CUNROLL_H

#include "common_cunroll.h"
#include "utils_cunroll.h"

template<typename iT, typename vT>
int scale_y(const iT   m,
            const vT   beta,
            vT        *d_y)
{
    if (beta == (vT)0.0) {
        memset(d_y, 0, sizeof(vT) * m);
    }
    else if (beta != (vT)1.0) {
        #pragma omp parallel for
        for (iT i = 0; i < m; i++) {
            d_y[i] = beta * d_y[i];
        }
    }

    return ANONYMOUSLIB_SUCCESS;
}


template<typename iT, typename vT>
inline void partition_fast_track(const vT           *d_value_partition,
                                 const vT           *d_x,
                                 const iT           *d_column_index_partition,
                                 vT                 *d_calibrator,
                                 vT                 *d_y,
                                 const iT            row_start,
                                 const iT            par_id,
                                 const int           tid,
                                 const iT            start_row_start,
                                 const vT            alpha,
                                 const int           sigma,
                                 const int           stride_vT,
                                 const bool          direct)
{
    double value256d[4], x256d[4], sum256d[4];
    double x0, x1, x2, x3;

    C_setzero_pd(sum256d);

    for (int i = 0; i < ANONYMOUSLIB_CSR5_SIGMA; i++)
    {
        // load values
        C_load_pd(value256d, &d_value_partition[i * ANONYMOUSLIB_CSR5_OMEGA]);

        x0 = d_x[d_column_index_partition[i * ANONYMOUSLIB_CSR5_OMEGA]];
        x1 = d_x[d_column_index_partition[i * ANONYMOUSLIB_CSR5_OMEGA + 1]];
        x2 = d_x[d_column_index_partition[i * ANONYMOUSLIB_CSR5_OMEGA + 2]];
        x3 = d_x[d_column_index_partition[i * ANONYMOUSLIB_CSR5_OMEGA + 3]];

        C_set_pd(x256d, x3, x2, x1, x0);

        C_fmadd_pd(sum256d, value256d, x256d, sum256d);
    }

    // 水平求和
    vT sum = sum256d[0] + sum256d[1] + sum256d[2] + sum256d[3];

    if (row_start == start_row_start && !direct)
        d_calibrator[tid * stride_vT] += sum* alpha;
    else {
        if (direct)
            //d_y[row_start] = sum * alpha;
            d_y[row_start] += sum* alpha;
        else
            d_y[row_start] += sum* alpha;
    }
}
// ...existing code...
template<typename iT, typename uiT, typename vT>
void spmv_csr5_compute_kernel(const iT           *d_column_index,
                              const vT           *d_value,
                              const iT           *d_row_pointer,
                              const vT           *d_x,
                              const uiT          *d_partition_pointer,
                              const uiT          *d_partition_descriptor,
                              const iT           *d_partition_descriptor_offset_pointer,
                              const iT           *d_partition_descriptor_offset,
                              vT                 *d_calibrator,
                              vT                 *d_y,
                              const iT            p,
                              const int           num_packet,
                              const int           bit_y_offset,
                              const int           bit_scansum_offset,
                              const vT            alpha,
                              const int           c_sigma)
{
    const int num_thread = omp_get_max_threads();
    const int chunk = ceil((double)(p-1) / (double)num_thread);
    const int stride_vT = ANONYMOUSLIB_X86_CACHELINE / sizeof(vT);
    const int num_thread_active = ceil((p-1.0)/chunk);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        iT start_row_start = tid < num_thread_active ? d_partition_pointer[tid * chunk] & 0x7FFFFFFF : 0;

        vT  s_sum[8]; // allocate a cache line
        vT  s_first_sum[8]; // allocate a cache line
        uint64_t s_cond[8]; // allocate a cache line
        int s_y_idx[16]; // allocate a cache line

        int inc0, inc1, inc2, inc3;
        vT x256d0, x256d1, x256d2, x256d3;

double  C_sum256d[4], C_tmp_sum256d[4], C_first_sum256d[4], C_last_sum256d[4];
double  C_value256d[4], C_x256d[4];

int32_t C_scansum_offset128i[4], C_y_offset128i[4], C_y_idx128i[4], C_descriptor128i[4];

int64_t C_start256i[4], C_stop256i[4];
int64_t C_local_bit256i[4];
int64_t C_direct256i[4];
int64_t C_tmp256i_64[4];

C_setzero_pd(C_sum256d);
C_setzero_pd(C_tmp_sum256d);
C_setzero_pd(C_first_sum256d);
C_setzero_pd(C_last_sum256d);
C_setzero_pd(C_value256d);
C_setzero_pd(C_x256d);
C_setzero_si256_i64(C_start256i);
C_setzero_si256_i64(C_stop256i);
C_setzero_si256_i64(C_local_bit256i);
C_setzero_si256_i64(C_direct256i);
C_setzero_si256_i64(C_tmp256i_64);
            int32_t *d_column_index_partition128i;
            int32_t *d_partition_descriptor128i;

#pragma omp for schedule(static, chunk)
for (int par_id = 0; par_id < p - 1; par_id++)
{
    const iT *d_column_index_partition = &d_column_index[par_id * ANONYMOUSLIB_CSR5_OMEGA * c_sigma];
    const vT *d_value_partition        = &d_value[par_id * ANONYMOUSLIB_CSR5_OMEGA * c_sigma];

    uiT row_start     = d_partition_pointer[par_id];
    const iT row_stop = d_partition_pointer[par_id + 1] & 0x7FFFFFFF;

    if (row_start == row_stop) // fast track
    {
        bool fast_direct = (d_partition_descriptor[par_id * ANONYMOUSLIB_CSR5_OMEGA * num_packet] >>
                            (31 - (bit_y_offset + bit_scansum_offset))) & 0x1;

        partition_fast_track<iT, vT>(
            d_value_partition, d_x, d_column_index_partition,
            d_calibrator, d_y, row_start, par_id, tid,
            start_row_start, alpha, c_sigma, stride_vT, fast_direct
        );
    }
    else // normal track
    {
        const bool empty_rows = (row_start >> 31) & 0x1;
        row_start &= 0x7FFFFFFF;

        vT *d_y_local = &d_y[row_start + 1];
        const int offset_pointer = empty_rows ? d_partition_descriptor_offset_pointer[par_id] : 0;
                
        // d_column_index_partition128i = (__m128i *)d_column_index_partition;
        // d_partition_descriptor128i  = (__m128i *)&d_partition_descriptor[par_id * ANONYMOUSLIB_CSR5_OMEGA * num_packet];
        d_column_index_partition128i = (int32_t *)&d_column_index_partition;
        d_partition_descriptor128i  = (int32_t *)&d_partition_descriptor[par_id * ANONYMOUSLIB_CSR5_OMEGA * num_packet];

        C_setzero_pd(C_first_sum256d);
        C_setzero_si256_i64(C_stop256i);
        C_load_si128_i32(C_descriptor128i, (int32_t*)d_partition_descriptor128i);
        C_srli_epi32(C_y_offset128i, C_descriptor128i, 32 - bit_y_offset);
        C_slli_epi32(C_scansum_offset128i, C_descriptor128i, bit_y_offset);
        C_srli_epi32(C_scansum_offset128i, C_scansum_offset128i, 32 - bit_scansum_offset);
        C_slli_epi32(C_descriptor128i, C_descriptor128i, bit_y_offset + bit_scansum_offset);

        int32_t tmp32[4];
        C_srli_epi32(tmp32, C_descriptor128i, 31);
        C_cvtepu32_epi64(C_local_bit256i, (uint32_t*)tmp32);
        bool first_direct = false;   
        C_store_si256_i64((int64_t*)s_cond, C_local_bit256i);
                if(s_cond[0])
                    first_direct = true;
                     
        bool first_all_direct = false;
        if (par_id == tid * chunk)
            first_all_direct = first_direct;

        // descriptor128i |= 0x80000000
        int32_t tmp128[4] = {INT32_MIN,0, 0, 0};
        C_or_si128(C_descriptor128i, C_descriptor128i, tmp128);

        // recompute local_bit256i after OR
        C_srli_epi32(tmp32, C_descriptor128i, 31);
        C_cvtepu32_epi64(C_local_bit256i, (uint32_t*)tmp32);

        // start256i = 1 - local_bit256i
        C_set1_epi64x(C_start256i, 0x1);
        C_sub_epi64(C_start256i, C_start256i, C_local_bit256i);

        // direct256i = local_bit256i & [1,1,1,0]
        C_set_epi64x((uint64_t*)C_direct256i, 0x1, 0x1, 0x1, 0);
        C_and_si256((uint64_t*)C_direct256i, (uint64_t*)C_direct256i, (uint64_t*)C_local_bit256i);

        // load value256d
        C_load_pd(C_value256d, d_value_partition);

        // load x256d
        C_set_pd(C_x256d,
                 d_x[d_column_index_partition[3]],
                 d_x[d_column_index_partition[2]],
                 d_x[d_column_index_partition[1]],
                 d_x[d_column_index_partition[0]]);

        // sum256d = value256d * x256d
        C_mul_pd_restrict(C_sum256d, C_value256d, C_x256d);
        // step 1. thread-level seg sum
#if ANONYMOUSLIB_CSR5_SIGMA > 23
    int ly = 0;
#endif
uint64_t C_mask_one64[4] = {1ULL, 1ULL, 1ULL, 1ULL};
uint64_t C_mask_all1[4] = {
    0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL
};

    for (int i = 1; i < ANONYMOUSLIB_CSR5_SIGMA; i++)
    {
        double x256d0 = d_x[d_column_index_partition[i * ANONYMOUSLIB_CSR5_OMEGA]];
        double x256d1 = d_x[d_column_index_partition[i * ANONYMOUSLIB_CSR5_OMEGA + 1]];
        double x256d2 = d_x[d_column_index_partition[i * ANONYMOUSLIB_CSR5_OMEGA + 2]];
        double x256d3 = d_x[d_column_index_partition[i * ANONYMOUSLIB_CSR5_OMEGA + 3]];
        C_set_pd(C_x256d, x256d3, x256d2, x256d1, x256d0);

#if ANONYMOUSLIB_CSR5_SIGMA > 23
        int norm_i = i - (32 - bit_y_offset - bit_scansum_offset);

        if (!(ly || norm_i) || (ly && !(norm_i % 32)))
        {
            ly++;
            C_load_si128_i32((int32_t*)C_descriptor128i, (int32_t*)&d_partition_descriptor128i[ly]);
        }
        norm_i = !ly ? i : norm_i;
        norm_i = 31 - norm_i % 32;

        int32_t C_tmp128[4];
        C_srli_epi32(C_tmp128, (int32_t*)C_descriptor128i, norm_i);
        C_cvtepu32_epi64((int64_t*)C_tmp256i_64, (uint32_t*)C_tmp128);
        C_and_si256((uint64_t*)C_local_bit256i, (uint64_t*)C_tmp256i_64, (uint64_t*)C_mask_one64);
#else
        int32_t C_tmp128[4];
        C_srli_epi32(C_tmp128, (int32_t*)C_descriptor128i, 31 - i);
        C_cvtepu32_epi64((int64_t*)C_tmp256i_64, (uint32_t*)C_tmp128);
        C_and_si256((uint64_t*)C_local_bit256i, (uint64_t*)C_tmp256i_64, (uint64_t*)C_mask_one64);
#endif

        int store_to_offchip = C_testz_si256((uint64_t*)C_local_bit256i, (uint64_t*)C_mask_all1);

        if (!store_to_offchip)
        {
            if (empty_rows) {
                C_i32gather_epi32((int32_t*)C_y_idx128i,
                                  &d_partition_descriptor_offset[offset_pointer],
                                  (int32_t*)C_y_offset128i,
                                  4);
            } else {
                for (int jj = 0; jj < 4; ++jj) C_y_idx128i[jj] = C_y_offset128i[jj];
            }

            C_store_si128_i32(s_y_idx, (int32_t*)C_y_idx128i);
            C_store_pd(s_sum, C_sum256d);

            C_and_si256((uint64_t*)C_tmp256i_64, (uint64_t*)C_direct256i, (uint64_t*)C_local_bit256i);
            C_store_si256_i64((int64_t*)s_cond, (int64_t*)C_tmp256i_64);

            inc0 = inc1 = inc2 = inc3 = 0;
            if (s_cond[0]) { d_y_local[s_y_idx[0]] += s_sum[0]* alpha; inc0 = 1; }
            if (s_cond[1]) { d_y_local[s_y_idx[1]] += s_sum[1]* alpha; inc1 = 1; }
            if (s_cond[2]) { d_y_local[s_y_idx[2]] += s_sum[2]* alpha; inc2 = 1; }
            if (s_cond[3]) { d_y_local[s_y_idx[3]] += s_sum[3]* alpha; inc3 = 1; }

            int32_t C_inc128[4] = { inc0, inc1, inc2, inc3 };
            C_add_epi32_128((int32_t*)C_y_offset128i, (int32_t*)C_y_offset128i, C_inc128);
            int64_t C_cmp_direct_eq1[4];
            int64_t C_cmp_local_eq1[4];

            C_cmpeq_epi64((uint64_t*)C_cmp_direct_eq1, (int64_t*)C_direct256i, (int64_t*)C_mask_one64);
            C_cmpeq_epi64((uint64_t*)C_cmp_local_eq1, (int64_t*)C_local_bit256i, (int64_t*)C_mask_one64);

            for (int jj = 0; jj < 4; ++jj) {
                C_tmp256i_64[jj] = (~((uint64_t)C_cmp_direct_eq1[jj])) & ((uint64_t)C_cmp_local_eq1[jj]);
            }

            for (int jj = 0; jj < 4; ++jj) {
                if (C_tmp256i_64[jj] == 0)
                    C_first_sum256d[jj] = C_first_sum256d[jj];
                else
                    C_first_sum256d[jj] = C_sum256d[jj];
            }

            for (int jj = 0; jj < 4; ++jj) {
                if (C_local_bit256i[jj] != 0)
                    C_sum256d[jj] = 0.0;
            }

            for (int jj = 0; jj < 4; ++jj) {
                C_direct256i[jj] |= C_local_bit256i[jj];
                C_stop256i[jj]   += C_local_bit256i[jj];
            }
        }
        C_load_pd(C_value256d, &d_value_partition[i * ANONYMOUSLIB_CSR5_OMEGA]);
        C_fmadd_pd(C_sum256d, C_value256d, C_x256d, C_sum256d);
}
                int64_t C_ones64[4];
                int64_t C_zeros64[4];
                int64_t C_tmp256i_b[4];
                int64_t C_tmp256i_c[4];
                int64_t C_tmp256i_d[4];
                int32_t tmp256i32_128[8];
                int32_t tmp256i32_perm1[8];
                int32_t tmp256i32_perm2[8];
                int32_t tmp256i32_perm3[8];
                int32_t idx_perm_pattern[8] = {0,0,1,1,2,2,3,3};
                int32_t add0101[8] = {0,1,0,1,0,1,0,1};
                int32_t add3210[4] = {0,1,2,3};
                int32_t cast_si32[8];
                int64_t cast_si64[4];
                double tmp_pd_mask[4];
                double tmp_pd_cast[4];
                double tmp_pd_and[4];
                double tmp_pd_and2[4];

                // set constants
                C_set1_epi64x(C_ones64, 1);   // {1,1,1,1}
                C_set1_epi64x(C_zeros64, 0);  // {0,0,0,0}

                // tmp256i = _mm256_cmpeq_epi64(direct256i, set1(1));
                C_cmpeq_epi64((uint64_t*)C_tmp256i_64, C_direct256i, C_ones64);

                // first_sum256d = _mm256_and_pd(_mm256_castsi256_pd(tmp256i), first_sum256d);
                C_castsi256_pd(tmp_pd_cast, C_tmp256i_64);     // interpret mask bits as doubles
                C_and_pd(C_first_sum256d, C_first_sum256d,tmp_pd_cast);

                // tmp256i = _mm256_cmpeq_epi64(tmp256i, set1(0));
                C_cmpeq_epi64((uint64_t*)C_tmp256i_b, C_tmp256i_64, C_zeros64);

                // first_sum256d = first_sum256d + (_mm256_castsi256_pd(tmp256i) & sum256d)
                C_castsi256_pd(tmp_pd_and, C_tmp256i_b);
                C_and_pd(tmp_pd_and, tmp_pd_and, C_sum256d);
                C_add_pd(C_first_sum256d, C_first_sum256d, tmp_pd_and);

                // last_sum256d = sum256d;
                for (int jj = 0; jj < 4; ++jj) C_last_sum256d[jj] = C_sum256d[jj];

                // tmp256i = _mm256_cmpeq_epi64(start256i, set1(1));
                C_cmpeq_epi64((uint64_t*)C_tmp256i_c, C_start256i, C_ones64);

                // sum256d = _mm256_and_pd(_mm256_castsi256_pd(tmp256i), first_sum256d);
                C_castsi256_pd(tmp_pd_cast, C_tmp256i_c);
                C_and_pd(C_sum256d, tmp_pd_cast, C_first_sum256d);

                // sum256d = _mm256_permute4x64_pd(sum256d, 0x39);
                C_permute4x64_pd(C_sum256d, C_sum256d, 0x39);

                // mask out highest lane: sum256d &= {~0, ~0, ~0, 0}
                {
                uint64_t mask64[4] = {
                    0xFFFFFFFFFFFFFFFFLL,
                    0xFFFFFFFFFFFFFFFFLL,
                    0xFFFFFFFFFFFFFFFFLL,
                    0x0000000000000000LL
                };

                    C_castsi256_pd(tmp_pd_mask, mask64);
                    C_and_pd(C_sum256d, tmp_pd_mask, C_sum256d);
                }

                // tmp_sum256d = sum256d;
                for (int jj = 0; jj < 4; ++jj) C_tmp_sum256d[jj] = C_sum256d[jj];

                // sum256d = hscan_avx(sum256d);
                hscan_avx(C_sum256d, C_sum256d);

                // scansum_offset128i += {0,1,2,3}
                C_add_epi32_128(C_scansum_offset128i, C_scansum_offset128i, add3210);

                // tmp256i = cast128->256 (128bit scansum_offset -> 256bit lanes of 8*int32)
                castsi128_si256(tmp256i32_128, C_scansum_offset128i); // tmp256i32_128: 8x int32

                // tmp256i = permutevar8x32(tmp256i, {0,0,1,1,2,2,3,3})
                C_permutevar8x32_epi32(tmp256i32_perm1, tmp256i32_128, idx_perm_pattern);

                // tmp256i = tmp256i + tmp256i
                C_add_epi32(tmp256i32_perm2, tmp256i32_perm1, tmp256i32_perm1);

                // tmp256i = tmp256i + {0,1,0,1,0,1,0,1}
                C_add_epi32(tmp256i32_perm3, tmp256i32_perm2, add0101);

                C_castpd_si256(cast_si64, C_sum256d); // cast double->int64 bits

                // reinterpret int64[4] 为 int32[8] 以供 permutevar8x32
                for (int jj = 0; jj < 8; ++jj) {
                    cast_si32[jj] = ((int32_t*)cast_si64)[jj];
                }

                // permute 8xint32
                int32_t tmpPerm32[8];
                C_permutevar8x32_epi32(tmpPerm32, cast_si32, tmp256i32_perm3);
                C_castsi256_pd(tmp_pd_cast, (int64_t*)tmpPerm32);

                // sum256d = tmp_pd_cast - sum256d
                C_sub_pd(C_sum256d, tmp_pd_cast, C_sum256d);

                // sum256d += tmp_sum256d
                C_add_pd(C_sum256d, C_sum256d, C_tmp_sum256d);

                // tmp256i = _mm256_cmpgt_epi64(start256i, stop256i);
                C_cmpgt_epi64((uint64_t*)C_tmp256i_d, C_start256i, C_stop256i);

                // tmp256i = _mm256_cmpeq_epi64(tmp256i, 0);
                C_cmpeq_epi64((uint64_t*)C_tmp256i_b, C_tmp256i_d, C_zeros64);

                // last_sum256d += (cast(tmp256i) & sum256d)
                C_castsi256_pd(tmp_pd_and2, C_tmp256i_b);
                C_and_pd(tmp_pd_and2, tmp_pd_and2, C_sum256d);
                C_add_pd(C_last_sum256d, C_last_sum256d, tmp_pd_and2);

                // y_idx128i = empty_rows ? gather(...) : y_offset128i
                if (empty_rows) {
                    // gather into C_y_idx128i
                    C_i32gather_epi32(C_y_idx128i,
                                    &d_partition_descriptor_offset[offset_pointer],
                                    C_y_offset128i,
                                    4); 
                } else {
                    for (int jj = 0; jj < 4; ++jj) C_y_idx128i[jj] = C_y_offset128i[jj];
                }

                // store to s_cond, s_y_idx, s_sum (模拟原始 AVX store)
                C_store_si256_i64((int64_t*)s_cond, C_direct256i);     // s_cond <- direct256i (按你文档用法)
                C_store_si128_i32(s_y_idx, C_y_idx128i);               // s_y_idx <- C_y_idx128i
                C_store_pd(s_sum, C_last_sum256d);                     // s_sum <- C_last_sum256d
                if (s_cond[0]) {d_y_local[s_y_idx[0]] += s_sum[0]* alpha; C_store_pd(s_first_sum, C_first_sum256d);}
                if (s_cond[1]) d_y_local[s_y_idx[1]] += s_sum[1]* alpha;
                if (s_cond[2]) d_y_local[s_y_idx[2]] += s_sum[2]* alpha;
                if (s_cond[3]) d_y_local[s_y_idx[3]] += s_sum[3]* alpha;

                // only use calibrator if this partition does not contain the first element of the row "row_start"
                if (row_start == start_row_start && !first_all_direct)
                    d_calibrator[tid * stride_vT] += (s_cond[0] ? s_first_sum[0] : s_sum[0])* alpha;
                else{
                    if(first_direct)
                        //d_y[row_start] = (s_cond[0] ? s_first_sum[0] : s_sum[0])* alpha;
                        d_y[row_start] += (s_cond[0] ? s_first_sum[0] : s_sum[0])* alpha;
                    else
                        d_y[row_start] += (s_cond[0] ? s_first_sum[0] : s_sum[0])* alpha;
                }
            }
        }
    }
}

template<typename iT, typename uiT, typename vT>
void spmv_csr5_calibrate_kernel(const uiT *d_partition_pointer,
                                vT        *d_calibrator,
                                vT        *d_y,
                                const iT   p)
{
    int num_thread = omp_get_max_threads();
    int chunk = ceil((double)(p-1) / (double)num_thread);
    int stride_vT = ANONYMOUSLIB_X86_CACHELINE / sizeof(vT);
    // calculate the number of maximal active threads (for a static loop scheduling with size chunk)
    int num_thread_active = ceil((p-1.0)/chunk);
    int num_cali = num_thread_active < num_thread ? num_thread_active : num_thread;

    for (int i = 0; i < num_cali; i++)
    {
        d_y[(d_partition_pointer[i * chunk] << 1) >> 1] += d_calibrator[i * stride_vT];
    }
}

template<typename iT, typename uiT, typename vT>
void spmv_csr5_tail_partition_kernel(const iT           *d_row_pointer,
                                     const iT           *d_column_index,
                                     const vT           *d_value,
                                     const vT           *d_x,
                                     vT                 *d_y,
                                     const iT            tail_partition_start,
                                     const iT            p,
                                     const iT            m,
                                     const int           sigma,
                                     const vT            alpha)
{
    const iT index_first_element_tail = (p - 1) * ANONYMOUSLIB_CSR5_OMEGA * sigma;
    
    #pragma omp parallel for
    for (iT row_id = tail_partition_start; row_id < m; row_id++)
    {
        const iT idx_start = row_id == tail_partition_start ? (p - 1) * ANONYMOUSLIB_CSR5_OMEGA * sigma : d_row_pointer[row_id];
        const iT idx_stop  = d_row_pointer[row_id + 1];

        vT sum = 0;
        for (iT idx = idx_start; idx < idx_stop; idx++)
            sum += d_value[idx] * d_x[d_column_index[idx]];// * alpha;
        sum = sum * alpha;//support alpha
        if(row_id == tail_partition_start && d_row_pointer[row_id] != index_first_element_tail){
            d_y[row_id] = d_y[row_id] + sum;
        }else{
            //d_y[row_id] = sum;
            d_y[row_id] = d_y[row_id] + sum;
        }
    }
}


template<typename ANONYMOUSLIB_IT, typename ANONYMOUSLIB_UIT, typename ANONYMOUSLIB_VT>
int csr5_spmv(const int                 sigma,
              const ANONYMOUSLIB_IT         p,
              const ANONYMOUSLIB_IT         m,
              const int                 bit_y_offset,
              const int                 bit_scansum_offset,
              const int                 num_packet,
              const ANONYMOUSLIB_IT        *row_pointer,
              const ANONYMOUSLIB_IT        *column_index,
              const ANONYMOUSLIB_VT        *value,
              const ANONYMOUSLIB_UIT       *partition_pointer,
              const ANONYMOUSLIB_UIT       *partition_descriptor,
              const ANONYMOUSLIB_IT        *partition_descriptor_offset_pointer,
              const ANONYMOUSLIB_IT        *partition_descriptor_offset,
              ANONYMOUSLIB_VT              *calibrator,
              const ANONYMOUSLIB_IT         tail_partition_start,
              const ANONYMOUSLIB_VT         alpha,
              const ANONYMOUSLIB_VT         beta,
              const ANONYMOUSLIB_VT        *x,
              ANONYMOUSLIB_VT              *y)
{
    int err = ANONYMOUSLIB_SUCCESS;
    scale_y<ANONYMOUSLIB_IT, ANONYMOUSLIB_VT>(m, beta, y);
    const int num_thread = omp_get_max_threads();
    memset(calibrator,0,ANONYMOUSLIB_X86_CACHELINE*num_thread);
    spmv_csr5_compute_kernel
            <ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>
            (column_index, value, row_pointer, x,
             partition_pointer, partition_descriptor,
             partition_descriptor_offset_pointer, partition_descriptor_offset,
             calibrator, y, p,
             num_packet, bit_y_offset, bit_scansum_offset, alpha, sigma);

    spmv_csr5_calibrate_kernel
            <ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>
            (partition_pointer, calibrator, y, p);

    spmv_csr5_tail_partition_kernel
            <ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>
            (row_pointer, column_index, value, x, y,
             tail_partition_start, p, m, sigma, alpha);

    return err;
}

#endif // CSR5_SPMV_CUNROLL_H
