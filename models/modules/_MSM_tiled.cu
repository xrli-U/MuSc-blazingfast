extern "C" __global__ __launch_bounds__(256)
void tiled_min(const float* __restrict__ data, const float* __restrict__ norms,
               float* __restrict__ minima, int images, int patches, int stride,
               int channels, int query, int first_ref) {
    const int tx = threadIdx.x, ty = threadIdx.y, tid = 16 * ty + tx;
    const int ref = first_ref + blockIdx.z;
    const int row0 = blockIdx.y * 128, col0 = blockIdx.x * 128;
    const float* a = data + (long long)query * channels * stride + row0;
    const float* b = data + (long long)ref * channels * stride + col0;
    __shared__ __align__(16) float sa[16][128], sb[16][128];
    float acc[8][8] = {};
    for (int k0 = 0; k0 < channels; k0 += 16) {
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            int k = tid / 32 + 8 * half, offset = (tid % 32) * 4;
            *reinterpret_cast<float4*>(&sa[k][offset]) =
                *reinterpret_cast<const float4*>(a + (long long)(k0 + k) * stride + offset);
            *reinterpret_cast<float4*>(&sb[k][offset]) =
                *reinterpret_cast<const float4*>(b + (long long)(k0 + k) * stride + offset);
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
            __align__(16) float av[8], bv[8];
            #pragma unroll
            for (int group = 0; group < 2; ++group) {
                *reinterpret_cast<float4*>(&av[4 * group]) =
                    *reinterpret_cast<const float4*>(&sa[k][ty * 4 + 64 * group]);
                *reinterpret_cast<float4*>(&bv[4 * group]) =
                    *reinterpret_cast<const float4*>(&sb[k][tx * 4 + 64 * group]);
            }
            #pragma unroll
            for (int u = 0; u < 8; ++u)
                #pragma unroll
                for (int v = 0; v < 8; ++v)
                    acc[u][v] = fmaf(av[u], bv[v], acc[u][v]);
        }
        __syncthreads();
    }
    float na[8], nb[8], row_min[8], col_min[8];
    const float inf = __int_as_float(0x7f800000);
    #pragma unroll
    for (int t = 0; t < 8; ++t) {
        int offset = (t / 4) * 64 + t % 4;
        na[t] = norms[query * stride + row0 + ty * 4 + offset];
        nb[t] = norms[ref * stride + col0 + tx * 4 + offset];
        row_min[t] = inf;
        col_min[t] = inf;
    }
    #pragma unroll
    for (int u = 0; u < 8; ++u) {
        #pragma unroll
        for (int v = 0; v < 8; ++v) {
            float sq = fmaxf(0.f, fmaf(-2.f, acc[u][v], na[u] + nb[v]));
            row_min[u] = fminf(row_min[u], sq);
            col_min[v] = fminf(col_min[v], sq);
        }
    }
    #pragma unroll
    for (int t = 0; t < 8; ++t) {
        const int offset = (t / 4) * 64 + t % 4;
        sa[tx][ty * 4 + offset] = row_min[t];
        sb[ty][tx * 4 + offset] = col_min[t];
    }
    __syncthreads();
    const int index = tid % 128;
    float v = inf;
    #pragma unroll
    for (int q = 0; q < 16; ++q)
        v = fminf(v, tid < 128 ? sa[q][index] : sb[q][index]);
    int patch = (tid < 128 ? row0 : col0) + index;
    if (patch < patches) {
        long long pair = tid < 128 ? (long long)query * images + ref : (long long)ref * images + query;
        atomicMin(reinterpret_cast<unsigned*>(minima + pair * patches + patch), __float_as_uint(v));
    }
}
