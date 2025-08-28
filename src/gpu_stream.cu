// STREAM-like CUDA microbenchmark (Triad + Copy), vectorized & unrolled.
// Professional: explicit args, clear output keys, no hidden globals.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// -------------------- Tunables (compile-time) --------------------
#ifndef VEC_BYTES
#define VEC_BYTES 16   // 16-byte vectorized access: float4 / double2
#endif

#ifndef UNROLL
#define UNROLL 4       // how many vector chunks each thread handles per loop step
#endif

#ifndef USE_DOUBLE
#define USE_DOUBLE 0   // 0 = float (bandwidth-friendly), 1 = double
#endif

#if VEC_BYTES != 16
#error "This build expects VEC_BYTES==16 (float4/double2)."
#endif
// -----------------------------------------------------------------

// Vector types for 16B access
typedef float4  vtype_f;
typedef double2 vtype_d;

template<typename T, typename VT>
__device__ __forceinline__ VT ld_vec(const T* p){ return *reinterpret_cast<const VT*>(p); }

template<typename T, typename VT>
__device__ __forceinline__ void st_vec(T* p, VT v){ *reinterpret_cast<VT*>(p) = v; }

template<typename T, typename VT, bool COPY>
__global__ void stream_vec(T* __restrict__ a,
                           const T* __restrict__ b,
                           const T* __restrict__ c,
                           T alpha, size_t n_vec) {
    // n_vec counts 16-byte "vector elements": 4 floats or 2 doubles
    const size_t tid    = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;

    for (size_t base = tid; base < n_vec; base += stride * UNROLL) {
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const size_t idx = base + (size_t)u * stride;
            if (idx < n_vec) {
                const char* bp = (const char*)b + idx * VEC_BYTES;
                char*       ap = (char*)a + idx * VEC_BYTES;

                VT vb = ld_vec<T,VT>((const T*)bp);

#if USE_DOUBLE
                double2 xb = *reinterpret_cast<double2*>(&vb);
                double2 xa;
                if (COPY) {
                    xa = xb; // a = b
                } else {
                    const char* cp = (const char*)c + idx * VEC_BYTES;
                    VT vc = ld_vec<T,VT>((const T*)cp);
                    double2 xc = *reinterpret_cast<double2*>(&vc);
                    xa.x = xb.x + (double)alpha * xc.x; // triad
                    xa.y = xb.y + (double)alpha * xc.y;
                }
                VT va = *reinterpret_cast<VT*>(&xa);
#else
                float4 xb = *reinterpret_cast<float4*>(&vb);
                float4 xa;
                if (COPY) {
                    xa = xb; // a = b
                } else {
                    const char* cp = (const char*)c + idx * VEC_BYTES;
                    VT vc = ld_vec<T,VT>((const T*)cp);
                    float4 xc = *reinterpret_cast<float4*>(&vc);
                    xa.x = xb.x + (float)alpha * xc.x; // triad
                    xa.y = xb.y + (float)alpha * xc.y;
                    xa.z = xb.z + (float)alpha * xc.z;
                    xa.w = xb.w + (float)alpha * xc.w;
                }
                VT va = *reinterpret_cast<VT*>(&xa);
#endif
                st_vec<T,VT>((T*)ap, va);
            }
        }
    }
}

static inline void ck(cudaError_t e, const char* where){
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s: %s\n", where, cudaGetErrorString(e));
        std::exit(1);
    }
}

int main(int argc, char** argv){
    // Args:
    //   -n <elements>      : scalar elements (floats or doubles)
    //   -t <iters>         : iterations (default 25)
    //   -bs <block size>   : threads per block (default 512)
    //   -op triad|copy     : kernel operation (default triad)
    size_t N = 0;
    int iters = 25;
    int block_size = 512;
    int is_copy = 0; // default triad

    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) {
            N = std::strtoull(argv[++i], nullptr, 10);
        } else if (!std::strcmp(argv[i], "-t") && i + 1 < argc) {
            iters = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "-bs") && i + 1 < argc) {
            block_size = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "-op") && i + 1 < argc) {
            is_copy = !std::strcmp(argv[++i], "copy"); // triad by default
        }
    }
    if (N == 0) {
        std::fprintf(stderr, "Usage: %s -n <elements> [-t iters] [-bs block] [-op triad|copy]\n", argv[0]);
        return 2;
    }

    int dev = 0;
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess) {
        std::fprintf(stderr, "[GPU] Device: %s, CC %d.%d\n", prop.name, prop.major, prop.minor);
    }

    const size_t scalar_size     = (USE_DOUBLE ? sizeof(double) : sizeof(float));
    const size_t scalars_per_vec = (size_t)(VEC_BYTES / scalar_size);
    const size_t n_vec           = N / scalars_per_vec;                 // vector chunks we will actually process
    const size_t N_eff           = n_vec * scalars_per_vec;             // snapped element count actually processed
    const size_t bytes           = N_eff * scalar_size;                 // device allocations based on N_eff
    const size_t streams         = is_copy ? 2u : 3u;
    const double ws_gib          = (double)(streams * bytes) / (1024.0 * 1024.0 * 1024.0);

    std::fprintf(stderr,
        "[GPU] op=%s  N_req=%zu  N_eff=%zu  WS=%.2f GiB  vec=%zu scalars  n_vec=%zu  block=%d  UNROLL=%d  VEC_BYTES=%d\n",
        (is_copy ? "copy" : "triad"),
        N, N_eff, ws_gib, scalars_per_vec, n_vec, block_size, UNROLL, VEC_BYTES);

    void *a = nullptr, *b = nullptr, *c = nullptr;
    ck(cudaMalloc(&a, bytes), "cudaMalloc(a)");
    ck(cudaMalloc(&b, bytes), "cudaMalloc(b)");
    if (!is_copy) ck(cudaMalloc(&c, bytes), "cudaMalloc(c)");
    ck(cudaMemset(a, 0, bytes), "cudaMemset(a)");
    ck(cudaMemset(b, 0, bytes), "cudaMemset(b)");
    if (!is_copy) ck(cudaMemset(c, 0, bytes), "cudaMemset(c)");

    // Grid config in "vector elements" space
    const int maxBlocks = 65535;
    int grid = (int)((n_vec + (size_t)block_size - 1) / (size_t)block_size);
    if (grid < 1) grid = 1;
    if (grid > maxBlocks) grid = maxBlocks;
    std::fprintf(stderr, "[GPU] launch: grid=%d, block=%d, threads=%lld\n",
                 grid, block_size, (long long)grid * block_size);

#if USE_DOUBLE
    using T  = double;
    using VT = vtype_d;
#else
    using T  = float;
    using VT = vtype_f;
#endif

    // Warmup
    if (is_copy)
        stream_vec<T,VT,true>  <<<grid, block_size>>> ((T*)a, (const T*)b, (const T*)c, (T)1.1, n_vec);
    else
        stream_vec<T,VT,false> <<<grid, block_size>>> ((T*)a, (const T*)b, (const T*)c, (T)1.1, n_vec);
    ck(cudaDeviceSynchronize(), "warmup");

    // Timed loop
    cudaEvent_t s, e;
    ck(cudaEventCreate(&s), "cudaEventCreate(s)");
    ck(cudaEventCreate(&e), "cudaEventCreate(e)");
    ck(cudaEventRecord(s), "cudaEventRecord(s)");
    for (int it = 0; it < iters; ++it) {
        if (is_copy)
            stream_vec<T,VT,true>  <<<grid, block_size>>> ((T*)a, (const T*)b, (const T*)c, (T)1.1, n_vec);
        else
            stream_vec<T,VT,false> <<<grid, block_size>>> ((T*)a, (const T*)b, (const T*)c, (T)1.1, n_vec);
    }
    ck(cudaEventRecord(e), "cudaEventRecord(e)");
    ck(cudaEventSynchronize(e), "cudaEventSynchronize(e)");
    float ms = 0.0f;
    ck(cudaEventElapsedTime(&ms, s, e), "cudaEventElapsedTime");

    // Compute throughput (GiB/s) using N_eff (snapped to vector multiple)
    const long double moved = (long double)N_eff * (long double)streams * (long double)scalar_size * (long double)iters;
    const long double gib   = moved / (1024.0L * 1024.0L * 1024.0L);
    const long double secs  = (long double)ms / 1e3L;

    if (is_copy) std::printf("GPU_Copy_GiBs: %.3Lf\n",  gib / secs);
    else         std::printf("GPU_Triad_GiBs: %.3Lf\n", gib / secs);

    cudaFree(a); cudaFree(b); if (c) cudaFree(c);
    return 0;
}
