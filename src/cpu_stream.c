// STREAM-like CPU microbenchmark (Triad + Copy)
// Professional: clear args, no hidden globals, portable fallback for NT stores.

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <omp.h>

// ---------------- Non-temporal store helper ----------------

#ifndef NT_STORE
#define NT_STORE 1
#endif

#ifndef __has_builtin
#define __has_builtin(x) 0
#endif

#if NT_STORE && (__has_builtin(__builtin_nontemporal_store))
  #define HAS_NT_BUILTIN 1
#else
  #define HAS_NT_BUILTIN 0
#endif

static inline void nt_store_double(double *dst, double v) {
#if HAS_NT_BUILTIN
  __builtin_nontemporal_store(v, dst);
#else
  *dst = v;   // safe fallback
#endif
}

// ------------------------------------------------------------

static inline double now_sec(void) { return omp_get_wtime(); }

static inline void *xaligned_alloc(size_t alignment, size_t size) {
    void *p = NULL;
    if (posix_memalign(&p, alignment, size) != 0) return NULL;
    return p;
}

typedef enum { OP_TRIAD, OP_COPY } op_t;

int main(int argc, char **argv) {
    // Args: -n <elements> -t <iters> [-op triad|copy] [-nt 0|1]
    size_t N = 0;
    int iters = 25;
    op_t op = OP_TRIAD;
    int use_nt = 1;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) {
            N = strtoull(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "-t") && i + 1 < argc) {
            iters = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-op") && i + 1 < argc) {
            ++i;
            op = (!strcmp(argv[i], "copy")) ? OP_COPY : OP_TRIAD;
        } else if (!strcmp(argv[i], "-nt") && i + 1 < argc) {
            use_nt = atoi(argv[++i]);
        }
    }
    if (N == 0) {
        fprintf(stderr, "Usage: %s -n <elements> [-t iters] [-op triad|copy] [-nt 0|1]\n", argv[0]);
        return 2;
    }

    const size_t elem = sizeof(double);
    const size_t streams = (op == OP_COPY ? 2u : 3u);
    const size_t bytes = N * elem;
    const double ws_gib = (double)(streams * bytes) / (1024.0 * 1024.0 * 1024.0);

    fprintf(stderr, "[CPU] op=%s  N=%zu  WS=%.2f GiB  nt_store=%s\n",
            (op == OP_COPY ? "copy" : "triad"), N, ws_gib, use_nt ? "on" : "off");

    double *a = (double *)xaligned_alloc(64, bytes);
    double *b = (double *)xaligned_alloc(64, bytes);
    double *c = (op == OP_COPY) ? NULL : (double *)xaligned_alloc(64, bytes);
    if (!a || !b || (op == OP_TRIAD && !c)) {
        fprintf(stderr, "CPU alloc failed\n");
        free(a); free(b); if (c) free(c);
        return 1;
    }

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) {
        a[i] = 0.0; 
        b[i] = 1.0; 
        if (c) c[i] = 2.0;
    }

    // Warmup
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) {
        const double val = (op == OP_COPY) ? b[i] : (b[i] + 1.1 * c[i]);
        if (use_nt) nt_store_double(&a[i], val); else a[i] = val;
    }

    const double t0 = now_sec();
    for (int it = 0; it < iters; ++it) {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < N; ++i) {
            #pragma omp simd
            for (size_t j = i; j < i + 1; ++j) {
                const double val = (op == OP_COPY) ? b[j] : (b[j] + 1.1 * c[j]);
                if (use_nt) nt_store_double(&a[j], val); else a[j] = val;
            }
        }
    }
    const double secs = now_sec() - t0;

    long double moved = (long double)N * (long double)streams * (long double)elem * (long double)iters;
    long double gib = moved / (1024.0L * 1024.0L * 1024.0L);

    if (op == OP_COPY) 
        printf("CPU_Copy_GiBs: %.3Lf\n", gib / secs);
    else
        printf("CPU_Triad_GiBs: %.3Lf\n", gib / secs);

    free(a); free(b); if (c) free(c);
    return 0;
}
