/**
 * test_matmul.c - Checks tiled_mat_mat_mul_fpga against a float32 reference
 *
 * Runs against the software model of the device (FPGA_MODEL), so it needs
 * neither a board nor XRT. What it is really testing is the host's half of the
 * contract - the blocking, the rounding of ragged dimensions up to a multiple
 * of FPGA_ARRAY_DIM, the zero padding, the tile-major de-tiling, the resident-A
 * launch pattern, and the float32 sum across k blocks - since the model is
 * bit-exact with the RTL by construction.
 *
 * Two passes over the same shapes:
 *
 * - integer operands, where the whole pipeline is exact and any structural
 *   mistake shows up as an exact mismatch. Every value is a small integer, so
 *   the power-of-two rescale, the float16 conversion, the fixed-point
 *   accumulate and the conversion back are all lossless, and the result must
 *   equal the float32 reference bit for bit
 *
 * - random operands, checked against a relative tolerance, which is the
 *   realistic accuracy signal
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "config.h"
#include "fpga.h"
#include "mat_ops.h"

// float16 carries 11 significant bits, so a single rounded value is good to
// about 2**-11. Errors across a dot product are signed and largely cancel, but
// this stays deliberately loose: a structural bug misplaces whole rows or
// columns and misses by a factor, not by a few ulp
#define REL_TOL 2e-2f

typedef struct {
    int m, k, n;
    const char *what;
} Shape;

static const Shape SHAPES[] = {
    {   8,   8,   8, "one tile"                        },
    {   7,  13,   5, "tiny, ragged in every dimension" },
    {   1,   1,   1, "degenerate"                      },
    {  64,  64,  64, "one sub-block"                   },
    { 256, 256, 256, "exactly one full block"          },
    { 264, 264, 264, "one element past a full block"   },
    { 100, 784,  37, "ragged, K spans four blocks"     },
    { 128, 300, 512, "multi-block K and N"             },
    {  10, 100, 600, "wide N, K inside one block"      },
};

#define NUM_SHAPES ((int)(sizeof(SHAPES) / sizeof(SHAPES[0])))

/**
 * @brief Fills a matrix with small integers, keeping every stage of the
 *        pipeline exact
 *
 * The largest magnitude the result can reach is `k * lim**2`; float16 holds
 * integers exactly up to 2048, so the range shrinks as K grows.
 *
 * @param[out] X   Matrix to fill
 * @param[in]  len Number of elements
 * @param[in]  k   Contraction length of the matmul it feeds
 */
static void fill_int(float *X, int len, int k) {
    int lim = (k > 500) ? 1 : 2;

    for (int i = 0; i < len; i++) {
        X[i] = (float)(rand() % (2 * lim + 1) - lim);
    }
}

/**
 * @brief Fills a matrix with random values spanning a few orders of magnitude
 *
 * @param[out] X   Matrix to fill
 * @param[in]  len Number of elements
 */
static void fill_rand(float *X, int len) {
    for (int i = 0; i < len; i++) {
        float u = (float)rand() / (float)RAND_MAX * 2.f - 1.f;

        // a spread of exponents, so the per-row/column scales actually differ
        X[i] = u * ldexpf(1.f, rand() % 9 - 4);
    }
}

/**
 * @brief Runs one shape both ways and reports the worst disagreement
 *
 * @param[in] s     Shape to test
 * @param[in] exact Whether the two results are required to match exactly
 * @return          1 if the case passed, 0 otherwise
 */
static int run_case(Shape s, int exact) {
    float *A = malloc((size_t)s.m * s.k * sizeof(float));
    float *B = malloc((size_t)s.k * s.n * sizeof(float));
    float *C = malloc((size_t)s.m * s.n * sizeof(float));
    float *C_ref = malloc((size_t)s.m * s.n * sizeof(float));

    if (!A || !B || !C || !C_ref) {
        fprintf(stderr, "  out of memory for %dx%dx%d\n", s.m, s.k, s.n);
        exit(1);
    }

    if (exact) {
        fill_int(A, s.m * s.k, s.k);
        fill_int(B, s.k * s.n, s.k);
    } else {
        fill_rand(A, s.m * s.k);
        fill_rand(B, s.k * s.n);
    }

    naive_mat_mat_mul(A, B, s.m, s.k, s.n, C_ref);
    tiled_mat_mat_mul_fpga(A, B, s.m, s.k, s.n, C);

    // the reference's own magnitude sets what counts as a relative error;
    // scaling against the largest |C| avoids dividing by a near-zero element
    float max_ref = 0.f;
    for (int i = 0; i < s.m * s.n; i++) {
        max_ref = fmaxf(max_ref, fabsf(C_ref[i]));
    }

    float worst = 0.f;
    int bad_i = -1;

    for (int i = 0; i < s.m * s.n; i++) {
        float err = fabsf(C[i] - C_ref[i]);

        if (!exact && max_ref > 0.f) {
            err /= max_ref;
        }

        if (err > worst) {
            worst = err;
            bad_i = i;
        }
    }

    int pass = exact ? (worst == 0.f) : (worst <= REL_TOL);

    if (pass) {
        printf("  %-34s %3dx%-4d @ %4dx%-4d  %s %.3g\n",
               s.what, s.m, s.k, s.k, s.n,
               exact ? "max abs err" : "max rel err", worst);
    } else {
        printf("  %-34s %3dx%-4d @ %4dx%-4d  FAILED: %s %.3g at (%d,%d), "
               "got %.9g want %.9g\n",
               s.what, s.m, s.k, s.k, s.n,
               exact ? "max abs err" : "max rel err", worst,
               bad_i / s.n, bad_i % s.n, C[bad_i], C_ref[bad_i]);
    }

    free(A);
    free(B);
    free(C);
    free(C_ref);

    return pass;
}

int main(void) {
    int failures = 0;

    srand(240);

    fpga_init();

    printf("test_matmul: FPGA_BLOCK_DIM=%d FPGA_ARRAY_DIM=%d\n\n",
           FPGA_BLOCK_DIM, FPGA_ARRAY_DIM);

    printf("integer operands (must be exact):\n");
    for (int i = 0; i < NUM_SHAPES; i++) {
        failures += !run_case(SHAPES[i], 1);
    }

    printf("\nrandom operands (relative tolerance %.3g):\n", (double)REL_TOL);
    for (int i = 0; i < NUM_SHAPES; i++) {
        failures += !run_case(SHAPES[i], 0);
    }

    fpga_cleanup();

    printf("\n");

    if (failures != 0) {
        printf("test_matmul: FAILED -- %d of %d cases\n", failures, 2 * NUM_SHAPES);

        return 1;
    }

    printf("test_matmul: all %d cases passed!\n", 2 * NUM_SHAPES);

    return 0;
}
