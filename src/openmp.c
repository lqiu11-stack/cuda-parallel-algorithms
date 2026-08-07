#include "openmp.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

/*
 * Precompute all 16 unique glider patterns:
 *   2 base gliders × 2 reflections × 4 rotations = 16
 * Each pattern is stored row-major as a flat 9-element array.
 * The rotation transform is: (x,y) -> (y, 2-x), matching cpu.c exactly.
 */
static void init_glider_patterns(unsigned char patterns[16][9]) {
    static const unsigned char BASE[2][3][3] = {
        {{1, 0, 1}, {1, 1, 0}, {0, 0, 0}},
        {{0, 1, 0}, {1, 0, 0}, {1, 0, 1}}
    };
    int idx = 0;
    for (int g = 0; g < 2; g++) {
        for (int reflect = 0; reflect <= 1; reflect++) {
            for (int rotate = 0; rotate < 4; rotate++) {
                for (int gx = 0; gx < 3; gx++) {
                    for (int gy = 0; gy < 3; gy++) {
                        int tx = reflect ? 2 - gx : gx;
                        int ty = gy;
                        for (int r = 0; r < rotate; r++) {
                            int tmp = tx;
                            tx = ty;
                            ty = 2 - tmp;
                        }
                        patterns[idx][gy * 3 + gx] = BASE[g][ty][tx];
                    }
                }
                idx++;
            }
        }
    }
}

uint64_t openmp_countGliders(const unsigned char *cells, const size_t width, const size_t height) {
    unsigned char patterns[16][9];
    init_glider_patterns(patterns);

    const size_t width2  = width  - 2;
    const size_t height2 = height - 2;
    uint64_t count = 0;

    /*
     * Each (x,y) window is independent: no shared writes until the reduction.
     * collapse(2) fuses both loops into a single iteration space so the
     * scheduler distributes work evenly across threads with no synchronisation
     * inside the loop body.
     */
#pragma omp parallel for reduction(+:count) collapse(2) schedule(static)
    for (size_t y = 0; y < height2; y++) {
        for (size_t x = 0; x < width2; x++) {
            for (int p = 0; p < 16; p++) {
                int match = 1;
                for (int gy = 0; gy < 3 && match; gy++) {
                    for (int gx = 0; gx < 3 && match; gx++) {
                        if (cells[(y + gy) * width + (x + gx)] != patterns[p][gy * 3 + gx])
                            match = 0;
                    }
                }
                if (match) {
                    count++;
                    break;
                }
            }
        }
    }
    return count;
}

size_t openmp_histogram(const int *numbers, size_t length, int bin_width, int *output) {
    const size_t hist_len = (size_t)ceil(HISTOGRAM_MAX_VALUE / (double)bin_width);
    memset(output, 0, sizeof(int) * hist_len);

    /*
     * Thread-private histograms eliminate all atomic contention on shared
     * bins.  Each thread increments its own copy; after the parallel region
     * the private copies are merged sequentially.  The calloc zero-inits all
     * private storage in one allocation.
     */
    const int nthreads = omp_get_max_threads();
    int *local_hists = (int *)calloc((size_t)nthreads * hist_len, sizeof(int));

#pragma omp parallel
    {
        int tid   = omp_get_thread_num();
        int *priv = local_hists + (size_t)tid * hist_len;

#pragma omp for schedule(static)
        for (size_t i = 0; i < length; i++)
            priv[numbers[i] / bin_width]++;
    }

    for (int t = 0; t < nthreads; t++) {
        const int *priv = local_hists + (size_t)t * hist_len;
        for (size_t b = 0; b < hist_len; b++)
            output[b] += priv[b];
    }

    free(local_hists);
    return hist_len;
}

void openmp_emboss(const unsigned char *pixels, const size_t width, const size_t height, unsigned char *output) {
    static const float KERNEL[3][3] = {
        {-2.0f, -1.0f, 0.0f},
        {-1.0f,  0.0f, 1.0f},
        { 0.0f,  1.0f, 2.0f}
    };

    const size_t out_width  = width  - 2;
    const size_t out_height = height - 2;

    /*
     * Every output pixel depends only on a fixed 3x3 neighbourhood of the
     * input; there are no write conflicts between output pixels.
     * collapse(2) maximises the parallel iteration space for better load
     * balance, especially on images whose dimensions are not multiples of
     * the thread count.
     */
#pragma omp parallel for collapse(2) schedule(static)
    for (size_t y = 0; y < out_height; y++) {
        for (size_t x = 0; x < out_width; x++) {
            float pixel_sum = 0.0f;
            for (int ky = 0; ky < 3; ky++) {
                for (int kx = 0; kx < 3; kx++) {
                    const size_t off = (width * (y + ky) + (x + kx)) * 3;
                    const float grey = 0.2126f * pixels[off]
                                     + 0.7152f * pixels[off + 1]
                                     + 0.0722f * pixels[off + 2];
                    pixel_sum += grey * KERNEL[ky][kx];
                }
            }
            pixel_sum += 128.0f;
            if (pixel_sum < 0.0f)   pixel_sum = 0.0f;
            if (pixel_sum > 255.0f) pixel_sum = 255.0f;
            output[y * out_width + x] = (unsigned char)pixel_sum;
        }
    }
}
