#include "cuda.cuh"

#include <cmath>
#include <cstring>

/* ================================================================
 *  Shared glider pattern precomputation
 *  16 patterns (2 base × 2 reflections × 4 rotations), stored in
 *  CUDA constant memory so every thread reads from the fast
 *  read-only cache rather than global DRAM.
 * ================================================================ */

__constant__ unsigned char d_glider_patterns[16][9];

static void init_glider_patterns_host(unsigned char patterns[16][9]) {
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

/* ================================================================
 *  Count Gliders
 * ================================================================ */

__global__ void countGlidersKernel(
        const unsigned char * __restrict__ cells,
        size_t width, size_t height,
        unsigned long long *count)
{
    const size_t x = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = (size_t)blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width - 2 || y >= height - 2) return;

    for (int p = 0; p < 16; p++) {
        int match = 1;
        for (int gy = 0; gy < 3 && match; gy++) {
            for (int gx = 0; gx < 3 && match; gx++) {
                if (cells[(y + gy) * width + (x + gx)] != d_glider_patterns[p][gy * 3 + gx])
                    match = 0;
            }
        }
        if (match) {
            atomicAdd(count, 1ULL);
            return;
        }
    }
}

uint64_t cuda_countGliders(const unsigned char *cells, const size_t width, const size_t height) {
    /* Upload patterns once; they persist in constant memory for the
     * lifetime of the CUDA context, avoiding redundant transfers during
     * the 100-run benchmark. */
    static bool patterns_ready = false;
    if (!patterns_ready) {
        unsigned char h_patterns[16][9];
        init_glider_patterns_host(h_patterns);
        CUDA_CALL(cudaMemcpyToSymbol(d_glider_patterns, h_patterns, sizeof(h_patterns)));
        patterns_ready = true;
    }

    unsigned char *d_cells;
    CUDA_CALL(cudaMalloc(&d_cells, width * height));
    CUDA_CALL(cudaMemcpy(d_cells, cells, width * height, cudaMemcpyHostToDevice));

    unsigned long long *d_count;
    CUDA_CALL(cudaMalloc(&d_count, sizeof(unsigned long long)));
    CUDA_CALL(cudaMemset(d_count, 0, sizeof(unsigned long long)));

    dim3 block(16, 16);
    dim3 grid(((unsigned int)(width  - 2) + 15) / 16,
              ((unsigned int)(height - 2) + 15) / 16);
    countGlidersKernel<<<grid, block>>>(d_cells, width, height, d_count);
    CUDA_CHECK();

    unsigned long long h_count = 0;
    CUDA_CALL(cudaMemcpy(&h_count, d_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    CUDA_CALL(cudaFree(d_cells));
    CUDA_CALL(cudaFree(d_count));
    return (uint64_t)h_count;
}

/* ================================================================
 *  Histogram
 *
 *  Two-level atomics strategy:
 *    1. Each block maintains a private histogram in shared memory.
 *       Intra-block contention is absorbed by fast on-chip atomics.
 *    2. After the parallel scan each block merges its private histogram
 *       into the global result with global-memory atomics.
 *  Shared memory size = hist_len × 4 bytes (≤ 1020 bytes; always safe).
 * ================================================================ */

__global__ void histogramKernel(
        const int * __restrict__ numbers,
        size_t length, int bin_width,
        int *histogram, int hist_len)
{
    extern __shared__ int local_hist[];

    for (int i = threadIdx.x; i < hist_len; i += blockDim.x)
        local_hist[i] = 0;
    __syncthreads();

    const size_t idx    = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)blockDim.x * gridDim.x;
    for (size_t i = idx; i < length; i += stride)
        atomicAdd(&local_hist[numbers[i] / bin_width], 1);
    __syncthreads();

    for (int i = threadIdx.x; i < hist_len; i += blockDim.x)
        atomicAdd(&histogram[i], local_hist[i]);
}

size_t cuda_histogram(const int *numbers, size_t length, int bin_width, int *output) {
    const size_t hist_len = (size_t)std::ceil(HISTOGRAM_MAX_VALUE / (double)bin_width);
    std::memset(output, 0, sizeof(int) * hist_len);

    int *d_numbers, *d_histogram;
    CUDA_CALL(cudaMalloc(&d_numbers,   length    * sizeof(int)));
    CUDA_CALL(cudaMalloc(&d_histogram, hist_len  * sizeof(int)));

    CUDA_CALL(cudaMemcpy(d_numbers, numbers, length * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemset(d_histogram, 0, hist_len * sizeof(int)));

    const int block_size = 256;
    int grid_size = (int)((length + block_size - 1) / block_size);
    if (grid_size > 1024) grid_size = 1024;

    histogramKernel<<<grid_size, block_size, hist_len * sizeof(int)>>>(
        d_numbers, length, bin_width, d_histogram, (int)hist_len);
    CUDA_CHECK();

    CUDA_CALL(cudaMemcpy(output, d_histogram, hist_len * sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CALL(cudaFree(d_numbers));
    CUDA_CALL(cudaFree(d_histogram));
    return hist_len;
}

/* ================================================================
 *  Emboss — tiled shared-memory implementation
 *
 *  Each 16×16 thread block loads an 18×18 greyscale tile into shared
 *  memory (the extra border ring = the convolution halo).  The 9
 *  neighbouring greyscale values needed per output pixel are then read
 *  from the fast on-chip tile rather than from global DRAM, cutting
 *  global memory traffic by ≈ 6× compared with a naïve implementation.
 *
 *  Emboss kernel weights stored in constant memory so all threads share
 *  the same broadcast read via the constant cache.
 * ================================================================ */

#define BLOCK_W 16
#define BLOCK_H 16

__constant__ float d_emboss_kernel[9] = {
    -2.0f, -1.0f, 0.0f,
    -1.0f,  0.0f, 1.0f,
     0.0f,  1.0f, 2.0f
};

__global__ void embossKernel(
        const unsigned char * __restrict__ pixels,
        unsigned int width, unsigned int height,
        unsigned char *output)
{
    /* Shared tile: (BLOCK_H+2) rows × (BLOCK_W+2) cols of greyscale floats */
    __shared__ float tile[BLOCK_H + 2][BLOCK_W + 2];

    const unsigned int in_x_base = blockIdx.x * BLOCK_W;
    const unsigned int in_y_base = blockIdx.y * BLOCK_H;

    /* --- Load input tile into shared memory (includes 1-pixel halo) ---
     * Thread count = BLOCK_W*BLOCK_H = 256.
     * Tile size    = (BLOCK_W+2)*(BLOCK_H+2) = 324.
     * Each thread loads one element in the first pass; 68 threads load
     * a second element to cover the remainder.                          */
    const int tid       = threadIdx.y * BLOCK_W + threadIdx.x;
    const int tile_w    = BLOCK_W + 2;
    const int tile_h    = BLOCK_H + 2;
    const int tile_size = tile_w * tile_h;

    for (int i = tid; i < tile_size; i += BLOCK_W * BLOCK_H) {
        const int tile_y = i / tile_w;
        const int tile_x = i % tile_w;
        const int in_x   = (int)in_x_base + tile_x;
        const int in_y   = (int)in_y_base + tile_y;

        float grey = 0.0f;
        if (in_x < (int)width && in_y < (int)height) {
            const int off = (in_y * (int)width + in_x) * 3;
            grey = 0.2126f * pixels[off]
                 + 0.7152f * pixels[off + 1]
                 + 0.0722f * pixels[off + 2];
        }
        tile[tile_y][tile_x] = grey;
    }
    __syncthreads();

    /* --- Compute one output pixel per thread --- */
    const unsigned int out_x = in_x_base + threadIdx.x;
    const unsigned int out_y = in_y_base + threadIdx.y;
    const unsigned int out_w = width  - 2;
    const unsigned int out_h = height - 2;

    if (out_x >= out_w || out_y >= out_h) return;

    float pixel_sum = 0.0f;
    for (int ky = 0; ky < 3; ky++) {
        for (int kx = 0; kx < 3; kx++) {
            pixel_sum += tile[threadIdx.y + ky][threadIdx.x + kx]
                       * d_emboss_kernel[ky * 3 + kx];
        }
    }

    pixel_sum += 128.0f;
    pixel_sum  = fmaxf(0.0f, fminf(255.0f, pixel_sum));
    output[out_y * out_w + out_x] = (unsigned char)pixel_sum;
}

void cuda_emboss(const unsigned char *pixels, const size_t width, const size_t height, unsigned char *output) {
    const size_t in_size  = width * height * 3;
    const size_t out_size = (width - 2) * (height - 2);

    unsigned char *d_pixels, *d_output;
    CUDA_CALL(cudaMalloc(&d_pixels, in_size));
    CUDA_CALL(cudaMalloc(&d_output, out_size));

    CUDA_CALL(cudaMemcpy(d_pixels, pixels, in_size, cudaMemcpyHostToDevice));

    dim3 block(BLOCK_W, BLOCK_H);
    dim3 grid(((unsigned int)(width  - 2) + BLOCK_W - 1) / BLOCK_W,
              ((unsigned int)(height - 2) + BLOCK_H - 1) / BLOCK_H);

    embossKernel<<<grid, block>>>(d_pixels, (unsigned int)width, (unsigned int)height, d_output);
    CUDA_CHECK();

    CUDA_CALL(cudaMemcpy(output, d_output, out_size, cudaMemcpyDeviceToHost));

    CUDA_CALL(cudaFree(d_pixels));
    CUDA_CALL(cudaFree(d_output));
}
