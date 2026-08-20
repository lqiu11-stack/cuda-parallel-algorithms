COM4521/COM6521 Assignment — Parallel Computing with GPUs

Implementation of three algorithms (Count Gliders, Histogram, Emboss) using both OpenMP and CUDA, based on a provided single-threaded reference implementation.

Deadline: 5pm Thursday 21st May (Week 12) Weighting: 80% of module mark

Overview

This project extends a provided single-threaded C project (cpu.c) containing three algorithms. The task is to implement equivalent, correct, and optimised versions of each algorithm using:

OpenMP — in openmp.c
CUDA — in cuda.cu

Only these two files may be modified or submitted. No other project files should be changed, so that submissions remain compatible with the marking project.

Algorithms
1. Count Gliders

Counts the number of "gliders" (Conway's Game of Life pattern) present in a frame.

Inputs: cells (array of unsigned char), width, height
Output: integer count of gliders
A valid glider match requires a 3x3 region to match one of 16 patterns (4 stages × 4 rotations, see assignment Figure 1).
Reference implementation: cpu.c::cpu_countGliders()
Functions to implement: openmp_countGliders(), cuda_countGliders() (naming follows the assignment's _countGliders convention)
Run reference/example:
  <executable> CPU CG cg_16_in.png
2. Histogram

Note: Thrust/CUB may not be used for this algorithm.

Builds a histogram of an array of integer values.

Inputs: numbers (array of int, range [0, 255]), length, bin_width, histogram_out (pre-allocated host array)
Output: populated histogram_out
Bin assignment: bin i covers the inclusive range [i * bin_width, (i+1) * bin_width - 1]
Reference implementation: cpu.c::cpu_histogram()
Functions to implement: openmp_histogram(), cuda_histogram()
Run reference/example:
  <executable> CPU H 12 100000 10
  <executable> CPU H histogram_in.csv 10
3. Emboss

Applies an emboss convolution kernel to an RGB image, producing a greyscale embossed output.

Inputs: pixels (3-channel RGB), width, height
Output: output (1-channel greyscale, (width-2) x (height-2))
Kernel:
  -2 -1  0
  -1  0  1
   0  1  2
Result is normalised (+128) and clamped to a valid pixel range. Edge pixels (no full 3x3 neighbourhood) are excluded, so the output image is 2 pixels smaller in each dimension.
Reference implementation: cpu.c::cpu_emboss()
Functions to implement: openmp_chromaticaberration() / cuda_chromaticaberration() (per assignment's function naming for this stage)
Run reference/example:
  <executable> CPU E e_in.png e_out.png
Build
Build in Release mode configuration. The submitted code must build without errors or warnings (IntelliSense-only warnings excepted) on university managed desktops.
Do not modify or add any files other than openmp.c and cuda.cu.
Benchmarking
Always benchmark in Release mode, averaging timing over multiple runs.
Use the --bench runtime argument to repeat an algorithm's execution (default: 100 times, configurable in config.h).
Benchmark across a range of input sizes to evaluate how each implementation scales.
Constraints
Third-party libraries are restricted to those introduced in lectures/labs.
Thrust and CUB are permitted except for the Histogram algorithm.
All allocated memory must be freed to avoid memory leaks (important for --bench mode, which repeats runs).
Input/output behaviour of each algorithm must remain unchanged, even if internally redesigned for performance.
Deliverables

Submit a single zip file via Blackboard containing:

openmp.c
cuda.cu
A technical report (.pdf or .docx)
Report requirements
Max 3000 words, figures encouraged.
For each implementation (OpenMP and CUDA), cover:
A clear description of the approach
Justification for the approach
Limitations of the approach and how they were identified
Comparison against the other implementation
The CUDA section is expected to be proportionally longer, reflecting its greater share of the module's taught content.
Marking
Assessed primarily on the written report; up to 30% of the mark may be deducted if implementations fail correctness tests.
Assessment criteria: Terminology, Knowledge, Understanding, Communication, Correctness.
Learning objectives assessed:
Compare and contrast different parallel computing architectures
Implement programs for both GPU and multicore (CPU) architectures
Apply benchmarking and profiling techniques to evaluate GPU program performance
Identify performance bottlenecks and apply optimisations to improve code efficiency
Late submission: 5% deduction per working day late; more than 5 working days late is graded 0.
Partial submissions still receive marks — submit even if incomplete.
Getting Help
Use lab classes for demonstrator/module leader feedback.
Questions: course Google group COM4521-group@sheffield.ac.uk (public — do not post assignment code; ask about ideas/techniques/error messages only) or the course's anonymous Google form.
Do not email teaching assistants or the module leader directly for assignment help.
