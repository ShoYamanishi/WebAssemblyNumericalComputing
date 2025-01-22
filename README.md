# WebAssemblyNumericalComputing

A study on the numerical computing with WebAssembly in C++ on the web browsers

# Run the Test Scripts on Your Browser
You can run the performance tests on your browser.
You need to run a local server, such as `http.server` in Python3.

The `./public` directory contains all the necessary files (*.html, *.css, *.js, *.wasm. *.data ) for the local HTTP server.

For example:
```
$ cd WebAssemblyNumericalComputing/public
$ python3 -m http.server 5173  
Serving HTTP on :: port 5173 (http://[::]:5173/) ...
```

Then open 'http://localhost:5173/' on your browser.
This will get you the top-level page, which contains the links to the actual test pages.
Clicking one of those links will open a page for the specific type of computation.
The page automatically starts the test script to measure the performance.

The test script is written in C++ and compiled to WASM by Emscripten.
When it finishes, the page will update the tables with the times measured in milliseconds in the tables.

Each table is accompanied by a smaller table below.
The smaller table contains the numbers sampled from the test program compiled
from the same C++ code by clang++, and natively executed on Mac Mini M1 2020.
You can make the side-by-side comparison of the time taken on your browser and on the Mac for each problem type and the size.

# Description (work in progress)

Collection of test programs written in C++ for some types of numerical computing.
The C++ code is already compiled to WASM so that you can run and test by yourself.

The purpose is to gain insight what problem size in what time kinda thing.
which is difficult to obtain from other benchmarks and performance reports.

to amortize the overhead, and take advantage of GPU

- NEON SIMD 

- Multi-thread not considered. To exploit the power of the multi-threading
for the numerical computing, a very fine thread scheduing and synchronization are required. As the browsers do not support them, the multi-threadding is not considered.

- WebGPU or WebGL not considered for two reasons. 
One is the lack availability of WebGPU compute shaders on the browsers at the time of writing.
The other is the usefulness. Based on my experience, in reality, the application
of GPGPU is limited to
- N-Body type Particle simulation
- GPU-based collision detection
- Back-prop for the NN learning.

Series of Matrix-Matrix and Matrix-Vector multiplications and additions where the
intermediate results can stay in the GPU memory.


If you are interestedin the numerical computing on Mac and iOS devices,
please check my sister project [AppleNumericalComputing on Github](https://github.com/ShoYamanishi/AppleNumericalComputing).
It utilizes Accelerate framework and Metal compute shaders, as well as Arm NEON SIMD, and CPU multi-threadding.

- Add some implementations and experiments with WebGPU, when it matures.
N-Body and mat-vec multiplication, the overhead of using WebGPU ( data transfer, pipeline setup, invocation etc)
can not be amortized 

# Topics
Following topics are covered in this project.

- Memory Copy
- Element-wise Multiplication of Two Vectors
- Dot Product Calculation
- Prefix-Sum
- In-Place Sort
- N-Body Particle Simulation
- Convolution 2D
- Sparse Matrix-Vector Multiplication
- Dense Matrix-Vector Multiplication
- Cholesky Factorization
- Jacobi Solver
- Gauss-Seidel Solver
- Lemke LCP Solver
- Conjugate Gradient Solver
- 512-Point Radix-2 FFT

### Memory Copy
This is a simple copy of the content of a region in memory to another non-overlapping region.
Following implementations are tested.

- plain implementation in C++.
- memcpy()

The C++ test code is found in [src/test_memcpy.cpp](src/test_memcpy.cpp).

### Element-wise Multiplication of Two Vectors
This is the saxpy/daxpy type of operation. Each element-wise multiplication is totally
independent from the others. This operation can be maximally parallelized.

Following implementations are tested.

- plain implementation in C++.
- C++ with NEON SIMD.
- the reference implementation of saxpy/daxpy from CLPACK on Netlib.

The C++ test code is found in [src/test_saxpy.cpp](src/test_saxpy.cpp).

### Dot Product Calculation
This is the inner product calculation of the vector pairs of various sizes.

Following implementations are tested.

- plain implementation in C++.
- C++ with NEON SIMD.
- the reference implementation of sdot/ddot from CLPACK on Netlib.

The C++ test code is found in [src/test_dot.cpp](src/test_dot.cpp).

### Prefix-Sum
This is a scanning operation, which is parallelized.

Following implementations are tested.

- plain implementation in C++.
- std::inclusive_scan()

The C++ test code is found in [src/test_prefix_sum.cpp](src/test_prefix_sum.cpp).

### In-Place Sort
This is for the in-place sort algorithms.

Following implementations are tested.

- std::sort()
- boost::sort::spreadsort()
- boost::sort::block_indirect_sort()

The C++ test code is found in [src/test_sort.cpp](src/test_sort.cpp).

### N-Body Particle Simulation
This is for the performance of one step for the N-Body simulation,
where each of N objects interacts with all the other N - 1 objects.
It simulates a simplified particle physics simulation in 3D.
At each step, at each particle, N-1 forces are collected based on their distances,
and the velocity and the position are updated by the simple Euler step.

Following implementations are tested.

- plain implementation in C++ with the 'array of structures' data structure.
- plain implementation in C++ with the 'structure of arrays' data structure.
- C++ with NEON SIMD with the 'structure of arrays' data structure.

The C++ test code is found in [src/test_nbody.cpp](src/test_nbody.cpp).

### Convolution 2D
This is for the 2D filtering operation with the convolution with a 5x5 kernel.
This type of operation is usually done with GPU, but this topic may be still
useful to get some rough idea about what kind of performance you can get on the browser
for this type of operations. Only a plain implementation in C++ is considered.

The C++ test code is found in [src/test_convolution_2d.cpp](src/test_convolution_2d.cpp).

### Sparse Matrix-Vector Multiplication
This is for the performance of the sparse matrix-vector multiplication, where the matrix
elements are stored in the CSR ( compressed sparse row) form. Only a plain implementation in C++ is considered.

The C++ test code is found in [src/test_sparse_matrix_vector.cpp](src/test_sparse_matrix_vector.cpp).

### Dense Matrix-Vector Multiplication
This is for the performance of the dense matrix-vector multiplication.

Following implementations are tested.
- plain implementation in C++.
- C++ with NEON SIMD.
- the reference implementation of sgemv/dgemv from CLPACK on Netlib.

The C++ test code is found in [src/test_dense_matrix_vector.cpp](src/test_dense_matrix_vector.cpp).

### Cholesky Factorization
This is for the performance of the inverse operation for the PD matrices with
Cholesky factorization.

Following implementations are tested.
- plain implementations in C++.
- Eigen::LLT from Eigen3.

The C++ test code is found in [src/test_cholesky.cpp](src/test_cholesky.cpp).

### Jacobi Solver
This is for the performance of the iterative Jacobi solver.
The number of iterations is fixed to 10.

Following implementations are tested.
- plain implementations in C++.
- C++ with NEON SIMD.

The C++ test code is found in [src/test_jacobi_solver.cpp](src/test_jacobi_solver.cpp).

### Gauss-Seidel Solver
This is for the performance of the iterative Gauss-Seidel solver.
The number of iterations is fixed to 10.

Following implementations are tested.
- plain implementations in C++.
- C++ with NEON SIMD.

The C++ test code is found in [src/test_gauss_seidel_solver.cpp](src/test_gauss_seidel_solver.cpp).

### Lemke LCP Solver
This is for the performance of the Lemke LCP (Linear Complentarity Problem) solver.
The number of pivots can not be fixed, but the input data have been sampled from a velocity-space constraint-based real rigid body simulation with the hexagonal friction cone.

Following implementations are tested.
- plain implementations in C++.
- C++ with NEON SIMD.

The C++ test code is found in [src/test_lcp.cpp](src/test_lcp.cpp).

### Conjugate Gradient Solver
This is for the performance of the conjugate gradient solver.
The test data are artifically generated with the condition numbers of 10.0, 1000.0, and 100000.0. Only a plain implementation in C++ is considered.

The C++ test code is found in [src/test_conjugate_gradient_solver.cpp](src/test_conjugate_gradient_solver.cpp).

### 512-Point Radix-2 FFT
This page measures the performance of the 512-point radix-2 FFT.
This type of operation is usually done with a DSP, but this topic may be still
useful to get some rough idea about what kind of performance you can get on the browser
for this type of operations. Only a plain implementation in C++ is considered.

The C++ test code is found in [src/test_fft.cpp](src/test_fft.cpp).

# Build

## Requirements:

### Emscripten

Follow the instruction on [the official page](https://emscripten.org/).

### Eigen3 (used for Cholesky factorization)
You can skip this step, if you don't run the tests for Cholesky factorization.

- Download the zip file from the following official location:
 https://gitlab.com/libeigen/eigen/-/releases/3.4.0 on Gitlab.

- Install it to Emscripten as follows.

```
% source <path/to>/emsdk_env.sh
% cd <path/to>eigen-3.4.0
% mkdir build 
% cd build
% emcmake cmake -DCMAKE_BUILD_TYPE=Release ..
% cmake --build . --target install
```

## Build
Run the build script [build.sh](build.sh).
```
% source <path/to>/emsdk_env.sh
% cd <path/to>WebAssemblyNumericalComputing
% ./build.sh
```
It's a small convenience script to invoke emcc to compile the c++ files to WASM.
This will arrange all the necessary files in [public/](public/).

# Build for MacOS.

You can run the same c++ test programs on an MacOS for ARM.
They have the following dependencies.

- [Eigen3](https://eigen.tuxfamily.org/dox/) used by [src/test_cholesky.cpp](src/test_cholesky.cpp).
- [Boost](https://www.boost.org/) used by [src/test_sort.cpp](src/test_sort.cpp).


## Build
Run the build script [build_native_macos.sh](build_native_macos.sh).
```
% cd <path/to>WebAssemblyNumericalComputing
% ./build_native_macos.sh
```
This will generate the binaries in `native_output/`.
The boost library is assumed to be installed at `/opt/homebrew/include`.

It's a small convenience script to invoke clang++. It should work
on any other ARM environment that has a LLVM, Eigen3, and Boost.


