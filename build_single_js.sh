#!/bin/sh 

INPUT_FILES="src/test_memcpy.cpp src/test_saxpy.cpp src/test_dot.cpp src/test_prefix_sum.cpp src/test_sort.cpp src/test_nbody.cpp src/test_convolution_2d.cpp src/test_sparse_matrix_vector.cpp src/test_dense_matrix_vector.cpp src/test_cholesky.cpp src/test_jacobi_solver.cpp src/test_gauss_seidel_solver.cpp src/test_fft.cpp src/test_lcp.cpp src/test_conjugate_gradient_solver.cpp"

LCP_TEST_FILE_LIST="--preload-file test_patterns/sample_data_1024_mu0.2.txt --preload-file test_patterns/sample_data_128_sym.txt --preload-file test_patterns/sample_data_32_mu0.8.txt --preload-file test_patterns/sample_data_64_mu0.2.txt --preload-file test_patterns/sample_data_1024_mu0.8.txt --preload-file test_patterns/sample_data_256_mu0.2.txt --preload-file test_patterns/sample_data_32_sym.txt --preload-file test_patterns/sample_data_64_mu0.8.txt --preload-file test_patterns/sample_data_1024_sym.txt --preload-file test_patterns/sample_data_256_mu0.8.txt --preload-file test_patterns/sample_data_512_mu0.2.txt --preload-file test_patterns/sample_data_64_sym.txt --preload-file test_patterns/sample_data_128_mu0.2.txt --preload-file test_patterns/sample_data_256_sym.txt --preload-file test_patterns/sample_data_512_mu0.8.txt --preload-file test_patterns/sample_data_128_mu0.8.txt --preload-file test_patterns/sample_data_32_mu0.2.txt --preload-file test_patterns/sample_data_512_sym.txt"

EMCC_COMMANDLINE="emcc -sASSERTIONS -sALLOW_MEMORY_GROWTH=1 -sSTACK_SIZE=524288 -sUSE_BOOST_HEADERS=1 ${LCP_TEST_FILE_LIST} -lembind -flto --closure 1 -O3 -ffast-math -I./src"

OUTPUT_DIR_EMCC="public"

mkdir -p ${OUTPUT_DIR_EMCC}

echo "${EMCC_COMMANDLINE} ${INPUT_FILES} -o ${OUTPUT_DIR_EMCC}/test_wasm_all.js"
${EMCC_COMMANDLINE} ${INPUT_FILES} -o ${OUTPUT_DIR_EMCC}/test_wasm_all.js

