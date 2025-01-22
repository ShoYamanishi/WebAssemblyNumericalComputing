#!/bin/sh 

TEST_TYPES="memcpy saxpy dot prefix_sum sort nbody convolution_2d sparse_matrix_vector dense_matrix_vector cholesky jacobi_solver gauss_seidel_solver fft lcp conjugate_gradient_solver"

LCP_TEST_FILE_LIST="--preload-file test_patterns/sample_data_1024_mu0.2.txt --preload-file test_patterns/sample_data_128_sym.txt --preload-file test_patterns/sample_data_32_mu0.8.txt --preload-file test_patterns/sample_data_64_mu0.2.txt --preload-file test_patterns/sample_data_1024_mu0.8.txt --preload-file test_patterns/sample_data_256_mu0.2.txt --preload-file test_patterns/sample_data_32_sym.txt --preload-file test_patterns/sample_data_64_mu0.8.txt --preload-file test_patterns/sample_data_1024_sym.txt --preload-file test_patterns/sample_data_256_mu0.8.txt --preload-file test_patterns/sample_data_512_mu0.2.txt --preload-file test_patterns/sample_data_64_sym.txt --preload-file test_patterns/sample_data_128_mu0.2.txt --preload-file test_patterns/sample_data_256_sym.txt --preload-file test_patterns/sample_data_512_mu0.8.txt --preload-file test_patterns/sample_data_128_mu0.8.txt --preload-file test_patterns/sample_data_32_mu0.2.txt --preload-file test_patterns/sample_data_512_sym.txt"

EMCC_COMMANDLINE="emcc -sASSERTIONS -sALLOW_MEMORY_GROWTH=1 -sSTACK_SIZE=524288 -lembind -flto --closure 1 -O3 -ffast-math -I./src"

OUTPUT_DIR_EMCC="public"

mkdir -p ${OUTPUT_DIR_EMCC}


for n in ${TEST_TYPES}; do  

  EXTRA_PARAM_EMCC=""

  if [ "$n" == "lcp" ]; then
    EXTRA_PARAM_EMCC="${LCP_TEST_FILE_LIST}"
  fi

  if [ "$n" == "sort" ]; then
    EXTRA_PARAM_EMCC="-sUSE_BOOST_HEADERS=1"
  fi

  echo "${EMCC_COMMANDLINE} ${EXTRA_PARAM_EMCC} src/test_${n}.cpp -o ${OUTPUT_DIR_EMCC}/${n}.js"
  ${EMCC_COMMANDLINE} ${EXTRA_PARAM_EMCC} src/test_${n}.cpp -o ${OUTPUT_DIR_EMCC}/${n}.js

  echo "cp src_html/${n}.html ${OUTPUT_DIR_EMCC}"
  cp src_html/${n}.html ${OUTPUT_DIR_EMCC}

done 

cp src_html/main_style.css ${OUTPUT_DIR_EMCC}
cp src_html/index.html ${OUTPUT_DIR_EMCC}
