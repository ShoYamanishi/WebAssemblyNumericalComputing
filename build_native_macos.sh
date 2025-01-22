#!/bin/sh 

MAC_NATIVE_COMMANDLINE="clang++ -Wall -std=c++17 -stdlib=libc++ -O3 -I/opt/homebrew/include -I./src"

TEST_TYPES="memcpy saxpy dot prefix_sum sort nbody convolution_2d sparse_matrix_vector dense_matrix_vector cholesky jacobi_solver gauss_seidel_solver fft lcp conjugate_gradient_solver"

OUTPUT_DIR_MAC_NATIVE="native_output"

mkdir -p ${OUTPUT_DIR_MAC_NATIVE}

for n in ${TEST_TYPES}; do
  
  EXTRA_PARAM_MAC_NATIVE=""

  if [ "$n" == "sort" ]; then
    EXTRA_PARAM_MAC_NATIVE="${EXTRA_PARAM} -I/opt/homebrew/include"
  fi

  echo "${MAC_NATIVE_COMMANDLINE} ${EXTRA_PARAM_MAC_NATIVE} src/test_${n}.cpp -o ${OUTPUT_DIR_MAC_NATIVE}/${n}"
  ${MAC_NATIVE_COMMANDLINE} ${EXTRA_PARAM_MAC_NATIVE} src/test_${n}.cpp -o ${OUTPUT_DIR_MAC_NATIVE}/${n}

done 

for n in ${TEST_TYPES}; do

  PARAM=""
  if [ "$#" != "0" ]; then
    PARAM="print_diag"
  fi

  echo "${OUTPUT_DIR_MAC_NATIVE}/${n} ${PARAM}"
  ${OUTPUT_DIR_MAC_NATIVE}/${n} ${PARAM}
done 
