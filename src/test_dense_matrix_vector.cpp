#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <emscripten/bind.h>
#endif

#include <algorithm>
#include <arm_neon.h>
#include <sstream>

#include "test_case_with_time_measurements.h"
#include "test_pattern_generation.h"
#include "netlib_clapack_reference.h"

template< class T, bool IS_COL_MAJOR >
class TestCaseDenseMV : public TestCaseWithTimeMeasurements  {

static constexpr double RMS_PASSING_THRESHOLD = 1.0e-4;

protected:

    const int m_M;
    const int m_N;
    T*        m_matrix;
    T*        m_vector;
    T*        m_output_vector;

public:

    TestCaseDenseMV( const string& case_name, const int M, const int N )
        :TestCaseWithTimeMeasurements { case_name }
        ,m_M                          { M         }
        ,m_N                          { N         }
        ,m_matrix                     { nullptr   }
        ,m_vector                     { nullptr   }
        ,m_output_vector              { nullptr   }
    {
        ;
    }

    virtual ~TestCaseDenseMV()
    {
        ;
    }

    virtual void compareTruth( const T* const baseline )
    {
        auto rms = getRMSDiffTwoVectors( getOutputVector(), baseline, m_M );
        setRMS( rms );

        setTrueFalse( (rms < RMS_PASSING_THRESHOLD) ? true : false );
    }

    virtual T* getOutputVector()
    {
        return m_output_vector;
    }

    virtual void setInitialStates( T* matrix, T* vector, T* output_vector )
    {
        m_matrix        = matrix;
        m_vector        = vector;
        m_output_vector = output_vector;
    }

    virtual void run() = 0;
};


template< class T, bool IS_COL_MAJOR >
class TestCaseDenseMV_baseline : public TestCaseDenseMV< T, IS_COL_MAJOR > {

  public:

    TestCaseDenseMV_baseline( const string& case_name, const int M, const int N )
        :TestCaseDenseMV<T, IS_COL_MAJOR>{ case_name, M, N }
    {
        ;
    }

    virtual ~TestCaseDenseMV_baseline()
    {
        ;
    }

    virtual void run()
    {
        for ( int i = 0; i < this->m_M; i++ ) {

            this->m_output_vector[i] = 0.0;

            for ( int j = 0; j < this->m_N; j++ ) {

                const int mat_index = linear_index_mat<IS_COL_MAJOR>( i, j, this->m_M, this->m_N );

                this->m_output_vector[i] += ( this->m_matrix[ mat_index ] * this->m_vector[j] );
            }
        }
    }
};


template< class T, bool IS_COL_MAJOR >
class TestCaseDenseMV_blas : public TestCaseDenseMV< T, IS_COL_MAJOR > {

  public:

    TestCaseDenseMV_blas( const string& case_name, const int M, const int N )
        :TestCaseDenseMV<T, IS_COL_MAJOR>{ case_name, M, N }
    {
        ;
    }

    virtual ~TestCaseDenseMV_blas()
    {
        ;
    }

    virtual void run()
    {
        char    order{ IS_COL_MAJOR ? 'N' : 'T' };
        integer one  { 1 };
        integer M    { this->m_M };
        integer N    { this->m_N };
        T       alpha{ 1.0 };
        T       beta { 1.0 };

        if constexpr ( is_same< float,T >::value ) {

            sgemv_( &order, &M,  &N, &alpha, this->m_matrix, &M, this->m_vector, &one, &beta, this->m_output_vector, &one );
        }
        else { // is_same< double,T >::value
            dgemv_( &order, &M,  &N, &alpha, this->m_matrix, &M, this->m_vector, &one, &beta, this->m_output_vector, &one );
        }
    }
};

template< class T, bool IS_COL_MAJOR >
class TestCaseDenseMV_NEON : public TestCaseDenseMV<T, IS_COL_MAJOR> {

  protected:
    const int m_factor_loop_unrolling;

  public:

    TestCaseDenseMV_NEON( const string& case_name, const int M, const int N, const int factor_loop_unrolling )
        :TestCaseDenseMV<T, IS_COL_MAJOR>{ case_name, M, N }
        ,m_factor_loop_unrolling         { factor_loop_unrolling }
    {
        static_assert( is_same< float,T >::value || is_same< double,T >::value );
    }

    virtual ~TestCaseDenseMV_NEON()
    {
        ;
    }


    virtual void run()
    {
        calc_block( 0, this->m_M );
    }

    virtual void calc_block( const int row_begin, const int row_end_past_one )
    {
        if constexpr (IS_COL_MAJOR) {

            switch( m_factor_loop_unrolling ) {

              case 1:
                run_col_major_loop_unrolling_1( row_begin, row_end_past_one );
                break;

              case 2:
                run_col_major_loop_unrolling_2( row_begin, row_end_past_one );
                break;

              case 4:
                run_col_major_loop_unrolling_4( row_begin, row_end_past_one );
                break;

              case 8:
              default:
                run_col_major_loop_unrolling_8( row_begin, row_end_past_one );
                break;
            }
        }
        else {
            switch( m_factor_loop_unrolling ) {

              case 1:
                run_row_major_loop_unrolling_1( row_begin, row_end_past_one );
                break;

              case 2:
                run_row_major_loop_unrolling_2( row_begin, row_end_past_one );
                break;

              case 4:
                run_row_major_loop_unrolling_4( row_begin, row_end_past_one );
                break;

              case 8:
              default:
                run_row_major_loop_unrolling_8( row_begin, row_end_past_one );
                break;
            }
        }
    }

    virtual void run_col_major_loop_unrolling_1( const int row_begin, const int row_end_past_one )
    {
        if constexpr ( is_same< float,T >::value ) {

            for ( int i = row_begin; i < row_end_past_one; i += 4 ) {

                float32x4_t qw_row_sum1 = { 0.0, 0.0, 0.0, 0.0 };

                for ( int j = 0; j < this->m_N; j++ ) {

                    const int mat_index1 = linear_index_mat<IS_COL_MAJOR>( i, j, this->m_M, this->m_N );
                    const float col_v = this->m_vector[j];
                    const float32x4_t qw_mat1 = vld1q_f32( &(this->m_matrix[ mat_index1 ]) );
                    const float32x4_t qw_col = { col_v, col_v, col_v, col_v };
                    const float32x4_t qw_mc1  = vmulq_f32( qw_mat1, qw_col );

                    qw_row_sum1 = vaddq_f32( qw_mc1, qw_row_sum1 );
                }

                memcpy(&(this->m_output_vector[i  ]), &qw_row_sum1, sizeof(float)*4);
            }
        }
        else {

            for ( int i = row_begin; i < row_end_past_one; i += 2 ) {

                float64x2_t qw_row_sum1 = { 0.0, 0.0 };

                for ( int j = 0; j < this->m_N; j++ ) {

                    const int mat_index1 = linear_index_mat<IS_COL_MAJOR>( i, j, this->m_M, this->m_N );
                    const float col_v = this->m_vector[j];
                    const float64x2_t qw_mat1 = vld1q_f64( &(this->m_matrix[ mat_index1 ]) );
                    const float64x2_t qw_col = { col_v, col_v };
                    const float64x2_t qw_mc1  = vmulq_f64( qw_mat1, qw_col );

                    qw_row_sum1 = vaddq_f64( qw_mc1, qw_row_sum1 );
                }

                memcpy(&(this->m_output_vector[i  ]), &qw_row_sum1, sizeof(double)*2);
            }
        }
    }

    virtual void run_col_major_loop_unrolling_2( const int row_begin, const int row_end_past_one )
    {
        if constexpr ( is_same< float,T >::value ) {

            for ( int i = row_begin; i < row_end_past_one; i += 8 ) {

                float32x4_t qw_row_sum1 = { 0.0, 0.0, 0.0, 0.0 };
                float32x4_t qw_row_sum2 = { 0.0, 0.0, 0.0, 0.0 };

                for ( int j = 0; j < this->m_N; j++ ) {

                    const int mat_index1 = linear_index_mat<IS_COL_MAJOR>( i, j, this->m_M, this->m_N );
                    const int mat_index2 = mat_index1 + 4 ;
                    const float col_v = this->m_vector[j];
                    const float32x4_t qw_mat1 = vld1q_f32( &(this->m_matrix[ mat_index1 ]) );
                    const float32x4_t qw_mat2 = vld1q_f32( &(this->m_matrix[ mat_index2 ]) );
                    const float32x4_t qw_col = { col_v, col_v, col_v, col_v };
                    const float32x4_t qw_mc1  = vmulq_f32( qw_mat1, qw_col );
                    const float32x4_t qw_mc2  = vmulq_f32( qw_mat2, qw_col );

                    qw_row_sum1 = vaddq_f32( qw_mc1, qw_row_sum1 );
                    qw_row_sum2 = vaddq_f32( qw_mc2, qw_row_sum2 );
                }

                memcpy(&(this->m_output_vector[i  ]), &qw_row_sum1, sizeof(float)*4);
                memcpy(&(this->m_output_vector[i+4]), &qw_row_sum2, sizeof(float)*4);
            }
        }
        else {

            for ( int i = row_begin; i < row_end_past_one; i += 4 ) {

                float64x2_t qw_row_sum1 = { 0.0, 0.0 };
                float64x2_t qw_row_sum2 = { 0.0, 0.0 };

                for ( int j = 0; j < this->m_N; j++ ) {

                    const int mat_index1 = linear_index_mat<IS_COL_MAJOR>( i, j, this->m_M, this->m_N );
                    const int mat_index2 = mat_index1 + 2 ;
                    const float col_v = this->m_vector[j];
                    const float64x2_t qw_mat1 = vld1q_f64( &(this->m_matrix[ mat_index1 ]) );
                    const float64x2_t qw_mat2 = vld1q_f64( &(this->m_matrix[ mat_index2 ]) );
                    const float64x2_t qw_col = { col_v, col_v };
                    const float64x2_t qw_mc1  = vmulq_f64( qw_mat1, qw_col );
                    const float64x2_t qw_mc2  = vmulq_f64( qw_mat2, qw_col );

                    qw_row_sum1 = vaddq_f64( qw_mc1, qw_row_sum1 );
                    qw_row_sum2 = vaddq_f64( qw_mc2, qw_row_sum2 );
                }

                memcpy(&(this->m_output_vector[i  ]), &qw_row_sum1, sizeof(double)*2);
                memcpy(&(this->m_output_vector[i+2]), &qw_row_sum2, sizeof(double)*2);
            }
        }
    }

    virtual void run_col_major_loop_unrolling_4( const int row_begin, const int row_end_past_one )
    {
        if constexpr ( is_same< float,T >::value ) {

            for ( int i = row_begin; i < row_end_past_one; i += 16 ) {

                float32x4_t qw_row_sum1 = { 0.0, 0.0, 0.0, 0.0 };
                float32x4_t qw_row_sum2 = { 0.0, 0.0, 0.0, 0.0 };
                float32x4_t qw_row_sum3 = { 0.0, 0.0, 0.0, 0.0 };
                float32x4_t qw_row_sum4 = { 0.0, 0.0, 0.0, 0.0 };

                for ( int j = 0; j < this->m_N; j++ ) {

                    const int mat_index1 = linear_index_mat<IS_COL_MAJOR>( i, j, this->m_M, this->m_N );
                    const int mat_index2 = mat_index1 +  4 ;
                    const int mat_index3 = mat_index1 +  8 ;
                    const int mat_index4 = mat_index1 + 12 ;
                    const float col_v = this->m_vector[j];
                    const float32x4_t qw_mat1 = vld1q_f32( &(this->m_matrix[ mat_index1 ]) );
                    const float32x4_t qw_mat2 = vld1q_f32( &(this->m_matrix[ mat_index2 ]) );
                    const float32x4_t qw_mat3 = vld1q_f32( &(this->m_matrix[ mat_index3 ]) );
                    const float32x4_t qw_mat4 = vld1q_f32( &(this->m_matrix[ mat_index4 ]) );
                    const float32x4_t qw_col = { col_v, col_v, col_v, col_v };
                    const float32x4_t qw_mc1  = vmulq_f32( qw_mat1, qw_col );
                    const float32x4_t qw_mc2  = vmulq_f32( qw_mat2, qw_col );
                    const float32x4_t qw_mc3  = vmulq_f32( qw_mat3, qw_col );
                    const float32x4_t qw_mc4  = vmulq_f32( qw_mat4, qw_col );

                    qw_row_sum1 = vaddq_f32( qw_mc1, qw_row_sum1 );
                    qw_row_sum2 = vaddq_f32( qw_mc2, qw_row_sum2 );
                    qw_row_sum3 = vaddq_f32( qw_mc3, qw_row_sum3 );
                    qw_row_sum4 = vaddq_f32( qw_mc4, qw_row_sum4 );
                }

                memcpy(&(this->m_output_vector[i   ]), &qw_row_sum1, sizeof(float)*4);
                memcpy(&(this->m_output_vector[i+ 4]), &qw_row_sum2, sizeof(float)*4);
                memcpy(&(this->m_output_vector[i+ 8]), &qw_row_sum3, sizeof(float)*4);
                memcpy(&(this->m_output_vector[i+12]), &qw_row_sum4, sizeof(float)*4);
            }
        }
        else {

            for ( int i = row_begin; i < row_end_past_one; i += 8 ) {

                float64x2_t qw_row_sum1 = { 0.0, 0.0 };
                float64x2_t qw_row_sum2 = { 0.0, 0.0 };
                float64x2_t qw_row_sum3 = { 0.0, 0.0 };
                float64x2_t qw_row_sum4 = { 0.0, 0.0 };

                for ( int j = 0; j < this->m_N; j++ ) {

                    const int mat_index1 = linear_index_mat<IS_COL_MAJOR>( i, j, this->m_M, this->m_N );
                    const int mat_index2 = mat_index1 + 2 ;
                    const int mat_index3 = mat_index1 + 4 ;
                    const int mat_index4 = mat_index1 + 6 ;
                    const float col_v = this->m_vector[j];
                    const float64x2_t qw_mat1 = vld1q_f64( &(this->m_matrix[ mat_index1 ]) );
                    const float64x2_t qw_mat2 = vld1q_f64( &(this->m_matrix[ mat_index2 ]) );
                    const float64x2_t qw_mat3 = vld1q_f64( &(this->m_matrix[ mat_index3 ]) );
                    const float64x2_t qw_mat4 = vld1q_f64( &(this->m_matrix[ mat_index4 ]) );
                    const float64x2_t qw_col = { col_v, col_v };
                    const float64x2_t qw_mc1  = vmulq_f64( qw_mat1, qw_col );
                    const float64x2_t qw_mc2  = vmulq_f64( qw_mat2, qw_col );
                    const float64x2_t qw_mc3  = vmulq_f64( qw_mat3, qw_col );
                    const float64x2_t qw_mc4  = vmulq_f64( qw_mat4, qw_col );

                    qw_row_sum1 = vaddq_f64( qw_mc1, qw_row_sum1 );
                    qw_row_sum2 = vaddq_f64( qw_mc2, qw_row_sum2 );
                    qw_row_sum3 = vaddq_f64( qw_mc3, qw_row_sum3 );
                    qw_row_sum4 = vaddq_f64( qw_mc4, qw_row_sum4 );
                }

                memcpy(&(this->m_output_vector[i  ]), &qw_row_sum1, sizeof(double)*2);
                memcpy(&(this->m_output_vector[i+2]), &qw_row_sum2, sizeof(double)*2);
                memcpy(&(this->m_output_vector[i+4]), &qw_row_sum3, sizeof(double)*2);
                memcpy(&(this->m_output_vector[i+6]), &qw_row_sum4, sizeof(double)*2);
            }
        }
    }


    virtual void run_col_major_loop_unrolling_8( const int row_begin, const int row_end_past_one )
    {
        if constexpr ( is_same< float,T >::value ) {

            for ( int i = row_begin; i < row_end_past_one; i += 32 ) {

                float32x4_t qw_row_sum1 = { 0.0, 0.0, 0.0, 0.0 };
                float32x4_t qw_row_sum2 = { 0.0, 0.0, 0.0, 0.0 };
                float32x4_t qw_row_sum3 = { 0.0, 0.0, 0.0, 0.0 };
                float32x4_t qw_row_sum4 = { 0.0, 0.0, 0.0, 0.0 };
                float32x4_t qw_row_sum5 = { 0.0, 0.0, 0.0, 0.0 };
                float32x4_t qw_row_sum6 = { 0.0, 0.0, 0.0, 0.0 };
                float32x4_t qw_row_sum7 = { 0.0, 0.0, 0.0, 0.0 };
                float32x4_t qw_row_sum8 = { 0.0, 0.0, 0.0, 0.0 };

                for ( int j = 0; j < this->m_N; j++ ) {

                    const int mat_index1 = linear_index_mat<IS_COL_MAJOR>( i, j, this->m_M, this->m_N );
                    const int mat_index2 = mat_index1 +  4 ;
                    const int mat_index3 = mat_index1 +  8 ;
                    const int mat_index4 = mat_index1 + 12 ;
                    const int mat_index5 = mat_index1 + 16 ;
                    const int mat_index6 = mat_index1 + 20 ;
                    const int mat_index7 = mat_index1 + 24 ;
                    const int mat_index8 = mat_index1 + 28 ;
                    const float col_v = this->m_vector[j];
                    const float32x4_t qw_mat1 = vld1q_f32( &(this->m_matrix[ mat_index1 ]) );
                    const float32x4_t qw_mat2 = vld1q_f32( &(this->m_matrix[ mat_index2 ]) );
                    const float32x4_t qw_mat3 = vld1q_f32( &(this->m_matrix[ mat_index3 ]) );
                    const float32x4_t qw_mat4 = vld1q_f32( &(this->m_matrix[ mat_index4 ]) );
                    const float32x4_t qw_mat5 = vld1q_f32( &(this->m_matrix[ mat_index5 ]) );
                    const float32x4_t qw_mat6 = vld1q_f32( &(this->m_matrix[ mat_index6 ]) );
                    const float32x4_t qw_mat7 = vld1q_f32( &(this->m_matrix[ mat_index7 ]) );
                    const float32x4_t qw_mat8 = vld1q_f32( &(this->m_matrix[ mat_index8 ]) );
                    const float32x4_t qw_col = { col_v, col_v, col_v, col_v };
                    const float32x4_t qw_mc1  = vmulq_f32( qw_mat1, qw_col );
                    const float32x4_t qw_mc2  = vmulq_f32( qw_mat2, qw_col );
                    const float32x4_t qw_mc3  = vmulq_f32( qw_mat3, qw_col );
                    const float32x4_t qw_mc4  = vmulq_f32( qw_mat4, qw_col );
                    const float32x4_t qw_mc5  = vmulq_f32( qw_mat5, qw_col );
                    const float32x4_t qw_mc6  = vmulq_f32( qw_mat6, qw_col );
                    const float32x4_t qw_mc7  = vmulq_f32( qw_mat7, qw_col );
                    const float32x4_t qw_mc8  = vmulq_f32( qw_mat8, qw_col );

                    qw_row_sum1 = vaddq_f32( qw_mc1, qw_row_sum1 );
                    qw_row_sum2 = vaddq_f32( qw_mc2, qw_row_sum2 );
                    qw_row_sum3 = vaddq_f32( qw_mc3, qw_row_sum3 );
                    qw_row_sum4 = vaddq_f32( qw_mc4, qw_row_sum4 );
                    qw_row_sum5 = vaddq_f32( qw_mc5, qw_row_sum5 );
                    qw_row_sum6 = vaddq_f32( qw_mc6, qw_row_sum6 );
                    qw_row_sum7 = vaddq_f32( qw_mc7, qw_row_sum7 );
                    qw_row_sum8 = vaddq_f32( qw_mc8, qw_row_sum8 );
                }

                memcpy(&(this->m_output_vector[i   ]), &qw_row_sum1, sizeof(float)*4);
                memcpy(&(this->m_output_vector[i+ 4]), &qw_row_sum2, sizeof(float)*4);
                memcpy(&(this->m_output_vector[i+ 8]), &qw_row_sum3, sizeof(float)*4);
                memcpy(&(this->m_output_vector[i+12]), &qw_row_sum4, sizeof(float)*4);
                memcpy(&(this->m_output_vector[i+16]), &qw_row_sum5, sizeof(float)*4);
                memcpy(&(this->m_output_vector[i+20]), &qw_row_sum6, sizeof(float)*4);
                memcpy(&(this->m_output_vector[i+24]), &qw_row_sum7, sizeof(float)*4);
                memcpy(&(this->m_output_vector[i+28]), &qw_row_sum8, sizeof(float)*4);
            }
        }
        else {

            for ( int i = row_begin; i < row_end_past_one; i += 16 ) {

                float64x2_t qw_row_sum1 = { 0.0, 0.0 };
                float64x2_t qw_row_sum2 = { 0.0, 0.0 };
                float64x2_t qw_row_sum3 = { 0.0, 0.0 };
                float64x2_t qw_row_sum4 = { 0.0, 0.0 };
                float64x2_t qw_row_sum5 = { 0.0, 0.0 };
                float64x2_t qw_row_sum6 = { 0.0, 0.0 };
                float64x2_t qw_row_sum7 = { 0.0, 0.0 };
                float64x2_t qw_row_sum8 = { 0.0, 0.0 };

                for ( int j = 0; j < this->m_N; j++ ) {

                    const int mat_index1 = linear_index_mat<IS_COL_MAJOR>( i, j, this->m_M, this->m_N );
                    const int mat_index2 = mat_index1 +  2 ;
                    const int mat_index3 = mat_index1 +  4 ;
                    const int mat_index4 = mat_index1 +  6 ;
                    const int mat_index5 = mat_index1 +  8 ;
                    const int mat_index6 = mat_index1 + 10 ;
                    const int mat_index7 = mat_index1 + 12 ;
                    const int mat_index8 = mat_index1 + 14 ;
                    const float col_v = this->m_vector[j];
                    const float64x2_t qw_mat1 = vld1q_f64( &(this->m_matrix[ mat_index1 ]) );
                    const float64x2_t qw_mat2 = vld1q_f64( &(this->m_matrix[ mat_index2 ]) );
                    const float64x2_t qw_mat3 = vld1q_f64( &(this->m_matrix[ mat_index3 ]) );
                    const float64x2_t qw_mat4 = vld1q_f64( &(this->m_matrix[ mat_index4 ]) );
                    const float64x2_t qw_mat5 = vld1q_f64( &(this->m_matrix[ mat_index5 ]) );
                    const float64x2_t qw_mat6 = vld1q_f64( &(this->m_matrix[ mat_index6 ]) );
                    const float64x2_t qw_mat7 = vld1q_f64( &(this->m_matrix[ mat_index7 ]) );
                    const float64x2_t qw_mat8 = vld1q_f64( &(this->m_matrix[ mat_index8 ]) );
                    const float64x2_t qw_col = { col_v, col_v };
                    const float64x2_t qw_mc1  = vmulq_f64( qw_mat1, qw_col );
                    const float64x2_t qw_mc2  = vmulq_f64( qw_mat2, qw_col );
                    const float64x2_t qw_mc3  = vmulq_f64( qw_mat3, qw_col );
                    const float64x2_t qw_mc4  = vmulq_f64( qw_mat4, qw_col );
                    const float64x2_t qw_mc5  = vmulq_f64( qw_mat5, qw_col );
                    const float64x2_t qw_mc6  = vmulq_f64( qw_mat6, qw_col );
                    const float64x2_t qw_mc7  = vmulq_f64( qw_mat7, qw_col );
                    const float64x2_t qw_mc8  = vmulq_f64( qw_mat8, qw_col );

                    qw_row_sum1 = vaddq_f64( qw_mc1, qw_row_sum1 );
                    qw_row_sum2 = vaddq_f64( qw_mc2, qw_row_sum2 );
                    qw_row_sum3 = vaddq_f64( qw_mc3, qw_row_sum3 );
                    qw_row_sum4 = vaddq_f64( qw_mc4, qw_row_sum4 );
                    qw_row_sum5 = vaddq_f64( qw_mc5, qw_row_sum5 );
                    qw_row_sum6 = vaddq_f64( qw_mc6, qw_row_sum6 );
                    qw_row_sum7 = vaddq_f64( qw_mc7, qw_row_sum7 );
                    qw_row_sum8 = vaddq_f64( qw_mc8, qw_row_sum8 );
                }

                memcpy(&(this->m_output_vector[i   ]), &qw_row_sum1, sizeof(double)*2);
                memcpy(&(this->m_output_vector[i+ 2]), &qw_row_sum2, sizeof(double)*2);
                memcpy(&(this->m_output_vector[i+ 4]), &qw_row_sum3, sizeof(double)*2);
                memcpy(&(this->m_output_vector[i+ 6]), &qw_row_sum4, sizeof(double)*2);
                memcpy(&(this->m_output_vector[i+ 8]), &qw_row_sum5, sizeof(double)*2);
                memcpy(&(this->m_output_vector[i+10]), &qw_row_sum6, sizeof(double)*2);
                memcpy(&(this->m_output_vector[i+12]), &qw_row_sum7, sizeof(double)*2);
                memcpy(&(this->m_output_vector[i+14]), &qw_row_sum8, sizeof(double)*2);
            }
        }
    }


    virtual void run_row_major_loop_unrolling_1( const int row_begin, const int row_end_past_one )
    {
        if constexpr ( is_same< float,T >::value ) {

            for ( int i = row_begin; i < row_end_past_one; i++ ) {

                float32x4_t qw_lanewise_sum1 = { 0.0, 0.0, 0.0, 0.0 };

                for ( int j = 0; j < this->m_N; j+=4 ) {

                    const int mat_index1 = linear_index_mat<IS_COL_MAJOR>( i, j, this->m_M, this->m_N );
                    const float32x4_t qw_mat1 = vld1q_f32( &(this->m_matrix[ mat_index1 ] ) );
                    const float32x4_t qw_col1 = vld1q_f32( &(this->m_vector[ j     ]      ) );
                    const float32x4_t qw_mc1  = vmulq_f32( qw_mat1, qw_col1 );

                    qw_lanewise_sum1 = vaddq_f32( qw_mc1, qw_lanewise_sum1 );
                }
                this->m_output_vector[i] = qw_lanewise_sum1[0] + qw_lanewise_sum1[1] + qw_lanewise_sum1[2] + qw_lanewise_sum1[3];
            }
        }
        else {
            for ( int i = row_begin; i < row_end_past_one; i++ ) {

                float64x2_t qw_lanewise_sum1 = { 0.0, 0.0 };

                for ( int j = 0; j < this->m_N; j+=2 ) {

                    const int mat_index1 = linear_index_mat<IS_COL_MAJOR>( i, j, this->m_M, this->m_N );
                    const float64x2_t qw_mat1 = vld1q_f64( &(this->m_matrix[ mat_index1 ] ) );
                    const float64x2_t qw_col1 = vld1q_f64( &(this->m_vector[ j     ]      ) );
                    const float64x2_t qw_mc1  = vmulq_f64( qw_mat1, qw_col1 );

                    qw_lanewise_sum1 = vaddq_f64( qw_mc1, qw_lanewise_sum1 );
                }
                this->m_output_vector[i] = qw_lanewise_sum1[0] + qw_lanewise_sum1[1];
            }
        }
    }


    virtual void run_row_major_loop_unrolling_2( const int row_begin, const int row_end_past_one )
    {
        if constexpr ( is_same< float,T >::value ) {

            for ( int i = row_begin; i < row_end_past_one; i++ ) {

                float32x4_t qw_lanewise_sum1 = { 0.0, 0.0, 0.0, 0.0 };
                float32x4_t qw_lanewise_sum2 = { 0.0, 0.0, 0.0, 0.0 };

                for ( int j = 0; j < this->m_N; j+=8 ) {

                    const int mat_index1 = linear_index_mat<IS_COL_MAJOR>( i, j, this->m_M, this->m_N );
                    const int mat_index2 = mat_index1 + 4;
                    const float32x4_t qw_mat1 = vld1q_f32( &(this->m_matrix[ mat_index1 ] ) );
                    const float32x4_t qw_mat2 = vld1q_f32( &(this->m_matrix[ mat_index2 ] ) );
                    const float32x4_t qw_col1 = vld1q_f32( &(this->m_vector[ j     ]      ) );
                    const float32x4_t qw_col2 = vld1q_f32( &(this->m_vector[ j + 4 ]      ) );
                    const float32x4_t qw_mc1  = vmulq_f32( qw_mat1, qw_col1 );
                    const float32x4_t qw_mc2  = vmulq_f32( qw_mat2, qw_col2 );

                    qw_lanewise_sum1 = vaddq_f32( qw_mc1, qw_lanewise_sum1 );
                    qw_lanewise_sum2 = vaddq_f32( qw_mc2, qw_lanewise_sum2 );


                }
                this->m_output_vector[i] =   qw_lanewise_sum1[0] + qw_lanewise_sum1[1] + qw_lanewise_sum1[2] + qw_lanewise_sum1[3]
                                           + qw_lanewise_sum2[0] + qw_lanewise_sum2[1] + qw_lanewise_sum2[2] + qw_lanewise_sum2[3];
            }
        }
        else {
            for ( int i = row_begin; i < row_end_past_one; i++ ) {

                float64x2_t qw_lanewise_sum1 = { 0.0, 0.0 };
                float64x2_t qw_lanewise_sum2 = { 0.0, 0.0 };

                for ( int j = 0; j < this->m_N; j+=4 ) {

                    const int mat_index1 = linear_index_mat<IS_COL_MAJOR>( i, j, this->m_M, this->m_N );
                    const int mat_index2 = mat_index1 + 2;
                    const float64x2_t qw_mat1 = vld1q_f64( &(this->m_matrix[ mat_index1 ] ) );
                    const float64x2_t qw_mat2 = vld1q_f64( &(this->m_matrix[ mat_index2 ] ) );
                    const float64x2_t qw_col1 = vld1q_f64( &(this->m_vector[ j     ]      ) );
                    const float64x2_t qw_col2 = vld1q_f64( &(this->m_vector[ j + 2 ]      ) );
                    const float64x2_t qw_mc1  = vmulq_f64( qw_mat1, qw_col1 );
                    const float64x2_t qw_mc2  = vmulq_f64( qw_mat2, qw_col2 );

                    qw_lanewise_sum1 = vaddq_f64( qw_mc1, qw_lanewise_sum1 );
                    qw_lanewise_sum2 = vaddq_f64( qw_mc2, qw_lanewise_sum2 );
                }
                this->m_output_vector[i] = qw_lanewise_sum1[0] + qw_lanewise_sum1[1] + qw_lanewise_sum2[0] + qw_lanewise_sum2[1];
            }
        }
    }

    virtual void run_row_major_loop_unrolling_4( const int row_begin, const int row_end_past_one )
    {
        if constexpr ( is_same< float,T >::value ) {

            for ( int i = row_begin; i < row_end_past_one; i++ ) {

                float32x4_t qw_lanewise_sum1 = { 0.0, 0.0, 0.0, 0.0 };
                float32x4_t qw_lanewise_sum2 = { 0.0, 0.0, 0.0, 0.0 };
                float32x4_t qw_lanewise_sum3 = { 0.0, 0.0, 0.0, 0.0 };
                float32x4_t qw_lanewise_sum4 = { 0.0, 0.0, 0.0, 0.0 };

                for ( int j = 0; j < this->m_N; j+=16 ) {

                    const int mat_index1 = linear_index_mat<IS_COL_MAJOR>( i, j, this->m_M, this->m_N );
                    const int mat_index2 = mat_index1 +  4;
                    const int mat_index3 = mat_index1 +  8;
                    const int mat_index4 = mat_index1 + 12;
                    const float32x4_t qw_mat1 = vld1q_f32( &(this->m_matrix[ mat_index1 ] ) );
                    const float32x4_t qw_mat2 = vld1q_f32( &(this->m_matrix[ mat_index2 ] ) );
                    const float32x4_t qw_mat3 = vld1q_f32( &(this->m_matrix[ mat_index3 ] ) );
                    const float32x4_t qw_mat4 = vld1q_f32( &(this->m_matrix[ mat_index4 ] ) );
                    const float32x4_t qw_col1 = vld1q_f32( &(this->m_vector[ j      ]      ) );
                    const float32x4_t qw_col2 = vld1q_f32( &(this->m_vector[ j +  4 ]      ) );
                    const float32x4_t qw_col3 = vld1q_f32( &(this->m_vector[ j +  8 ]      ) );
                    const float32x4_t qw_col4 = vld1q_f32( &(this->m_vector[ j + 12 ]      ) );
                    const float32x4_t qw_mc1  = vmulq_f32( qw_mat1, qw_col1 );
                    const float32x4_t qw_mc2  = vmulq_f32( qw_mat2, qw_col2 );
                    const float32x4_t qw_mc3  = vmulq_f32( qw_mat3, qw_col3 );
                    const float32x4_t qw_mc4  = vmulq_f32( qw_mat4, qw_col4 );

                    qw_lanewise_sum1 = vaddq_f32( qw_mc1, qw_lanewise_sum1 );
                    qw_lanewise_sum2 = vaddq_f32( qw_mc2, qw_lanewise_sum2 );
                    qw_lanewise_sum3 = vaddq_f32( qw_mc3, qw_lanewise_sum3 );
                    qw_lanewise_sum4 = vaddq_f32( qw_mc4, qw_lanewise_sum4 );
                }
                this->m_output_vector[i] =   qw_lanewise_sum1[0] + qw_lanewise_sum1[1] + qw_lanewise_sum1[2] + qw_lanewise_sum1[3]
                                           + qw_lanewise_sum2[0] + qw_lanewise_sum2[1] + qw_lanewise_sum2[2] + qw_lanewise_sum2[3]
                                           + qw_lanewise_sum3[0] + qw_lanewise_sum3[1] + qw_lanewise_sum3[2] + qw_lanewise_sum3[3]
                                           + qw_lanewise_sum4[0] + qw_lanewise_sum4[1] + qw_lanewise_sum4[2] + qw_lanewise_sum4[3];
            }
        }
        else {
            for ( int i = row_begin; i < row_end_past_one; i++ ) {

                float64x2_t qw_lanewise_sum1 = { 0.0, 0.0 };
                float64x2_t qw_lanewise_sum2 = { 0.0, 0.0 };
                float64x2_t qw_lanewise_sum3 = { 0.0, 0.0 };
                float64x2_t qw_lanewise_sum4 = { 0.0, 0.0 };

                for ( int j = 0; j < this->m_N; j+=8 ) {

                    const int mat_index1 = linear_index_mat<IS_COL_MAJOR>( i, j, this->m_M, this->m_N );
                    const int mat_index2 = mat_index1 + 2;
                    const int mat_index3 = mat_index1 + 4;
                    const int mat_index4 = mat_index1 + 6;
                    const float64x2_t qw_mat1 = vld1q_f64( &(this->m_matrix[ mat_index1 ] ) );
                    const float64x2_t qw_mat2 = vld1q_f64( &(this->m_matrix[ mat_index2 ] ) );
                    const float64x2_t qw_mat3 = vld1q_f64( &(this->m_matrix[ mat_index3 ] ) );
                    const float64x2_t qw_mat4 = vld1q_f64( &(this->m_matrix[ mat_index4 ] ) );
                    const float64x2_t qw_col1 = vld1q_f64( &(this->m_vector[ j     ]      ) );
                    const float64x2_t qw_col2 = vld1q_f64( &(this->m_vector[ j + 2 ]      ) );
                    const float64x2_t qw_col3 = vld1q_f64( &(this->m_vector[ j + 4 ]      ) );
                    const float64x2_t qw_col4 = vld1q_f64( &(this->m_vector[ j + 6 ]      ) );
                    const float64x2_t qw_mc1  = vmulq_f64( qw_mat1, qw_col1 );
                    const float64x2_t qw_mc2  = vmulq_f64( qw_mat2, qw_col2 );
                    const float64x2_t qw_mc3  = vmulq_f64( qw_mat3, qw_col3 );
                    const float64x2_t qw_mc4  = vmulq_f64( qw_mat4, qw_col4 );

                    qw_lanewise_sum1 = vaddq_f64( qw_mc1, qw_lanewise_sum1 );
                    qw_lanewise_sum2 = vaddq_f64( qw_mc2, qw_lanewise_sum2 );
                    qw_lanewise_sum3 = vaddq_f64( qw_mc3, qw_lanewise_sum3 );
                    qw_lanewise_sum4 = vaddq_f64( qw_mc4, qw_lanewise_sum4 );
                }
                this->m_output_vector[i] =   qw_lanewise_sum1[0] + qw_lanewise_sum1[1] + qw_lanewise_sum2[0] + qw_lanewise_sum2[1]
                                           + qw_lanewise_sum3[0] + qw_lanewise_sum3[1] + qw_lanewise_sum4[0] + qw_lanewise_sum4[1];
            }
        }
    }


    virtual void run_row_major_loop_unrolling_8( const int row_begin, const int row_end_past_one )
    {
        if constexpr ( is_same< float,T >::value ) {

            for ( int i = row_begin; i < row_end_past_one; i++ ) {

                float32x4_t qw_lanewise_sum1 = { 0.0, 0.0, 0.0, 0.0 };
                float32x4_t qw_lanewise_sum2 = { 0.0, 0.0, 0.0, 0.0 };
                float32x4_t qw_lanewise_sum3 = { 0.0, 0.0, 0.0, 0.0 };
                float32x4_t qw_lanewise_sum4 = { 0.0, 0.0, 0.0, 0.0 };
                float32x4_t qw_lanewise_sum5 = { 0.0, 0.0, 0.0, 0.0 };
                float32x4_t qw_lanewise_sum6 = { 0.0, 0.0, 0.0, 0.0 };
                float32x4_t qw_lanewise_sum7 = { 0.0, 0.0, 0.0, 0.0 };
                float32x4_t qw_lanewise_sum8 = { 0.0, 0.0, 0.0, 0.0 };

                for ( int j = 0; j < this->m_N; j+=32 ) {

                    const int mat_index1 = linear_index_mat<IS_COL_MAJOR>( i, j, this->m_M, this->m_N );
                    const int mat_index2 = mat_index1 +  4;
                    const int mat_index3 = mat_index1 +  8;
                    const int mat_index4 = mat_index1 + 12;
                    const int mat_index5 = mat_index1 + 16;
                    const int mat_index6 = mat_index1 + 20;
                    const int mat_index7 = mat_index1 + 24;
                    const int mat_index8 = mat_index1 + 28;
                    const float32x4_t qw_mat1 = vld1q_f32( &(this->m_matrix[ mat_index1 ] ) );
                    const float32x4_t qw_mat2 = vld1q_f32( &(this->m_matrix[ mat_index2 ] ) );
                    const float32x4_t qw_mat3 = vld1q_f32( &(this->m_matrix[ mat_index3 ] ) );
                    const float32x4_t qw_mat4 = vld1q_f32( &(this->m_matrix[ mat_index4 ] ) );
                    const float32x4_t qw_mat5 = vld1q_f32( &(this->m_matrix[ mat_index5 ] ) );
                    const float32x4_t qw_mat6 = vld1q_f32( &(this->m_matrix[ mat_index6 ] ) );
                    const float32x4_t qw_mat7 = vld1q_f32( &(this->m_matrix[ mat_index7 ] ) );
                    const float32x4_t qw_mat8 = vld1q_f32( &(this->m_matrix[ mat_index8 ] ) );
                    const float32x4_t qw_col1 = vld1q_f32( &(this->m_vector[ j      ]     ) );
                    const float32x4_t qw_col2 = vld1q_f32( &(this->m_vector[ j +  4 ]     ) );
                    const float32x4_t qw_col3 = vld1q_f32( &(this->m_vector[ j +  8 ]     ) );
                    const float32x4_t qw_col4 = vld1q_f32( &(this->m_vector[ j + 12 ]     ) );
                    const float32x4_t qw_col5 = vld1q_f32( &(this->m_vector[ j + 16 ]     ) );
                    const float32x4_t qw_col6 = vld1q_f32( &(this->m_vector[ j + 20 ]     ) );
                    const float32x4_t qw_col7 = vld1q_f32( &(this->m_vector[ j + 24 ]     ) );
                    const float32x4_t qw_col8 = vld1q_f32( &(this->m_vector[ j + 28 ]     ) );
                    const float32x4_t qw_mc1  = vmulq_f32( qw_mat1, qw_col1 );
                    const float32x4_t qw_mc2  = vmulq_f32( qw_mat2, qw_col2 );
                    const float32x4_t qw_mc3  = vmulq_f32( qw_mat3, qw_col3 );
                    const float32x4_t qw_mc4  = vmulq_f32( qw_mat4, qw_col4 );
                    const float32x4_t qw_mc5  = vmulq_f32( qw_mat5, qw_col5 );
                    const float32x4_t qw_mc6  = vmulq_f32( qw_mat6, qw_col6 );
                    const float32x4_t qw_mc7  = vmulq_f32( qw_mat7, qw_col7 );
                    const float32x4_t qw_mc8  = vmulq_f32( qw_mat8, qw_col8 );

                    qw_lanewise_sum1 = vaddq_f32( qw_mc1, qw_lanewise_sum1 );
                    qw_lanewise_sum2 = vaddq_f32( qw_mc2, qw_lanewise_sum2 );
                    qw_lanewise_sum3 = vaddq_f32( qw_mc3, qw_lanewise_sum3 );
                    qw_lanewise_sum4 = vaddq_f32( qw_mc4, qw_lanewise_sum4 );
                    qw_lanewise_sum5 = vaddq_f32( qw_mc5, qw_lanewise_sum5 );
                    qw_lanewise_sum6 = vaddq_f32( qw_mc6, qw_lanewise_sum6 );
                    qw_lanewise_sum7 = vaddq_f32( qw_mc7, qw_lanewise_sum7 );
                    qw_lanewise_sum8 = vaddq_f32( qw_mc8, qw_lanewise_sum8 );
                }
                this->m_output_vector[i] =   qw_lanewise_sum1[0] + qw_lanewise_sum1[1] + qw_lanewise_sum1[2] + qw_lanewise_sum1[3]
                                           + qw_lanewise_sum2[0] + qw_lanewise_sum2[1] + qw_lanewise_sum2[2] + qw_lanewise_sum2[3]
                                           + qw_lanewise_sum3[0] + qw_lanewise_sum3[1] + qw_lanewise_sum3[2] + qw_lanewise_sum3[3]
                                           + qw_lanewise_sum4[0] + qw_lanewise_sum4[1] + qw_lanewise_sum4[2] + qw_lanewise_sum4[3]
                                           + qw_lanewise_sum5[0] + qw_lanewise_sum5[1] + qw_lanewise_sum5[2] + qw_lanewise_sum5[3]
                                           + qw_lanewise_sum6[0] + qw_lanewise_sum6[1] + qw_lanewise_sum6[2] + qw_lanewise_sum6[3]
                                           + qw_lanewise_sum7[0] + qw_lanewise_sum7[1] + qw_lanewise_sum7[2] + qw_lanewise_sum7[3]
                                           + qw_lanewise_sum8[0] + qw_lanewise_sum8[1] + qw_lanewise_sum8[2] + qw_lanewise_sum8[3];
            }
        }
        else {
            for ( int i = row_begin; i < row_end_past_one; i++ ) {

                float64x2_t qw_lanewise_sum1 = { 0.0, 0.0 };
                float64x2_t qw_lanewise_sum2 = { 0.0, 0.0 };
                float64x2_t qw_lanewise_sum3 = { 0.0, 0.0 };
                float64x2_t qw_lanewise_sum4 = { 0.0, 0.0 };
                float64x2_t qw_lanewise_sum5 = { 0.0, 0.0 };
                float64x2_t qw_lanewise_sum6 = { 0.0, 0.0 };
                float64x2_t qw_lanewise_sum7 = { 0.0, 0.0 };
                float64x2_t qw_lanewise_sum8 = { 0.0, 0.0 };

                for ( int j = 0; j < this->m_N; j+=16 ) {

                    const int mat_index1 = linear_index_mat<IS_COL_MAJOR>( i, j, this->m_M, this->m_N );
                    const int mat_index2 = mat_index1 +  2;
                    const int mat_index3 = mat_index1 +  4;
                    const int mat_index4 = mat_index1 +  6;
                    const int mat_index5 = mat_index1 +  8;
                    const int mat_index6 = mat_index1 + 10;
                    const int mat_index7 = mat_index1 + 12;
                    const int mat_index8 = mat_index1 + 14;
                    const float64x2_t qw_mat1 = vld1q_f64( &(this->m_matrix[ mat_index1 ] ) );
                    const float64x2_t qw_mat2 = vld1q_f64( &(this->m_matrix[ mat_index2 ] ) );
                    const float64x2_t qw_mat3 = vld1q_f64( &(this->m_matrix[ mat_index3 ] ) );
                    const float64x2_t qw_mat4 = vld1q_f64( &(this->m_matrix[ mat_index4 ] ) );
                    const float64x2_t qw_mat5 = vld1q_f64( &(this->m_matrix[ mat_index5 ] ) );
                    const float64x2_t qw_mat6 = vld1q_f64( &(this->m_matrix[ mat_index6 ] ) );
                    const float64x2_t qw_mat7 = vld1q_f64( &(this->m_matrix[ mat_index7 ] ) );
                    const float64x2_t qw_mat8 = vld1q_f64( &(this->m_matrix[ mat_index8 ] ) );
                    const float64x2_t qw_col1 = vld1q_f64( &(this->m_vector[ j      ]     ) );
                    const float64x2_t qw_col2 = vld1q_f64( &(this->m_vector[ j +  2 ]     ) );
                    const float64x2_t qw_col3 = vld1q_f64( &(this->m_vector[ j +  4 ]     ) );
                    const float64x2_t qw_col4 = vld1q_f64( &(this->m_vector[ j +  6 ]     ) );
                    const float64x2_t qw_col5 = vld1q_f64( &(this->m_vector[ j +  8 ]     ) );
                    const float64x2_t qw_col6 = vld1q_f64( &(this->m_vector[ j + 10 ]     ) );
                    const float64x2_t qw_col7 = vld1q_f64( &(this->m_vector[ j + 12 ]     ) );
                    const float64x2_t qw_col8 = vld1q_f64( &(this->m_vector[ j + 14 ]     ) );
                    const float64x2_t qw_mc1  = vmulq_f64( qw_mat1, qw_col1 );
                    const float64x2_t qw_mc2  = vmulq_f64( qw_mat2, qw_col2 );
                    const float64x2_t qw_mc3  = vmulq_f64( qw_mat3, qw_col3 );
                    const float64x2_t qw_mc4  = vmulq_f64( qw_mat4, qw_col4 );
                    const float64x2_t qw_mc5  = vmulq_f64( qw_mat5, qw_col5 );
                    const float64x2_t qw_mc6  = vmulq_f64( qw_mat6, qw_col6 );
                    const float64x2_t qw_mc7  = vmulq_f64( qw_mat7, qw_col7 );
                    const float64x2_t qw_mc8  = vmulq_f64( qw_mat8, qw_col8 );

                    qw_lanewise_sum1 = vaddq_f64( qw_mc1, qw_lanewise_sum1 );
                    qw_lanewise_sum2 = vaddq_f64( qw_mc2, qw_lanewise_sum2 );
                    qw_lanewise_sum3 = vaddq_f64( qw_mc3, qw_lanewise_sum3 );
                    qw_lanewise_sum4 = vaddq_f64( qw_mc4, qw_lanewise_sum4 );
                    qw_lanewise_sum5 = vaddq_f64( qw_mc5, qw_lanewise_sum5 );
                    qw_lanewise_sum6 = vaddq_f64( qw_mc6, qw_lanewise_sum6 );
                    qw_lanewise_sum7 = vaddq_f64( qw_mc7, qw_lanewise_sum7 );
                    qw_lanewise_sum8 = vaddq_f64( qw_mc8, qw_lanewise_sum8 );
                }
                this->m_output_vector[i] =   qw_lanewise_sum1[0] + qw_lanewise_sum1[1] + qw_lanewise_sum2[0] + qw_lanewise_sum2[1]
                                           + qw_lanewise_sum3[0] + qw_lanewise_sum3[1] + qw_lanewise_sum4[0] + qw_lanewise_sum4[1]
                                           + qw_lanewise_sum5[0] + qw_lanewise_sum5[1] + qw_lanewise_sum6[0] + qw_lanewise_sum6[1]
                                           + qw_lanewise_sum7[0] + qw_lanewise_sum7[1] + qw_lanewise_sum8[0] + qw_lanewise_sum8[1];
            }
        }
    }
};



template <class T, bool IS_COL_MAJOR>
class TestExecutorDenseMV : public TestExecutor {

  protected:

    const int             m_M;
    const int             m_N;
    default_random_engine m_e;
    T*                    m_matrix;
    T*                    m_vector;
    T*                    m_output_vector;
    T*                    m_output_vector_baseline;

  public:
    TestExecutorDenseMV(
        TestResults& results,
        const int    M,
        const int    N,
        const int    num_trials,
        const bool   repeatable,
        const T      low,
        const T      high
    )
        :TestExecutor             { results, M * N, num_trials }
        ,m_M                      { M }
        ,m_N                      { N }
        ,m_e                      { static_cast<unsigned int>( repeatable ? 0 : chrono::system_clock::now().time_since_epoch().count() ) }
        ,m_matrix                 { new T [ M * N ] }
        ,m_vector                 { new T [ N ] }
        ,m_output_vector          { new T [ M ] }
        ,m_output_vector_baseline { new T [ M ] }
    {
        memset( m_output_vector,          0, sizeof(T)*m_M );
        memset( m_output_vector_baseline, 0, sizeof(T)*m_M );

        generateDenseMatrixVector( m_M, m_N, m_matrix, m_vector, low, high, m_e );
    }

    virtual ~TestExecutorDenseMV()
    {
        delete[] m_matrix;
        delete[] m_vector;
        delete[] m_output_vector;
        delete[] m_output_vector_baseline;
    }

    void prepareForRun ( const int test_case, const int num )
    {
        memset( m_output_vector, 0,  sizeof(T) * m_M );

        auto t = dynamic_pointer_cast< TestCaseDenseMV<T,IS_COL_MAJOR> >( this->m_test_cases[ test_case ] );
        t->setInitialStates( m_matrix, m_vector, m_output_vector );
    }

    void cleanupAfterBatchRuns ( const int test_case )
    {
        auto t = dynamic_pointer_cast< TestCaseDenseMV<T,IS_COL_MAJOR> >( this->m_test_cases[ test_case ] );

        if ( test_case == 0 ) {
            memcpy( m_output_vector_baseline, m_output_vector, sizeof(T) * m_M );
        }
        t->compareTruth( m_output_vector_baseline );
    }
};


static const size_t NUM_TRIALS = 10;

struct matrix_dim {
    size_t M;
    size_t N;
};

struct matrix_dim matrix_dims[]= {
    {    256,    256 },
    {    512,    512 },
    {   1024,   1024 },
    {   2048,   2048 },
    {   4096,   4096 }
};

template<class T, bool IS_COL_MAJOR>
string testSuitePerType ( const bool print_diag, const T gen_low, const T gen_high )
{
    vector< string > case_names {
        "plain c++",
        "NEON loop unrolled order 1",
        "NEON loop unrolled order 2",
        "NEON loop unrolled order 4",
        "NEON loop unrolled order 8",
        "BLAS Netlib's CLAPACK reference",
    };

    vector< string > header_line {
        "matrix size",
        "256x256",
        "512x512",
        "1Kx1K",
        "2Kx2K",
        "4Kx4K"
    };

    TestResults results{ case_names, header_line };

    const int neon_num_lanes = ( is_same<float, T>::value )? 4 : 2;

    for( auto& dims : matrix_dims ) {

        const auto M = dims.M;
        const auto N = dims.N;

        const auto span_neon = IS_COL_MAJOR ? M : N;

        TestExecutorDenseMV<T, IS_COL_MAJOR> e( results, M, N, NUM_TRIALS, false, gen_low, gen_high );

        e.addTestCase( make_shared< TestCaseDenseMV_baseline    <T, IS_COL_MAJOR> > ( case_names[0], M, N ) );

        e.addTestCase( make_shared< TestCaseDenseMV_NEON <T, IS_COL_MAJOR> > ( case_names[1], M, N,  1 ) );

        if ( span_neon >= 2 * neon_num_lanes ) {
            e.addTestCase( make_shared< TestCaseDenseMV_NEON <T, IS_COL_MAJOR> > ( case_names[2], M, N,  2 ) );
        }

        if ( span_neon >= 4 * neon_num_lanes ) {
            e.addTestCase( make_shared< TestCaseDenseMV_NEON <T, IS_COL_MAJOR> > ( case_names[3], M, N,  4 ) );
        }

        if ( span_neon >= 8 * neon_num_lanes ) {
            e.addTestCase( make_shared< TestCaseDenseMV_NEON <T, IS_COL_MAJOR> > ( case_names[4], M, N,  8 ) );
        }

        e.addTestCase( make_shared< TestCaseDenseMV_blas <T, IS_COL_MAJOR> > ( case_names[5], M, N ) );

        e.execute( print_diag );
    }

    std::stringstream ss;

    results.printHTML( ss );

    return ss.str();
}

#ifdef __EMSCRIPTEN__

string testDenseMatrixVectorFloatColMajor()
{
    return testSuitePerType<float, true > ( true, -1.0, 1.0 );
}

string testDenseMatrixVectorFloatRowMajor()
{
    return testSuitePerType<float, false > ( true, -1.0, 1.0 );
}

string testDenseMatrixVectorDoubleColMajor()
{
    return testSuitePerType<double, true > ( true, -1.0, 1.0 );
}

string testDenseMatrixVectorDoubleRowMajor()
{
    return testSuitePerType<double, false > ( true, -1.0, 1.0 );
}

EMSCRIPTEN_BINDINGS( saxpy_module ) {
    emscripten::function( "testDenseMatrixVectorFloatColMajor",  &testDenseMatrixVectorFloatColMajor  ); 
    emscripten::function( "testDenseMatrixVectorFloatRowMajor",  &testDenseMatrixVectorFloatRowMajor  ); 
    emscripten::function( "testDenseMatrixVectorDoubleColMajor", &testDenseMatrixVectorDoubleColMajor ); 
    emscripten::function( "testDenseMatrixVectorDoubleRowMajor", &testDenseMatrixVectorDoubleRowMajor ); 
}

#else

int main( int argc, char* argv[] )
{
    const bool print_diag = (argc == 2);

    cout << "dense mul mat vec (float, col-major)\n\n";
    cout << testSuitePerType<float, true > ( print_diag, -1.0, 1.0 );
    cout << "\n\n";

    cout << "dense mul mat vec (float, row-major)\n\n";
    cout << testSuitePerType<float, false > ( print_diag, -1.0, 1.0 );
    cout << "\n\n";

    cout << "dense mul mat vec (double, col-major)\n\n";
    cout << testSuitePerType<double, true > ( print_diag, -1.0, 1.0 );
    cout << "\n\n";

    cout << "dense mul mat vec (double, row-major)\n\n";
    cout << testSuitePerType<double, false > ( print_diag, -1.0, 1.0 );
    cout << "\n\n";

    return 0;
}

#endif
