#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <emscripten/bind.h>
#endif

#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <thread>
#include <arm_neon.h>
#include "test_case_with_time_measurements.h"
#include "test_pattern_generation.h"
#include "netlib_clapack_reference.h"

template<class T>
class TestCaseSAXPY :public TestCaseWithTimeMeasurements {

static constexpr double RMS_PASSING_THRESHOLD = 1.0e-6;

protected:
    const size_t m_num_elements;
    const T*     m_x;
    T*           m_y;
    T            m_alpha;

public:
    TestCaseSAXPY( const string& case_name, const size_t num_elements )
        :TestCaseWithTimeMeasurements { case_name }
        ,m_num_elements{ num_elements }
        ,m_x           { nullptr      }
        ,m_y           { nullptr      }
        ,m_alpha       { 0.0          }
    {
        ;
    }

    virtual ~TestCaseSAXPY()
    {
        ;
    }

    virtual bool needsToCopyBackY() = 0;

    void calculateRMS( const T* baseline )
    {
        auto rms = getRMSDiffTwoVectors( baseline, getY(), m_num_elements );
        setRMS( rms );
         
        setTrueFalse( (rms < RMS_PASSING_THRESHOLD) ? true : false );
    }

    virtual void setX    ( const T* const x ){ m_x = x; }
    virtual void setY    ( T* const y       ){ m_y = y; }
    virtual void setAlpha( const T a        ){ m_alpha = a; }
    virtual T*   getY    ()                  { return m_y; }

    virtual void run() = 0;
};


template<class T>
class TestCaseSAXPY_baseline : public TestCaseSAXPY<T> {

  public:
    TestCaseSAXPY_baseline( const string& case_name, const size_t num_elements )
        :TestCaseSAXPY<T>{ case_name, num_elements }
    {
        ;
    }

    virtual ~TestCaseSAXPY_baseline()
    {
        ;
    }

    virtual bool needsToCopyBackY() { return false; }

    void run() {

        for ( size_t i = 0; i < this->m_num_elements ; i++ ) {

            this->m_y[ i ] = this->m_alpha * this->m_x[ i ] + this->m_y[ i ];
        }
    }
};

template<class T>
class TestCaseSAXPY_neon : public TestCaseSAXPY<T> {

  protected:
    const size_t m_factor_loop_unrolling;

    void ( TestCaseSAXPY_neon<T>::*m_calc_block )( const int, const int );

    void calc_block_factor_1( const int elem_begin, const int elem_end_past_one )
    {
        if constexpr ( is_same<float, T>::value ) {

            for ( size_t i = elem_begin;  i < elem_end_past_one; i += 4 ) {

                //__builtin_prefetch(&this->m_x[i+4], 0 );
                //__builtin_prefetch(&this->m_y[i+4], 1 );

                const float32x4_t x_quad    = vld1q_f32( &this->m_x[i] );
                const float32x4_t ax_quad   = vmulq_n_f32( x_quad, this->m_alpha );
                const float32x4_t y_quad    = vld1q_f32( &this->m_y[i] );
                const float32x4_t axpy_quad = vaddq_f32( ax_quad, y_quad );
                vst1q_f32( &(this->m_y[i]), axpy_quad );
            }
        }      
        else {
            for ( size_t i = elem_begin;  i < elem_end_past_one; i += 2 ) {

                //__builtin_prefetch(&this->m_x[i+2], 0 );
                //__builtin_prefetch(&this->m_y[i+2], 1 );

                const float64x2_t x_pair    = vld1q_f64( &this->m_x[i] );
                const float64x2_t ax_pair   = vmulq_n_f64( x_pair, this->m_alpha );
                const float64x2_t y_pair    = vld1q_f64( &this->m_y[i] );
                const float64x2_t axpy_pair = vaddq_f64( ax_pair, y_pair );
                vst1q_f64( &(this->m_y[i]), axpy_pair );
            }
        }
    }

    void calc_block_factor_2(const int elem_begin, const int elem_end_past_one)
    {
        if constexpr ( is_same<float, T>::value ) {

            for ( size_t i = elem_begin;  i < elem_end_past_one; i += 8 ) {

                //__builtin_prefetch(&this->m_x[i+8], 0 );
                //__builtin_prefetch(&this->m_y[i+8], 1 );

                const float32x4_t x_quad1    = vld1q_f32( &this->m_x[i] );
                const float32x4_t x_quad2    = vld1q_f32( &this->m_x[i+4] );
                const float32x4_t y_quad1    = vld1q_f32( &this->m_y[i] );
                const float32x4_t y_quad2    = vld1q_f32( &this->m_y[i+4] );
                const float32x4_t ax_quad1   = vmulq_n_f32( x_quad1, this->m_alpha );
                const float32x4_t ax_quad2   = vmulq_n_f32( x_quad2, this->m_alpha );
                const float32x4_t axpy_quad1 = vaddq_f32( ax_quad1, y_quad1 );
                const float32x4_t axpy_quad2 = vaddq_f32( ax_quad2, y_quad2 );
                vst1q_f32( &(this->m_y[i]),    axpy_quad1 );
                vst1q_f32( &(this->m_y[i+4]),  axpy_quad2 );
            }
        }      
        else {
            for ( size_t i = elem_begin;  i < elem_end_past_one; i += 4 ) {

                //__builtin_prefetch(&this->m_x[i+4], 0 );
                //__builtin_prefetch(&this->m_y[i+4], 1 );

                const float64x2_t x_pair1    = vld1q_f64( &this->m_x[i  ] );
                const float64x2_t x_pair2    = vld1q_f64( &this->m_x[i+2] );
                const float64x2_t y_pair1    = vld1q_f64( &this->m_y[i  ] );
                const float64x2_t y_pair2    = vld1q_f64( &this->m_y[i+2] );
                const float64x2_t ax_pair1   = vmulq_n_f64( x_pair1, this->m_alpha );
                const float64x2_t ax_pair2   = vmulq_n_f64( x_pair2, this->m_alpha );
                const float64x2_t axpy_pair1 = vaddq_f64( ax_pair1, y_pair1 );
                const float64x2_t axpy_pair2 = vaddq_f64( ax_pair2, y_pair2 );
                vst1q_f64( &(this->m_y[i  ]), axpy_pair1 );
                vst1q_f64( &(this->m_y[i+2]), axpy_pair2 );
            }
        }
    }

    void calc_block_factor_4(const int elem_begin, const int elem_end_past_one)
    {
        if constexpr ( is_same<float, T>::value ) {

            for ( size_t i = elem_begin;  i < elem_end_past_one; i += 16 ) {

                //__builtin_prefetch(&this->m_x[i+16], 0 );
                //__builtin_prefetch(&this->m_y[i+16], 1 );

                const float32x4_t x_quad1    = vld1q_f32( &this->m_x[i   ] );
                const float32x4_t x_quad2    = vld1q_f32( &this->m_x[i+ 4] );
                const float32x4_t x_quad3    = vld1q_f32( &this->m_x[i+ 8] );
                const float32x4_t x_quad4    = vld1q_f32( &this->m_x[i+12] );
                const float32x4_t y_quad1    = vld1q_f32( &this->m_y[i   ] );
                const float32x4_t y_quad2    = vld1q_f32( &this->m_y[i+ 4] );
                const float32x4_t y_quad3    = vld1q_f32( &this->m_y[i+ 8] );
                const float32x4_t y_quad4    = vld1q_f32( &this->m_y[i+12] );
                const float32x4_t ax_quad1   = vmulq_n_f32( x_quad1, this->m_alpha );
                const float32x4_t ax_quad2   = vmulq_n_f32( x_quad2, this->m_alpha );
                const float32x4_t ax_quad3   = vmulq_n_f32( x_quad3, this->m_alpha );
                const float32x4_t ax_quad4   = vmulq_n_f32( x_quad4, this->m_alpha );
                const float32x4_t axpy_quad1 = vaddq_f32( ax_quad1, y_quad1 );
                const float32x4_t axpy_quad2 = vaddq_f32( ax_quad2, y_quad2 );
                const float32x4_t axpy_quad3 = vaddq_f32( ax_quad3, y_quad3 );
                const float32x4_t axpy_quad4 = vaddq_f32( ax_quad4, y_quad4 );
                vst1q_f32( &(this->m_y[i   ]), axpy_quad1 );
                vst1q_f32( &(this->m_y[i+ 4]), axpy_quad2 );
                vst1q_f32( &(this->m_y[i+ 8]), axpy_quad3 );
                vst1q_f32( &(this->m_y[i+12]), axpy_quad4 );
            }
        }      
        else {
            for (size_t i = elem_begin;  i < elem_end_past_one; i += 8 ) {

                //__builtin_prefetch(&this->m_x[i+8], 0 );
                //__builtin_prefetch(&this->m_y[i+8], 1 );

                const float64x2_t x_pair1    = vld1q_f64( &this->m_x[i  ] );
                const float64x2_t x_pair2    = vld1q_f64( &this->m_x[i+2] );
                const float64x2_t x_pair3    = vld1q_f64( &this->m_x[i+4] );
                const float64x2_t x_pair4    = vld1q_f64( &this->m_x[i+6] );
                const float64x2_t y_pair1    = vld1q_f64( &this->m_y[i  ] );
                const float64x2_t y_pair2    = vld1q_f64( &this->m_y[i+2] );
                const float64x2_t y_pair3    = vld1q_f64( &this->m_y[i+4] );
                const float64x2_t y_pair4    = vld1q_f64( &this->m_y[i+6] );
                const float64x2_t ax_pair1   = vmulq_n_f64( x_pair1, this->m_alpha );
                const float64x2_t ax_pair2   = vmulq_n_f64( x_pair2, this->m_alpha );
                const float64x2_t ax_pair3   = vmulq_n_f64( x_pair3, this->m_alpha );
                const float64x2_t ax_pair4   = vmulq_n_f64( x_pair4, this->m_alpha );
                const float64x2_t axpy_pair1 = vaddq_f64( ax_pair1, y_pair1 );
                const float64x2_t axpy_pair2 = vaddq_f64( ax_pair2, y_pair2 );
                const float64x2_t axpy_pair3 = vaddq_f64( ax_pair3, y_pair3 );
                const float64x2_t axpy_pair4 = vaddq_f64( ax_pair4, y_pair4 );
                vst1q_f64( &(this->m_y[i  ]), axpy_pair1 );
                vst1q_f64( &(this->m_y[i+2]), axpy_pair2 );
                vst1q_f64( &(this->m_y[i+4]), axpy_pair3 );
                vst1q_f64( &(this->m_y[i+6]), axpy_pair4 );
            }
        }
    }

    void calc_block_factor_8(const int elem_begin, const int elem_end_past_one)
    {
        if constexpr ( is_same<float, T>::value ) {

            for ( size_t i = elem_begin;  i < elem_end_past_one; i += 32 ) {

                //__builtin_prefetch(&this->m_x[i+32], 0 );
                //__builtin_prefetch(&this->m_y[i+32], 1 );

                const float32x4_t x_quad1    = vld1q_f32( &this->m_x[i   ] );
                const float32x4_t x_quad2    = vld1q_f32( &this->m_x[i+ 4] );
                const float32x4_t x_quad3    = vld1q_f32( &this->m_x[i+ 8] );
                const float32x4_t x_quad4    = vld1q_f32( &this->m_x[i+12] );
                const float32x4_t x_quad5    = vld1q_f32( &this->m_x[i+16] );
                const float32x4_t x_quad6    = vld1q_f32( &this->m_x[i+20] );
                const float32x4_t x_quad7    = vld1q_f32( &this->m_x[i+24] );
                const float32x4_t x_quad8    = vld1q_f32( &this->m_x[i+28] );
                const float32x4_t y_quad1    = vld1q_f32( &this->m_y[i   ] );
                const float32x4_t y_quad2    = vld1q_f32( &this->m_y[i+ 4] );
                const float32x4_t y_quad3    = vld1q_f32( &this->m_y[i+ 8] );
                const float32x4_t y_quad4    = vld1q_f32( &this->m_y[i+12] );
                const float32x4_t y_quad5    = vld1q_f32( &this->m_y[i+16] );
                const float32x4_t y_quad6    = vld1q_f32( &this->m_y[i+20] );
                const float32x4_t y_quad7    = vld1q_f32( &this->m_y[i+24] );
                const float32x4_t y_quad8    = vld1q_f32( &this->m_y[i+28] );
                const float32x4_t ax_quad1   = vmulq_n_f32( x_quad1, this->m_alpha );
                const float32x4_t ax_quad2   = vmulq_n_f32( x_quad2, this->m_alpha );
                const float32x4_t ax_quad3   = vmulq_n_f32( x_quad3, this->m_alpha );
                const float32x4_t ax_quad4   = vmulq_n_f32( x_quad4, this->m_alpha );
                const float32x4_t ax_quad5   = vmulq_n_f32( x_quad5, this->m_alpha );
                const float32x4_t ax_quad6   = vmulq_n_f32( x_quad6, this->m_alpha );
                const float32x4_t ax_quad7   = vmulq_n_f32( x_quad7, this->m_alpha );
                const float32x4_t ax_quad8   = vmulq_n_f32( x_quad8, this->m_alpha );
                const float32x4_t axpy_quad1 = vaddq_f32( ax_quad1, y_quad1 );
                const float32x4_t axpy_quad2 = vaddq_f32( ax_quad2, y_quad2 );
                const float32x4_t axpy_quad3 = vaddq_f32( ax_quad3, y_quad3 );
                const float32x4_t axpy_quad4 = vaddq_f32( ax_quad4, y_quad4 );
                const float32x4_t axpy_quad5 = vaddq_f32( ax_quad5, y_quad5 );
                const float32x4_t axpy_quad6 = vaddq_f32( ax_quad6, y_quad6 );
                const float32x4_t axpy_quad7 = vaddq_f32( ax_quad7, y_quad7 );
                const float32x4_t axpy_quad8 = vaddq_f32( ax_quad8, y_quad8 );
                vst1q_f32( &(this->m_y[i   ]), axpy_quad1 );
                vst1q_f32( &(this->m_y[i+ 4]), axpy_quad2 );
                vst1q_f32( &(this->m_y[i+ 8]), axpy_quad3 );
                vst1q_f32( &(this->m_y[i+12]), axpy_quad4 );
                vst1q_f32( &(this->m_y[i+16]), axpy_quad5 );
                vst1q_f32( &(this->m_y[i+20]), axpy_quad6 );
                vst1q_f32( &(this->m_y[i+24]), axpy_quad7 );
                vst1q_f32( &(this->m_y[i+28]), axpy_quad8 );
            }
        }      
        else {
            for ( size_t i = elem_begin;  i < elem_end_past_one; i += 16 ) {

                //__builtin_prefetch(&this->m_x[i+16], 0 );
                //__builtin_prefetch(&this->m_y[i+16], 1 );

                const float64x2_t x_pair1    = vld1q_f64( &this->m_x[i   ] );
                const float64x2_t x_pair2    = vld1q_f64( &this->m_x[i+ 2] );
                const float64x2_t x_pair3    = vld1q_f64( &this->m_x[i+ 4] );
                const float64x2_t x_pair4    = vld1q_f64( &this->m_x[i+ 6] );
                const float64x2_t x_pair5    = vld1q_f64( &this->m_x[i+ 8] );
                const float64x2_t x_pair6    = vld1q_f64( &this->m_x[i+10] );
                const float64x2_t x_pair7    = vld1q_f64( &this->m_x[i+12] );
                const float64x2_t x_pair8    = vld1q_f64( &this->m_x[i+14] );
                const float64x2_t y_pair1    = vld1q_f64( &this->m_y[i   ] );
                const float64x2_t y_pair2    = vld1q_f64( &this->m_y[i+ 2] );
                const float64x2_t y_pair3    = vld1q_f64( &this->m_y[i+ 4] );
                const float64x2_t y_pair4    = vld1q_f64( &this->m_y[i+ 6] );
                const float64x2_t y_pair5    = vld1q_f64( &this->m_y[i+ 8] );
                const float64x2_t y_pair6    = vld1q_f64( &this->m_y[i+10] );
                const float64x2_t y_pair7    = vld1q_f64( &this->m_y[i+12] );
                const float64x2_t y_pair8    = vld1q_f64( &this->m_y[i+14] );
                const float64x2_t ax_pair1   = vmulq_n_f64( x_pair1, this->m_alpha );
                const float64x2_t ax_pair2   = vmulq_n_f64( x_pair2, this->m_alpha );
                const float64x2_t ax_pair3   = vmulq_n_f64( x_pair3, this->m_alpha );
                const float64x2_t ax_pair4   = vmulq_n_f64( x_pair4, this->m_alpha );
                const float64x2_t ax_pair5   = vmulq_n_f64( x_pair5, this->m_alpha );
                const float64x2_t ax_pair6   = vmulq_n_f64( x_pair6, this->m_alpha );
                const float64x2_t ax_pair7   = vmulq_n_f64( x_pair7, this->m_alpha );
                const float64x2_t ax_pair8   = vmulq_n_f64( x_pair8, this->m_alpha );
                const float64x2_t axpy_pair1 = vaddq_f64( ax_pair1, y_pair1 );
                const float64x2_t axpy_pair2 = vaddq_f64( ax_pair2, y_pair2 );
                const float64x2_t axpy_pair3 = vaddq_f64( ax_pair3, y_pair3 );
                const float64x2_t axpy_pair4 = vaddq_f64( ax_pair4, y_pair4 );
                const float64x2_t axpy_pair5 = vaddq_f64( ax_pair5, y_pair5 );
                const float64x2_t axpy_pair6 = vaddq_f64( ax_pair6, y_pair6 );
                const float64x2_t axpy_pair7 = vaddq_f64( ax_pair7, y_pair7 );
                const float64x2_t axpy_pair8 = vaddq_f64( ax_pair8, y_pair8 );
                vst1q_f64( &(this->m_y[i   ]), axpy_pair1 );
                vst1q_f64( &(this->m_y[i+ 2]), axpy_pair2 );
                vst1q_f64( &(this->m_y[i+ 4]), axpy_pair3 );
                vst1q_f64( &(this->m_y[i+ 6]), axpy_pair4 );
                vst1q_f64( &(this->m_y[i+ 8]), axpy_pair5 );
                vst1q_f64( &(this->m_y[i+10]), axpy_pair6 );
                vst1q_f64( &(this->m_y[i+12]), axpy_pair7 );
                vst1q_f64( &(this->m_y[i+14]), axpy_pair8 );
            }
        }
    }


  public:
    TestCaseSAXPY_neon( const string& case_name, const size_t num_elements, const size_t factor_loop_unrolling )
        :TestCaseSAXPY<T>       { case_name, num_elements }
        ,m_factor_loop_unrolling{ factor_loop_unrolling }
    {
        if ( factor_loop_unrolling == 1 ) {

            m_calc_block = &TestCaseSAXPY_neon::calc_block_factor_1;
        }
        else if ( factor_loop_unrolling == 2 ) {

            m_calc_block = &TestCaseSAXPY_neon::calc_block_factor_2;
        }
        else if ( factor_loop_unrolling == 4 ) {

            m_calc_block = &TestCaseSAXPY_neon::calc_block_factor_4;
        }
        else if ( factor_loop_unrolling == 8 ) {

            m_calc_block = &TestCaseSAXPY_neon::calc_block_factor_8;
        }
        else {
            assert( true );
        }
    }

    virtual ~TestCaseSAXPY_neon()
    {
        ;
    }

    virtual bool needsToCopyBackY()
    {
        return false;
    }

    virtual inline void call_block( const int elem_begin, const int elem_end_past_one ) {

        (this->*m_calc_block)( elem_begin, elem_end_past_one );
    }

    void run() {
        call_block( 0, this->m_num_elements );
    }
};


template<class T>
class TestCaseSAXPY_BLAS : public TestCaseSAXPY<T> {

  public:
    TestCaseSAXPY_BLAS( const string& case_name, const size_t num_elements )
        :TestCaseSAXPY<T>{ case_name, num_elements }
    {
        static_assert( is_same<float, T>::value || is_same<double, T>::value );
    }

    virtual ~TestCaseSAXPY_BLAS()
    {
        ;
    }

    virtual bool needsToCopyBackY() { return false; }

    void run();
};

template<>
void TestCaseSAXPY_BLAS<float>::run()
{
    integer one{ 1 };
    integer n  { static_cast<integer>(this->m_num_elements) };
    
    saxpy_( &n, &(this->m_alpha), const_cast<real*>( this->m_x) , &one, this->m_y, &one );
}

template<>
void TestCaseSAXPY_BLAS<double>::run()
{
    integer one{ 1 };
    integer n  { static_cast<integer>(this->m_num_elements) };

    daxpy_( &n, &(this->m_alpha), const_cast<doublereal*>( this->m_x) , &one, this->m_y, &one );
}

template <class T>
class TestExecutorSAXPY : public TestExecutor {

protected:

    const bool            m_repeatable;
    default_random_engine m_e;
    T*                    m_x;
    T*                    m_y_org;
    T                     m_alpha;
    T*                    m_y_out;
    T*                    m_y_baseline;

public:

    TestExecutorSAXPY(
        TestResults& results,
        const int    num_elements,
        const int    num_trials,
        const bool   repeatable,
        const T      min_val,
        const T      max_val 
    )
        :TestExecutor  { results, num_elements, num_trials }
        ,m_repeatable  { repeatable }
        ,m_e           { static_cast<unsigned int>( repeatable ? 0 : chrono::system_clock::now().time_since_epoch().count() ) }
        ,m_x           { new T[ num_elements ] }
        ,m_y_org       { new T[ num_elements ] }
        ,m_alpha       { 0.0 }
        ,m_y_out       { new T[ num_elements ] }
        ,m_y_baseline  { new T[ num_elements ] }
    {
        fillArrayWithRandomValues( m_e, m_x,     m_num_elements, min_val, max_val );
        fillArrayWithRandomValues( m_e, m_y_org, m_num_elements, min_val, max_val );

        m_alpha = getRandomNum( m_e, min_val, max_val );

        memset( m_y_out,      0, m_num_elements * sizeof(T) );
        memset( m_y_baseline, 0, m_num_elements * sizeof(T) );
    }

    virtual ~TestExecutorSAXPY()
    {
        delete[] m_x;
        delete[] m_y_org;
        delete[] m_y_out;
        delete[] m_y_baseline;
    }

    void cleanupAfterBatchRuns ( const int test_case )
    {
        auto t = dynamic_pointer_cast< TestCaseSAXPY<T> >( this->m_test_cases[ test_case ] );

        if ( test_case == 0 ) {

            memcpy( m_y_baseline, t->getY(), sizeof(T) * m_num_elements );
        }

        t->calculateRMS( m_y_baseline );
    }

    void prepareForRun ( const int test_case, const int num )
    {
        memcpy( m_y_out, m_y_org, sizeof(T)*m_num_elements );
        auto t = dynamic_pointer_cast< TestCaseSAXPY<T> >( this->m_test_cases[ test_case ] );
        t->setX     ( m_x     );
        t->setY     ( m_y_out );
        t->setAlpha ( m_alpha );
    }
};

static const size_t NUM_TRIALS = 100;

size_t nums_elements[]{ 128, 512, 2*1024, 8*1024, 32*1024, 128*1024, 512*1024, 4*1024*1024 };

template<class T>
string testSuitePerType ( const bool print_diag )
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
        "vector length",
        "128",
        "512",
        "2K",
        "8K",
        "32K",
        "128K",
        "512K",
        "4M"
    };

    TestResults results{ case_names, header_line };

    const int neon_num_lanes = (is_same<float, T>::value)?4:2;

    for( auto num_elements : nums_elements ) {

        TestExecutorSAXPY<T> e( results, num_elements, NUM_TRIALS, false, -1.0, 1.0 );

        e.addTestCase( make_shared< TestCaseSAXPY_baseline<T> > ( case_names[ 0 ], num_elements    ) );
        e.addTestCase( make_shared< TestCaseSAXPY_neon    <T> > ( case_names[ 1 ], num_elements, 1 ) );
        e.addTestCase( make_shared< TestCaseSAXPY_neon    <T> > ( case_names[ 2 ], num_elements, 2 ) );
        e.addTestCase( make_shared< TestCaseSAXPY_neon    <T> > ( case_names[ 3 ], num_elements, 4 ) );
        if ( num_elements >= 8 * neon_num_lanes ) {
            e.addTestCase( make_shared< TestCaseSAXPY_neon<T> > ( case_names[ 4 ], num_elements, 8 ) );
        }

        e.addTestCase( make_shared< TestCaseSAXPY_BLAS    <T> > ( case_names[ 5 ], num_elements ) );

        e.execute( print_diag );
    }

    std::stringstream ss;

    results.printHTML( ss );

    return ss.str();
}


#ifdef __EMSCRIPTEN__

string testSaxpy()
{
    return testSuitePerType< float >( true );
}

string testDaxpy()
{
    return testSuitePerType< double >( true );
}

EMSCRIPTEN_BINDINGS( saxpy_module ) {
    emscripten::function( "testSaxpy", &testSaxpy );
    emscripten::function( "testDaxpy", &testDaxpy );
}

#else

int main( int argc, char* argv[] )
{
    const bool print_diag = (argc == 2);

    cout << "saxpy (float)\n\n";
    cout << testSuitePerType< float  > ( print_diag );
    cout << "\n\n";

    cout << "daxpy (double)\n\n";
    cout << testSuitePerType< double > ( print_diag );
    cout << "\n\n";
    return 0;
}

#endif
