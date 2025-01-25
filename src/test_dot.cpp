#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <emscripten/bind.h>
#endif

#include <iostream>
#include <sstream>
#include <iomanip>
#include <thread>
#include <vector>
#include <arm_neon.h>
#include "test_case_with_time_measurements.h"
#include "test_pattern_generation.h"
#include "netlib_clapack_reference.h"

namespace Dot {

template<class T>
class TestCaseDOT : public TestCaseWithTimeMeasurements {

static constexpr double DIST_PASSING_THRESHOLD = 1.0e-3;

protected:
    const size_t m_num_elements;
    const T*     m_x;
    const T*     m_y;
    T            m_dot;

public:
    TestCaseDOT( const string& case_name, const size_t num_elements )
        :TestCaseWithTimeMeasurements { case_name }
        ,m_num_elements               { num_elements }
    {
        ;
    }

    virtual ~TestCaseDOT()
    {
        ;
    }

    virtual void setX( const T* const x ){ m_x = x; }
    virtual void setY( const T* const y ){ m_y = y; }
    virtual T    getResult(){ return m_dot; }

    void calculateDistance( const T baseline )
    {
         double dist = fabs( ( baseline - getResult() ) / baseline );
         this->setDist( dist );

         this->setTrueFalse( (dist < DIST_PASSING_THRESHOLD) ? true : false );
    }

    virtual void run() = 0;
};


template<class T>
class TestCaseDOT_baseline : public TestCaseDOT<T> {

  public:
    TestCaseDOT_baseline( const string& case_name, const size_t num_elements )
        :TestCaseDOT<T>{ case_name, num_elements }
    {
        ;
    }

    virtual ~TestCaseDOT_baseline()
    {
        ;
    }

    virtual void run()
    {
        this->m_dot = 0.0;

        for ( size_t i = 0; i < this->m_num_elements ; i++ ) {

            this->m_dot += this->m_x[ i ] * this->m_y[ i ];
        }
    }
};


template<class T>
class TestCaseDOT_neon : public TestCaseDOT<T> {

  protected:

    const size_t m_factor_loop_unrolling;

    void ( TestCaseDOT_neon<T>::*m_calc_block )( T*, const int, const int );

    void calc_block_factor_1( T* sum, const int elem_begin, const int elem_end_past_one )
    {
        if constexpr( is_same<float, T>::value ) {

            float32x4_t dot_quad1{0.0, 0.0, 0.0, 0.0};

            for ( size_t i = elem_begin; i < elem_end_past_one ; i+=4 ) {

                const float32x4_t x_quad1  = vld1q_f32( &this->m_x[i]  );
                const float32x4_t y_quad1  = vld1q_f32( &this->m_y[i]  );
                const float32x4_t xy_quad1 = vmulq_f32( x_quad1, y_quad1 );
                dot_quad1 = vaddq_f32( dot_quad1, xy_quad1 );
            } 
            *sum = dot_quad1[0] + dot_quad1[1] + dot_quad1[2] + dot_quad1[3];
        }
        else {
            float64x2_t dot_pair1{0.0, 0.0};

            for ( size_t i = elem_begin; i < elem_end_past_one ; i+=2 ) {

                const float64x2_t x_pair1  = vld1q_f64( &this->m_x[i]  );
                const float64x2_t y_pair1  = vld1q_f64( &this->m_y[i]  );
                const float64x2_t xy_pair1 = vmulq_f64( x_pair1, y_pair1 );
                dot_pair1 = vaddq_f64( dot_pair1, xy_pair1 );
            } 
            *sum = dot_pair1[0] + dot_pair1[1];
        }
    }

    void calc_block_factor_2( T* sum, const int elem_begin, const int elem_end_past_one )
    {
        if constexpr( is_same<float, T>::value ) {

            float32x4_t dot_quad1{0.0, 0.0, 0.0, 0.0};
            float32x4_t dot_quad2{0.0, 0.0, 0.0, 0.0};

            for ( size_t i = elem_begin; i < elem_end_past_one ; i+=8 ) {

                const float32x4_t x_quad1  = vld1q_f32( &this->m_x[i]  );
                const float32x4_t x_quad2  = vld1q_f32( &this->m_x[i+4]  );
                const float32x4_t y_quad1  = vld1q_f32( &this->m_y[i]  );
                const float32x4_t y_quad2  = vld1q_f32( &this->m_y[i+4]  );
                const float32x4_t xy_quad1 = vmulq_f32( x_quad1, y_quad1 );
                const float32x4_t xy_quad2 = vmulq_f32( x_quad2, y_quad2 );
                dot_quad1 = vaddq_f32( dot_quad1, xy_quad1 );
                dot_quad2 = vaddq_f32( dot_quad2, xy_quad2 );
            } 
            *sum =   dot_quad1[0] + dot_quad1[1] + dot_quad1[2] + dot_quad1[3]
                   + dot_quad2[0] + dot_quad2[1] + dot_quad2[2] + dot_quad2[3];
        }
        else {
            float64x2_t dot_pair1{0.0, 0.0};
            float64x2_t dot_pair2{0.0, 0.0};

            for ( size_t i = elem_begin; i < elem_end_past_one; i+=4 ) {

                const float64x2_t x_pair1  = vld1q_f64( &this->m_x[i]  );
                const float64x2_t x_pair2  = vld1q_f64( &this->m_x[i+2]  );
                const float64x2_t y_pair1  = vld1q_f64( &this->m_y[i]  );
                const float64x2_t y_pair2  = vld1q_f64( &this->m_y[i+2]  );
                const float64x2_t xy_pair1 = vmulq_f64( x_pair1, y_pair1 );
                const float64x2_t xy_pair2 = vmulq_f64( x_pair2, y_pair2 );
                dot_pair1 = vaddq_f64( dot_pair1, xy_pair1 );
                dot_pair2 = vaddq_f64( dot_pair2, xy_pair2 );
            } 
            *sum = dot_pair1[0] + dot_pair1[1] +  dot_pair2[0] + dot_pair2[1];
        }
    }

    void calc_block_factor_4( T* sum, const int elem_begin, const int elem_end_past_one )
    {
        if constexpr( is_same<float, T>::value ) {

            float32x4_t dot_quad1{0.0, 0.0, 0.0, 0.0};
            float32x4_t dot_quad2{0.0, 0.0, 0.0, 0.0};
            float32x4_t dot_quad3{0.0, 0.0, 0.0, 0.0};
            float32x4_t dot_quad4{0.0, 0.0, 0.0, 0.0};

            for ( size_t i = elem_begin; i < elem_end_past_one; i+=16 ) {

                const float32x4_t x_quad1  = vld1q_f32( &this->m_x[i]  );
                const float32x4_t x_quad2  = vld1q_f32( &this->m_x[i+4]  );
                const float32x4_t x_quad3  = vld1q_f32( &this->m_x[i+8]  );
                const float32x4_t x_quad4  = vld1q_f32( &this->m_x[i+12]  );
                const float32x4_t y_quad1  = vld1q_f32( &this->m_y[i]  );
                const float32x4_t y_quad2  = vld1q_f32( &this->m_y[i+4]  );
                const float32x4_t y_quad3  = vld1q_f32( &this->m_y[i+8]  );
                const float32x4_t y_quad4  = vld1q_f32( &this->m_y[i+12]  );
                const float32x4_t xy_quad1 = vmulq_f32( x_quad1, y_quad1 );
                const float32x4_t xy_quad2 = vmulq_f32( x_quad2, y_quad2 );
                const float32x4_t xy_quad3 = vmulq_f32( x_quad3, y_quad3 );
                const float32x4_t xy_quad4 = vmulq_f32( x_quad4, y_quad4 );
                dot_quad1 = vaddq_f32( dot_quad1, xy_quad1 );
                dot_quad2 = vaddq_f32( dot_quad2, xy_quad2 );
                dot_quad3 = vaddq_f32( dot_quad3, xy_quad3 );
                dot_quad4 = vaddq_f32( dot_quad4, xy_quad4 );
            } 
            *sum =   dot_quad1[0] + dot_quad1[1] + dot_quad1[2] + dot_quad1[3]
                   + dot_quad2[0] + dot_quad2[1] + dot_quad2[2] + dot_quad2[3]
                   + dot_quad3[0] + dot_quad3[1] + dot_quad3[2] + dot_quad3[3]
                   + dot_quad4[0] + dot_quad4[1] + dot_quad4[2] + dot_quad4[3];
        }
        else {
            float64x2_t dot_pair1{0.0, 0.0};
            float64x2_t dot_pair2{0.0, 0.0};
            float64x2_t dot_pair3{0.0, 0.0};
            float64x2_t dot_pair4{0.0, 0.0};

            for ( size_t i = elem_begin; i < elem_end_past_one; i+=8 ) {

                const float64x2_t x_pair1  = vld1q_f64( &this->m_x[i]  );
                const float64x2_t x_pair2  = vld1q_f64( &this->m_x[i+2]  );
                const float64x2_t x_pair3  = vld1q_f64( &this->m_x[i+4]  );
                const float64x2_t x_pair4  = vld1q_f64( &this->m_x[i+6]  );
                const float64x2_t y_pair1  = vld1q_f64( &this->m_y[i]  );
                const float64x2_t y_pair2  = vld1q_f64( &this->m_y[i+2]  );
                const float64x2_t y_pair3  = vld1q_f64( &this->m_y[i+4]  );
                const float64x2_t y_pair4  = vld1q_f64( &this->m_y[i+6]  );
                const float64x2_t xy_pair1 = vmulq_f64( x_pair1, y_pair1 );
                const float64x2_t xy_pair2 = vmulq_f64( x_pair2, y_pair2 );
                const float64x2_t xy_pair3 = vmulq_f64( x_pair3, y_pair3 );
                const float64x2_t xy_pair4 = vmulq_f64( x_pair4, y_pair4 );
                dot_pair1 = vaddq_f64( dot_pair1, xy_pair1 );
                dot_pair2 = vaddq_f64( dot_pair2, xy_pair2 );
                dot_pair3 = vaddq_f64( dot_pair3, xy_pair3 );
                dot_pair4 = vaddq_f64( dot_pair4, xy_pair4 );
            } 
            *sum =   dot_pair1[0] + dot_pair1[1] +  dot_pair2[0] + dot_pair2[1]
                   + dot_pair3[0] + dot_pair3[1] +  dot_pair4[0] + dot_pair4[1];
        }
    }

    void calc_block_factor_8( T* sum, const int elem_begin, const int elem_end_past_one )
    {
        if constexpr( is_same<float, T>::value ) {

            float32x4_t dot_quad1{0.0, 0.0, 0.0, 0.0};
            float32x4_t dot_quad2{0.0, 0.0, 0.0, 0.0};
            float32x4_t dot_quad3{0.0, 0.0, 0.0, 0.0};
            float32x4_t dot_quad4{0.0, 0.0, 0.0, 0.0};
            float32x4_t dot_quad5{0.0, 0.0, 0.0, 0.0};
            float32x4_t dot_quad6{0.0, 0.0, 0.0, 0.0};
            float32x4_t dot_quad7{0.0, 0.0, 0.0, 0.0};
            float32x4_t dot_quad8{0.0, 0.0, 0.0, 0.0};

            for ( size_t i = elem_begin; i < elem_end_past_one; i+=32 ) {

                const float32x4_t x_quad1  = vld1q_f32( &this->m_x[i   ]  );
                const float32x4_t x_quad2  = vld1q_f32( &this->m_x[i+ 4]  );
                const float32x4_t x_quad3  = vld1q_f32( &this->m_x[i+ 8]  );
                const float32x4_t x_quad4  = vld1q_f32( &this->m_x[i+12]  );
                const float32x4_t x_quad5  = vld1q_f32( &this->m_x[i+16   ]  );
                const float32x4_t x_quad6  = vld1q_f32( &this->m_x[i+20]  );
                const float32x4_t x_quad7  = vld1q_f32( &this->m_x[i+24]  );
                const float32x4_t x_quad8  = vld1q_f32( &this->m_x[i+28]  );
                const float32x4_t y_quad1  = vld1q_f32( &this->m_y[i   ]  );
                const float32x4_t y_quad2  = vld1q_f32( &this->m_y[i+ 4]  );
                const float32x4_t y_quad3  = vld1q_f32( &this->m_y[i+ 8]  );
                const float32x4_t y_quad4  = vld1q_f32( &this->m_y[i+12]  );
                const float32x4_t y_quad5  = vld1q_f32( &this->m_y[i+16]  );
                const float32x4_t y_quad6  = vld1q_f32( &this->m_y[i+20]  );
                const float32x4_t y_quad7  = vld1q_f32( &this->m_y[i+24]  );
                const float32x4_t y_quad8  = vld1q_f32( &this->m_y[i+28]  );
                const float32x4_t xy_quad1 = vmulq_f32( x_quad1, y_quad1 );
                const float32x4_t xy_quad2 = vmulq_f32( x_quad2, y_quad2 );
                const float32x4_t xy_quad3 = vmulq_f32( x_quad3, y_quad3 );
                const float32x4_t xy_quad4 = vmulq_f32( x_quad4, y_quad4 );
                const float32x4_t xy_quad5 = vmulq_f32( x_quad5, y_quad5 );
                const float32x4_t xy_quad6 = vmulq_f32( x_quad6, y_quad6 );
                const float32x4_t xy_quad7 = vmulq_f32( x_quad7, y_quad7 );
                const float32x4_t xy_quad8 = vmulq_f32( x_quad8, y_quad8 );
                dot_quad1 = vaddq_f32( dot_quad1, xy_quad1 );
                dot_quad2 = vaddq_f32( dot_quad2, xy_quad2 );
                dot_quad3 = vaddq_f32( dot_quad3, xy_quad3 );
                dot_quad4 = vaddq_f32( dot_quad4, xy_quad4 );
                dot_quad5 = vaddq_f32( dot_quad5, xy_quad5 );
                dot_quad6 = vaddq_f32( dot_quad6, xy_quad6 );
                dot_quad7 = vaddq_f32( dot_quad7, xy_quad7 );
                dot_quad8 = vaddq_f32( dot_quad8, xy_quad8 );
            } 
            *sum =   dot_quad1[0] + dot_quad1[1] + dot_quad1[2] + dot_quad1[3]
                   + dot_quad2[0] + dot_quad2[1] + dot_quad2[2] + dot_quad2[3]
                   + dot_quad3[0] + dot_quad3[1] + dot_quad3[2] + dot_quad3[3]
                   + dot_quad4[0] + dot_quad4[1] + dot_quad4[2] + dot_quad4[3]
                   + dot_quad5[0] + dot_quad5[1] + dot_quad5[2] + dot_quad5[3]
                   + dot_quad6[0] + dot_quad6[1] + dot_quad6[2] + dot_quad6[3]
                   + dot_quad7[0] + dot_quad7[1] + dot_quad7[2] + dot_quad7[3]
                   + dot_quad8[0] + dot_quad8[1] + dot_quad8[2] + dot_quad8[3];
        }
        else {
            float64x2_t dot_pair1{0.0, 0.0};
            float64x2_t dot_pair2{0.0, 0.0};
            float64x2_t dot_pair3{0.0, 0.0};
            float64x2_t dot_pair4{0.0, 0.0};
            float64x2_t dot_pair5{0.0, 0.0};
            float64x2_t dot_pair6{0.0, 0.0};
            float64x2_t dot_pair7{0.0, 0.0};
            float64x2_t dot_pair8{0.0, 0.0};

            for ( size_t i = elem_begin; i < elem_end_past_one; i+=16 ) {

                const float64x2_t x_pair1  = vld1q_f64( &this->m_x[i   ]  );
                const float64x2_t x_pair2  = vld1q_f64( &this->m_x[i+ 2]  );
                const float64x2_t x_pair3  = vld1q_f64( &this->m_x[i+ 4]  );
                const float64x2_t x_pair4  = vld1q_f64( &this->m_x[i+ 6]  );
                const float64x2_t x_pair5  = vld1q_f64( &this->m_x[i+ 8]  );
                const float64x2_t x_pair6  = vld1q_f64( &this->m_x[i+10]  );
                const float64x2_t x_pair7  = vld1q_f64( &this->m_x[i+12]  );
                const float64x2_t x_pair8  = vld1q_f64( &this->m_x[i+14]  );
                const float64x2_t y_pair1  = vld1q_f64( &this->m_y[i   ]  );
                const float64x2_t y_pair2  = vld1q_f64( &this->m_y[i+ 2]  );
                const float64x2_t y_pair3  = vld1q_f64( &this->m_y[i+ 4]  );
                const float64x2_t y_pair4  = vld1q_f64( &this->m_y[i+ 6]  );
                const float64x2_t y_pair5  = vld1q_f64( &this->m_y[i+ 8]  );
                const float64x2_t y_pair6  = vld1q_f64( &this->m_y[i+10]  );
                const float64x2_t y_pair7  = vld1q_f64( &this->m_y[i+12]  );
                const float64x2_t y_pair8  = vld1q_f64( &this->m_y[i+14]  );
                const float64x2_t xy_pair1 = vmulq_f64( x_pair1, y_pair1 );
                const float64x2_t xy_pair2 = vmulq_f64( x_pair2, y_pair2 );
                const float64x2_t xy_pair3 = vmulq_f64( x_pair3, y_pair3 );
                const float64x2_t xy_pair4 = vmulq_f64( x_pair4, y_pair4 );
                const float64x2_t xy_pair5 = vmulq_f64( x_pair5, y_pair5 );
                const float64x2_t xy_pair6 = vmulq_f64( x_pair6, y_pair6 );
                const float64x2_t xy_pair7 = vmulq_f64( x_pair7, y_pair7 );
                const float64x2_t xy_pair8 = vmulq_f64( x_pair8, y_pair8 );
                dot_pair1 = vaddq_f64( dot_pair1, xy_pair1 );
                dot_pair2 = vaddq_f64( dot_pair2, xy_pair2 );
                dot_pair3 = vaddq_f64( dot_pair3, xy_pair3 );
                dot_pair4 = vaddq_f64( dot_pair4, xy_pair4 );
                dot_pair5 = vaddq_f64( dot_pair5, xy_pair5 );
                dot_pair6 = vaddq_f64( dot_pair6, xy_pair6 );
                dot_pair7 = vaddq_f64( dot_pair7, xy_pair7 );
                dot_pair8 = vaddq_f64( dot_pair8, xy_pair8 );
            } 
            *sum =   dot_pair1[0] + dot_pair1[1] +  dot_pair2[0] + dot_pair2[1]
                   + dot_pair3[0] + dot_pair3[1] +  dot_pair4[0] + dot_pair4[1]
                   + dot_pair5[0] + dot_pair5[1] +  dot_pair6[0] + dot_pair6[1]
                   + dot_pair7[0] + dot_pair7[1] +  dot_pair8[0] + dot_pair8[1];
        }
    }

  public:

    TestCaseDOT_neon( const string& case_name, const size_t num_elements, const size_t factor_loop_unrolling )
        :TestCaseDOT<T>         { case_name, num_elements }
        ,m_factor_loop_unrolling{ factor_loop_unrolling }
    {
        if (factor_loop_unrolling == 1) {

            m_calc_block = &TestCaseDOT_neon::calc_block_factor_1;
        }
        else if (factor_loop_unrolling == 2) {

            m_calc_block = &TestCaseDOT_neon::calc_block_factor_2;
        }
        else if (factor_loop_unrolling == 4) {

            m_calc_block = &TestCaseDOT_neon::calc_block_factor_4;
        }
        else if (factor_loop_unrolling == 8) {

            m_calc_block = &TestCaseDOT_neon::calc_block_factor_8;
        }
        else {
            assert(true);
        }
    }

    virtual ~TestCaseDOT_neon()
    {
        ;
    }

    virtual inline void calc_block( T* sum, const int elem_begin, const int elem_end_past_one )
    {
        (this->*m_calc_block)( sum, elem_begin, elem_end_past_one );
    }

    void run()
    {
        calc_block( &(this->m_dot), 0, this->m_num_elements );
    }
};

template<class T>
class TestCaseDOT_BLAS : public TestCaseDOT<T> {

  public:

    TestCaseDOT_BLAS( const string& case_name, const size_t num_elements )
        :TestCaseDOT<T>{ case_name, num_elements }
    {
        static_assert( is_same<float, T>::value || is_same<double, T>::value );
    }

    virtual ~TestCaseDOT_BLAS()
    {
        ;
    }

    void run();
};

template<>
void TestCaseDOT_BLAS<float>::run()
{
    integer one{ 1 };
    integer n  { static_cast<integer>( this->m_num_elements ) };

    this->m_dot = sdot_( &n, const_cast<real*>(this->m_x), &one, const_cast<real*>(this->m_y), &one );
}

template<>
void TestCaseDOT_BLAS<double>::run()
{
    integer one{ 1 };
    integer n  { static_cast<integer>( this->m_num_elements ) };

    this->m_dot = ddot_( &n, const_cast<doublereal*>(this->m_x), &one, const_cast<doublereal*>(this->m_y), &one );
}

template <class T>
class TestExecutorDOT : public TestExecutor {

  protected:

    const bool            m_repeatable;
    default_random_engine m_e;
    T*                    m_x;
    T*                    m_y;
    T                     m_dot_baseline;

  public:

    TestExecutorDOT(
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
        ,m_x           { new T[num_elements] }
        ,m_y           { new T[num_elements] }
        ,m_dot_baseline{ 0.0}
    {
        fillArrayWithRandomValues( m_e, m_x, m_num_elements, min_val, max_val );
        fillArrayWithRandomValues( m_e, m_y, m_num_elements, min_val, max_val );
    }

    virtual ~TestExecutorDOT()
    {
        delete[] m_x;
        delete[] m_y;
    }

    void cleanupAfterBatchRuns ( const int test_case )
    {
        auto t = dynamic_pointer_cast< TestCaseDOT<T> >( this->m_test_cases[ test_case ] );

        if ( test_case == 0 ) {

            m_dot_baseline = t->getResult();
        }

        t->calculateDistance( m_dot_baseline );
    }

    void prepareForRun ( const int test_case, const int num )
    {
        auto t = dynamic_pointer_cast< TestCaseDOT<T> >( this->m_test_cases[ test_case ] );
        t->setX     ( m_x );
        t->setY     ( m_y );
    }
};

static const size_t NUM_TRIALS = 100;

static size_t nums_elements[]{ 32, 64, 128, 512, 2* 1024, 8*1024, 32*1024, 128*1024, 512*1024 };

template<class T>
string testSuitePerType( const bool print_diag )
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
        "32",
        "64",
        "128",
        "512",
        "2K",
        "8K",
        "32K",
        "128K",
        "512K"
    };

    TestResults results{ case_names, header_line };

    const int neon_num_lanes = (is_same<float, T>::value)?4:2;

    for( auto num_elements : nums_elements ) {

        TestExecutorDOT<T> e( results, num_elements, NUM_TRIALS, false, -1.0, 1.0 );

        e.addTestCase( make_shared< TestCaseDOT_baseline <T> > ( case_names[0], num_elements ) );

        e.addTestCase( make_shared< TestCaseDOT_neon<T> > ( case_names[1], num_elements,  1 ) );
        e.addTestCase( make_shared< TestCaseDOT_neon<T> > ( case_names[2], num_elements,  2 ) );
        e.addTestCase( make_shared< TestCaseDOT_neon<T> > ( case_names[3], num_elements,  4 ) );
        if ( num_elements >= 8 * neon_num_lanes ) {
            e.addTestCase( make_shared< TestCaseDOT_neon<T> > ( case_names[4], num_elements,  8 ) );
        }

        e.addTestCase( make_shared< TestCaseDOT_BLAS<T> > ( case_names[5], num_elements ) );

        e.execute( print_diag );
    }

    std::stringstream ss;

    results.printHTML( ss );

    return ss.str();
}

} // namespace Dot

#ifdef __EMSCRIPTEN__

string testSDot()
{
    return Dot::testSuitePerType<float>( true );
}

string testDDot()
{
    return Dot::testSuitePerType<double>( true );
}

EMSCRIPTEN_BINDINGS( dot_module ) {
    emscripten::function( "testSDot", &testSDot );
    emscripten::function( "testDDot", &testDDot );
}

#else

int main(int argc, char* argv[])
{
    const bool print_diag = (argc == 2);

    cout << "sdot (float)\n\n";
    cout << Dot::testSuitePerType<float>( print_diag );
    cout << "\n\n";

    cout << "ddot (double)\n\n";
    cout << Dot::testSuitePerType<double>( print_diag );
    cout << "\n\n";
    return 0;
}

#endif
