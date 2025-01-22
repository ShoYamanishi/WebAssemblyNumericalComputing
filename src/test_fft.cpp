#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <emscripten/bind.h>
#endif

#include <arm_neon.h>
#include <sstream>

#include "test_case_with_time_measurements.h"
#include "test_pattern_generation.h"

static const int FFT_LEN_512 = 512;

template< class T >
class TestCaseFFT512Radix2 : public TestCaseWithTimeMeasurements
{

static constexpr double DIST_PASSING_THRESHOLD = 1.0e-4;

protected:

    T* m_time_re;
    T* m_time_im;
    T* m_freq_re;
    T* m_freq_im;

public:
    TestCaseFFT512Radix2( const string& case_name )
        :TestCaseWithTimeMeasurements { case_name }
        ,m_time_re{ nullptr }
        ,m_time_im{ nullptr }
        ,m_freq_re{ nullptr }
        ,m_freq_im{ nullptr }
    {
        static_assert( is_same< float,T >::value || is_same< double,T >::value );
    }

    virtual ~TestCaseFFT512Radix2()
    {
        ;
    }

    virtual void compareTruth( const T* const baseline_re, const T* const baseline_im  )
    {
        auto dist =  getDistTwoVectors( getFreqRe(), baseline_re, FFT_LEN_512 );

        dist += getDistTwoVectors( getFreqIm(), baseline_im, FFT_LEN_512 );

        setDist( dist );

        setTrueFalse( (dist < DIST_PASSING_THRESHOLD) ? true : false );
    }

    virtual void setInitialStates( T* time_re, T* time_im, T* freq_re, T* freq_im )
    {
        m_time_re = time_re;
        m_time_im = time_im;
        m_freq_re = freq_re;
        m_freq_im = freq_im;
    }

    virtual T* getFreqRe()
    {
        return m_freq_re;
    }

    virtual T* getFreqIm()
    {
        return m_freq_im;
    }

    virtual void run() = 0;
};


template< class T >
class TestCaseFFT512Radix2_baseline : public TestCaseFFT512Radix2<T> {

  protected:

    int m_shuffled_indices_512_radix_2 [ FFT_LEN_512 ];
    T   m_cos_0_to_minus_pi_256        [ 256 ];
    T   m_sin_0_to_minus_pi_256        [ 256 ];

    void deinterleave( int* in, const int len, int* out )
    {
        int* out_even = out;
        int* out_odd  = &(out[len/2]);

        for ( int i = 0; i < len; i++ )  {
            if ( i % 2 == 0 ) {
                out_even[i/2] = in[i];
            }
            else {
                out_odd[i/2]  = in[i];
            }
        }
        if (len > 4) {
            deinterleave( out_even, len/2, in           );
            deinterleave( out_odd,  len/2, &(in[len/2]) );
        }    
    }

    void generate_shuffled_indices_512()
    {
        int indices1[ FFT_LEN_512 ];
        int indices2[ FFT_LEN_512 ];

        for ( int i = 0; i < 512; i++ ) {
            indices1[i] = i;
        }

        deinterleave( indices1, 512, indices2 );

        for ( int i = 0; i < 512; i++ ) {

            m_shuffled_indices_512_radix_2[i] = indices1[i];
        }
    }

    void shuffle_values_512_radix_2( T* src, T* dst ) {

        for ( int i = 0; i < FFT_LEN_512; i++ ) {
            dst[i] = src[ m_shuffled_indices_512_radix_2[i] ];
        }
    }

    void make_cos_sin_tables_0_to_minus_pi_step_256()
    {
        for ( int k = 0; k < 256 ; k++ ) {

            const T theta = -1.0 * M_PI * (T)k / 256.0;

            m_cos_0_to_minus_pi_256[ k ] = cos( theta ); // re
            m_sin_0_to_minus_pi_256[ k ] = sin( theta ); // im
        }
    }

    inline void butterfly_one_pair(

        const int straddle,
        const int trig_table_multiple,
        const int index_within_block,

        T* in_re,
        T* in_im,
        T* out_re,
        T* out_im
    ) {
        const T in_re1    = in_re[ index_within_block            ];
        const T in_im1    = in_im[ index_within_block            ];
        const T in_re2    = in_re[ index_within_block + straddle ];
        const T in_im2    = in_im[ index_within_block + straddle ];

        const int table_index = 0xff & ( trig_table_multiple * index_within_block );

        const T tw_re     = m_cos_0_to_minus_pi_256[ table_index ];
        const T tw_im     = m_sin_0_to_minus_pi_256[ table_index ];

        const T offset_re = tw_re * in_re2 - tw_im * in_im2;
        const T offset_im = tw_re * in_im2 + tw_im * in_re2;

        out_re[ index_within_block            ] = in_re1 + offset_re;
        out_im[ index_within_block            ] = in_im1 + offset_im;
        out_re[ index_within_block + straddle ] = in_re1 - offset_re;
        out_im[ index_within_block + straddle ] = in_im1 - offset_im;
    }


    inline void butterfly_one_layer(

        const int straddle,
        const int trig_table_multiple,

        T* in_re,
        T* in_im,
        T* out_re,
        T* out_im
    ) {
        for ( int block = 0; block < FFT_LEN_512; block += straddle*2 ) {

            T* in_re_p  = &( in_re[block]  );
            T* in_im_p  = &( in_im[block]  );
            T* out_re_p = &( out_re[block] );
            T* out_im_p = &( out_im[block] );

            for ( int i = 0; i < straddle; i++ ) {
                butterfly_one_pair( straddle, trig_table_multiple, i, in_re_p, in_im_p, out_re_p, out_im_p );
            }
        }
    }

    void cfft_512_forward()
    {
        T re1[FFT_LEN_512];
        T im1[FFT_LEN_512];
        T re2[FFT_LEN_512];
        T im2[FFT_LEN_512];

        shuffle_values_512_radix_2( this->m_time_re, re1 );
        shuffle_values_512_radix_2( this->m_time_im, im1 );
        butterfly_one_layer(   1, 256, re1, im1, re2, im2 );
        butterfly_one_layer(   2, 128, re2, im2, re1, im1 );
        butterfly_one_layer(   4,  64, re1, im1, re2, im2 );
        butterfly_one_layer(   8,  32, re2, im2, re1, im1 );
        butterfly_one_layer(  16,  16, re1, im1, re2, im2 );
        butterfly_one_layer(  32,   8, re2, im2, re1, im1 );
        butterfly_one_layer(  64,   4, re1, im1, re2, im2 );
        butterfly_one_layer( 128,   2, re2, im2, re1, im1 );
        butterfly_one_layer( 256,   1, re1, im1, re2, im2 );
        memcpy( this->m_freq_re, re2, sizeof(T) * FFT_LEN_512 );
        memcpy( this->m_freq_im, im2, sizeof(T) * FFT_LEN_512 );
   }

  public:

    TestCaseFFT512Radix2_baseline( const string& case_name )
        :TestCaseFFT512Radix2<T>{ case_name }
    {
        generate_shuffled_indices_512();
        make_cos_sin_tables_0_to_minus_pi_step_256();
    }

    virtual ~TestCaseFFT512Radix2_baseline()
    {
        ;
    }

    virtual void run()
    {
        cfft_512_forward();
    }
};


template< class T >
class TestCaseFFT512Radix2_NEON : public TestCaseFFT512Radix2_baseline<T> {

  protected:

    inline void butterfly_four_pairs_NEON(

        const int straddle,
        const int trig_table_multiple,
        const int index_within_block,

        T* in_re,
        T* in_im,
        T* out_re,
        T* out_im
    ) {
        if constexpr ( is_same< float,T >::value ) {

            const float32x4_t in_re1 = vld1q_f32( &( in_re[ index_within_block            ] ) );
            const float32x4_t in_im1 = vld1q_f32( &( in_im[ index_within_block            ] ) );
            const float32x4_t in_re2 = vld1q_f32( &( in_re[ index_within_block + straddle ] ) );
            const float32x4_t in_im2 = vld1q_f32( &( in_im[ index_within_block + straddle ] ) );

            const int         table_index1 = 0xff & ( trig_table_multiple * index_within_block         );
            const int         table_index2 = 0xff & ( trig_table_multiple * ( index_within_block + 1 ) );
            const int         table_index3 = 0xff & ( trig_table_multiple * ( index_within_block + 2 ) );
            const int         table_index4 = 0xff & ( trig_table_multiple * ( index_within_block + 3 ) );

            const float32x4_t tw_re  = { this->m_cos_0_to_minus_pi_256[ table_index1 ] , 
                                         this->m_cos_0_to_minus_pi_256[ table_index2 ] , 
                                         this->m_cos_0_to_minus_pi_256[ table_index3 ] , 
                                         this->m_cos_0_to_minus_pi_256[ table_index4 ]  };

            const float32x4_t tw_im  = { this->m_sin_0_to_minus_pi_256[ table_index1 ] , 
                                         this->m_sin_0_to_minus_pi_256[ table_index2 ] , 
                                         this->m_sin_0_to_minus_pi_256[ table_index3 ] , 
                                         this->m_sin_0_to_minus_pi_256[ table_index4 ]  };

            // const float offset_re = tw_re * v2_re - tw_im * v2_im;
            const float32x4_t offset_re_part1 = vmulq_f32( tw_re, in_re2 );
            const float32x4_t offset_re       = vmlsq_f32( offset_re_part1, tw_im, in_im2 );

            // const float offset_im = tw_re * v2_im + tw_im * v2_re;
            const float32x4_t offset_im_part1 = vmulq_f32( tw_re, in_im2 );
            const float32x4_t offset_im       = vmlaq_f32( offset_im_part1, tw_im, in_re2 );

            const float32x4_t v1_re = vaddq_f32( in_re1, offset_re );
            const float32x4_t v1_im = vaddq_f32( in_im1, offset_im );
            const float32x4_t v2_re = vsubq_f32( in_re1, offset_re );
            const float32x4_t v2_im = vsubq_f32( in_im1, offset_im );

            vst1q_f32( &( out_re[ index_within_block            ] ), v1_re );
            vst1q_f32( &( out_im[ index_within_block            ] ), v1_im );
            vst1q_f32( &( out_re[ index_within_block + straddle ] ), v2_re );
            vst1q_f32( &( out_im[ index_within_block + straddle ] ), v2_im );
        }
        else {

            const float64x2_t in_re1_1 = vld1q_f64( &( in_re[ index_within_block                ] ) );
            const float64x2_t in_re1_2 = vld1q_f64( &( in_re[ index_within_block + 2            ] ) );
            const float64x2_t in_im1_1 = vld1q_f64( &( in_im[ index_within_block                ] ) );
            const float64x2_t in_im1_2 = vld1q_f64( &( in_im[ index_within_block + 2            ] ) );
            const float64x2_t in_re2_1 = vld1q_f64( &( in_re[ index_within_block + straddle     ] ) );
            const float64x2_t in_re2_2 = vld1q_f64( &( in_re[ index_within_block + straddle + 2 ] ) );
            const float64x2_t in_im2_1 = vld1q_f64( &( in_im[ index_within_block + straddle     ] ) );
            const float64x2_t in_im2_2 = vld1q_f64( &( in_im[ index_within_block + straddle + 2 ] ) );

            const int         table_index1 = 0xff & ( trig_table_multiple * index_within_block         );
            const int         table_index2 = 0xff & ( trig_table_multiple * ( index_within_block + 1 ) );
            const int         table_index3 = 0xff & ( trig_table_multiple * ( index_within_block + 2 ) );
            const int         table_index4 = 0xff & ( trig_table_multiple * ( index_within_block + 3 ) );

            const float64x2_t tw_re_1  = { this->m_cos_0_to_minus_pi_256[ table_index1 ] , 
                                           this->m_cos_0_to_minus_pi_256[ table_index2 ]  }; 
            const float64x2_t tw_re_2  = { this->m_cos_0_to_minus_pi_256[ table_index3 ] , 
                                           this->m_cos_0_to_minus_pi_256[ table_index4 ]  };

            const float64x2_t tw_im_1  = { this->m_sin_0_to_minus_pi_256[ table_index1 ] , 
                                           this->m_sin_0_to_minus_pi_256[ table_index2 ] };
            const float64x2_t tw_im_2  = { this->m_sin_0_to_minus_pi_256[ table_index3 ] , 
                                           this->m_sin_0_to_minus_pi_256[ table_index4 ]  };

            // const float offset_re = tw_re * v2_re - tw_im * v2_im;
            const float64x2_t offset_re_part1_1 = vmulq_f64( tw_re_1, in_re2_1 );
            const float64x2_t offset_re_part1_2 = vmulq_f64( tw_re_2, in_re2_2 );
            const float64x2_t offset_re_1       = vmlsq_f64( offset_re_part1_1, tw_im_1, in_im2_1 );
            const float64x2_t offset_re_2       = vmlsq_f64( offset_re_part1_2, tw_im_2, in_im2_2 );

            // const float offset_im = tw_re * v2_im + tw_im * v2_re;
            const float64x2_t offset_im_part1_1 = vmulq_f64( tw_re_1, in_im2_1 );
            const float64x2_t offset_im_part1_2 = vmulq_f64( tw_re_2, in_im2_2 );
            const float64x2_t offset_im_1       = vmlaq_f64( offset_im_part1_1, tw_im_1, in_re2_1 );
            const float64x2_t offset_im_2       = vmlaq_f64( offset_im_part1_2, tw_im_2, in_re2_2 );

            const float64x2_t v1_re_1 = vaddq_f64( in_re1_1, offset_re_1 );
            const float64x2_t v1_re_2 = vaddq_f64( in_re1_2, offset_re_2 );
            const float64x2_t v1_im_1 = vaddq_f64( in_im1_1, offset_im_1 );
            const float64x2_t v1_im_2 = vaddq_f64( in_im1_2, offset_im_2 );
            const float64x2_t v2_re_1 = vsubq_f64( in_re1_1, offset_re_1 );
            const float64x2_t v2_re_2 = vsubq_f64( in_re1_2, offset_re_2 );
            const float64x2_t v2_im_1 = vsubq_f64( in_im1_1, offset_im_1 );
            const float64x2_t v2_im_2 = vsubq_f64( in_im1_2, offset_im_2 );

            vst1q_f64( &( out_re[ index_within_block                ] ), v1_re_1 );
            vst1q_f64( &( out_re[ index_within_block + 2            ] ), v1_re_2 );
            vst1q_f64( &( out_im[ index_within_block                ] ), v1_im_1 );
            vst1q_f64( &( out_im[ index_within_block + 2            ] ), v1_im_2 );
            vst1q_f64( &( out_re[ index_within_block + straddle     ] ), v2_re_1 );
            vst1q_f64( &( out_re[ index_within_block + straddle + 2 ] ), v2_re_2 );
            vst1q_f64( &( out_im[ index_within_block + straddle     ] ), v2_im_1 );
            vst1q_f64( &( out_im[ index_within_block + straddle + 2 ] ), v2_im_2 );
        }
    }

    inline void butterfly_one_layer_NEON(

        const int straddle,
        const int trig_table_multiple,

        T* in_re,
        T* in_im,
        T* out_re,
        T* out_im
    ) {
        for ( int block = 0; block < FFT_LEN_512; block += straddle*2 ) {

            T* in_re_p  = &( in_re[block]  );
            T* in_im_p  = &( in_im[block]  );
            T* out_re_p = &( out_re[block] );
            T* out_im_p = &( out_im[block] );

            for ( int i = 0; i < straddle; i+= 4 ) {
                butterfly_four_pairs_NEON( straddle, trig_table_multiple, i, in_re_p, in_im_p, out_re_p, out_im_p );
            }
        }
    }

    void cfft_512_forward_NEON()
    {
        T re1[FFT_LEN_512];
        T im1[FFT_LEN_512];
        T re2[FFT_LEN_512];
        T im2[FFT_LEN_512];

        this->shuffle_values_512_radix_2( this->m_time_re, re1 );
        this->shuffle_values_512_radix_2( this->m_time_im, im1 );
        this->butterfly_one_layer(   1, 256, re1, im1, re2, im2 );
        this->butterfly_one_layer(   2, 128, re2, im2, re1, im1 );
        butterfly_one_layer_NEON(   4,  64, re1, im1, re2, im2 );
        butterfly_one_layer_NEON(   8,  32, re2, im2, re1, im1 );
        butterfly_one_layer_NEON(  16,  16, re1, im1, re2, im2 );
        butterfly_one_layer_NEON(  32,   8, re2, im2, re1, im1 );
        butterfly_one_layer_NEON(  64,   4, re1, im1, re2, im2 );
        butterfly_one_layer_NEON( 128,   2, re2, im2, re1, im1 );
        butterfly_one_layer_NEON( 256,   1, re1, im1, re2, im2 );

        memcpy( this->m_freq_re, re2, sizeof(T) * FFT_LEN_512 );
        memcpy( this->m_freq_im, im2, sizeof(T) * FFT_LEN_512 );
   }

  public:

    TestCaseFFT512Radix2_NEON( const string& case_name )
        :TestCaseFFT512Radix2_baseline<T>{ case_name }
    {
        ;
    }

    virtual ~TestCaseFFT512Radix2_NEON()
    {
        ;
    }

    virtual void run()
    {
        cfft_512_forward_NEON();
    }
};



template <class T>
class TestExecutorFFT512Radix2 : public TestExecutor {

  protected:
    default_random_engine m_e;

    T* m_time_re;
    T* m_time_im;
    T* m_freq_re;
    T* m_freq_im;
    T* m_freq_re_baseline;
    T* m_freq_im_baseline;

  public:

    TestExecutorFFT512Radix2 (
        TestResults& results,
        const T      max_amp,
        const int    num_sines,
        const int    num_trials,
        const bool   repeatable
    )
        :TestExecutor      { results, 512, num_trials }
        ,m_e               { static_cast<unsigned int>( repeatable ? 0 : chrono::system_clock::now().time_since_epoch().count() ) }
        ,m_time_re         { new T [ FFT_LEN_512 ] }
        ,m_time_im         { new T [ FFT_LEN_512 ] }
        ,m_freq_re         { new T [ FFT_LEN_512 ] }
        ,m_freq_im         { new T [ FFT_LEN_512 ] }
        ,m_freq_re_baseline{ new T [ FFT_LEN_512 ] }
        ,m_freq_im_baseline{ new T [ FFT_LEN_512 ] }
    {
        generateRandomTimeVector512<T>( m_time_re, m_time_im, max_amp, num_sines, m_e );
    }

    void prepareForRun ( const int test_case, const int num )
    {
        auto t = dynamic_pointer_cast< TestCaseFFT512Radix2<T> >( this->m_test_cases[ test_case ] );
        t->setInitialStates( m_time_re, m_time_im, m_freq_re, m_freq_im );
    }


    void cleanupAfterBatchRuns ( const int test_case )
    {
        auto t = dynamic_pointer_cast< TestCaseFFT512Radix2<T> >( this->m_test_cases[ test_case ] );

        if ( test_case == 0 ) {

            memcpy( m_freq_re_baseline, t->getFreqRe(), sizeof(T)* FFT_LEN_512 );
            memcpy( m_freq_im_baseline, t->getFreqIm(), sizeof(T)* FFT_LEN_512 );
        }

        t->compareTruth( m_freq_re_baseline, m_freq_im_baseline );
    }

    virtual ~TestExecutorFFT512Radix2 ()
    {
        delete[] m_time_re;
        delete[] m_time_im;
        delete[] m_freq_re;
        delete[] m_freq_im;
        delete[] m_freq_re_baseline;
        delete[] m_freq_im_baseline;
    }
};


static const size_t NUM_TRIALS = 100;

template<class T>
string testSuitePerType ( const bool print_diag, const T max_amp, const int num_sines )
{
    vector< string > case_names {
        "plain c++",
        "NEON"
    };

    vector< string > header_line {
        "FFT 512",
        "-"
    };

    TestResults results{ case_names, header_line };

    TestExecutorFFT512Radix2<T> e( results, max_amp, num_sines, NUM_TRIALS, false );

    e.addTestCase( make_shared< TestCaseFFT512Radix2_baseline <T> > ( case_names[0] ) );

    e.addTestCase( make_shared< TestCaseFFT512Radix2_NEON  <T> > ( case_names[1] ) );

    e.execute( print_diag );

    std::stringstream ss;

    results.printHTML( ss );

    return ss.str();
}

#ifdef __EMSCRIPTEN__

string testFFTFloat()
{
    return testSuitePerType<float>( true, 10.0, 20 );
}

string testFFTDouble()
{
    return testSuitePerType<double>( true, 10.0, 20 );
}

EMSCRIPTEN_BINDINGS( saxpy_module ) {
    emscripten::function( "testFFTFloat",  &testFFTFloat  );
    emscripten::function( "testFFTDouble", &testFFTDouble );
}

#else

int main( int argc, char* argv[] )
{
    const bool print_diag = (argc == 2);

    cout << "fft-512 (float)\n\n";
    cout << testSuitePerType<float > ( print_diag, 10.0, 20 );
    cout << "\n\n";    

    cout << "fft-512 (double)\n\n";
    cout << testSuitePerType<double> ( print_diag, 10.0, 20 );
    cout << "\n\n";    

    return 0;
}

#endif
