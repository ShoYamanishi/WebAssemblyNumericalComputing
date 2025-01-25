#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <emscripten/bind.h>
#endif

#include <sstream>
#include "test_case_with_time_measurements.h"
#include "test_pattern_generation.h"

namespace Convolution2D {

template<class T>
class TestCaseConvolution2D :public TestCaseWithTimeMeasurements {

static constexpr double RMS_PASSING_THRESHOLD = 1.0e-6;

protected:

    const size_t        m_image_width;
    const size_t        m_image_height;
    const T*            m_image_in;
    const T*            m_kernel;
    T*                  m_image_out;

public:

    TestCaseConvolution2D( const string& case_name, const size_t width, const size_t height )
        :TestCaseWithTimeMeasurements { case_name }
        ,m_image_width                { width     }
        ,m_image_height               { height    }
        ,m_image_in                   { nullptr   }
        ,m_kernel                     { nullptr   }
        ,m_image_out                  { nullptr   }
    {
        ;
    }

    virtual ~TestCaseConvolution2D()
    {
        ;
    }

    virtual void compareTruth( const T* const baseline )
    {
        auto rms = getRMSDiffTwoVectors( getImageOut(), baseline, this->m_image_width * this->m_image_height );
        setRMS( rms ) ;

        setTrueFalse( (rms < RMS_PASSING_THRESHOLD) ? true : false );
    }

    virtual void setInitialStates( const T* const image_in, const T* kernel, T* image_out )
    {
        m_image_in  = image_in;
        m_kernel    = kernel;
        m_image_out = image_out;
    }

    virtual void run() = 0;

    virtual T* getImageOut() { return m_image_out; }
};


// Calculate one NxN convolution.
template<class T, int KERNEL_DIM>
static inline T calc_conv_one_point (
    const T* const kernel,
    const T* const in, 
    const size_t   image_width,
    const size_t   image_height,
    const size_t   center_x,
    const size_t   center_y
) {
    static_assert( KERNEL_DIM % 2 == 1 );// DIM must be odd.
    
    constexpr int KERN_OFFSET = KERNEL_DIM/2;

    T sum = 0;

    for ( int kern_y = 0; kern_y < KERNEL_DIM; kern_y++ ) {

        const int image_y = ( kern_y - KERN_OFFSET ) + center_y;

        if ( 0 <= image_y && image_y < image_height ) {

            for ( int kern_x = 0; kern_x < KERNEL_DIM; kern_x++ ) {

                const int image_x = ( kern_x - KERN_OFFSET ) + center_x;

                const T v =   ( 0 <= image_x && image_x < image_width )
                            ? ( kernel[ index_row_major(kern_x, kern_y, KERNEL_DIM) ] * in[ index_row_major(image_x, image_y, image_width) ] )
                            : 0
                            ;
                sum += v;
            }
        }
    }
    return sum;
}


// Calculate NxN convolution.
template<class T, int KERNEL_DIM>
static inline void calc_conv_row_block (

    const T* const kernel,
    const T* const in, 
          T* const out, 
    const size_t   width,
    const size_t   height,
    const size_t   row_start,
    const size_t   row_end_past_one

) {
    for ( size_t y = row_start; y < row_end_past_one; y++ ) {

        for ( size_t x = 0; x < width; x++ ) {

            out[ index_row_major( x, y, width) ] = calc_conv_one_point< T,KERNEL_DIM >( kernel, in, width, height, x, y );
        }
    }
}


template< class T, int KERNEL_DIM >
class TestCaseConvolution2D_baseline : public TestCaseConvolution2D<T> {

  public:
    TestCaseConvolution2D_baseline( const string& case_name, const size_t width, const size_t height )
        :TestCaseConvolution2D<T>( case_name, width, height )
    {
        ;
    }

    virtual ~TestCaseConvolution2D_baseline()
    {
        ;
    }

    virtual void run()
    {
        calc_conv_row_block< T, KERNEL_DIM > (
            this->m_kernel,
            this->m_image_in,
            this->m_image_out,
            this->m_image_width,
            this->m_image_height,
            0,
            this->m_image_height
        );
    }
};


template <class T, int KERNEL_DIM>
class TestExecutorConvolution2D : public TestExecutor {

  protected:

    const int             m_image_width;
    const int             m_image_height;
    const int             m_num_pixels;
    const bool            m_repeatable;
    default_random_engine m_e;
    T* const              m_image_in;
    T* const              m_image_out;
    T* const              m_image_out_baseline;
    T* const              m_kernel;

  public:

    TestExecutorConvolution2D(
        TestResults& results,
        const int    image_width,
        const int    image_height,
        const int    num_trials,
        const bool   repeatable, 
        const T      low,
        const T      high
    )
        :TestExecutor        { results, image_width * image_height, num_trials }
        ,m_image_width       { image_width }
        ,m_image_height      { image_height }
        ,m_num_pixels        { image_width * image_height }
        ,m_repeatable        { repeatable }
        ,m_e                 { static_cast<unsigned int>( repeatable ? 0 : chrono::system_clock::now().time_since_epoch().count() ) }
        ,m_image_in          { new T[ m_num_pixels] }
        ,m_image_out         { new T[ m_num_pixels] }
        ,m_image_out_baseline{ new T[ m_num_pixels] }
        ,m_kernel            { new T[ KERNEL_DIM * KERNEL_DIM ] }
    {
        fillArrayWithRandomValues( m_e,  m_image_in, m_num_pixels,            low, high );
        fillArrayWithRandomValues( m_e,  m_kernel,   KERNEL_DIM * KERNEL_DIM, low, high );
    }

    virtual ~TestExecutorConvolution2D()
    {
        delete[] m_image_in;
        delete[] m_image_out;
        delete[] m_image_out_baseline;
        delete[] m_kernel;
    }

    void prepareForRun ( const int test_case, const int num )
    {
        memset( m_image_out, 0,  sizeof(T) * m_num_pixels );

        auto t = dynamic_pointer_cast< TestCaseConvolution2D<T> >( this->m_test_cases[ test_case ] );
        t->setInitialStates( m_image_in, m_kernel, m_image_out );
    }

    void cleanupAfterBatchRuns ( const int test_case )
    {
        auto t = dynamic_pointer_cast< TestCaseConvolution2D<T> >( this->m_test_cases[ test_case ] );

        if ( test_case == 0 ) {

            memcpy( m_image_out_baseline, m_image_out, sizeof(T) * m_num_pixels );
        }
        t->compareTruth( m_image_out_baseline );
    }
};


static const size_t NUM_TRIALS = 10;

struct image_dim {
    size_t width;
    size_t height;
};

struct image_dim image_dims[]={
    {       64,      64},
    {      128,     128},
    {      256,     256},
    {      512,     512},
    {     1024,    1024},
    {   2*1024,  2*1024}
};

template< class T, int KERNEL_DIM >
string testSuitePerType ( const bool print_diag, const T gen_low, const T gen_high )
{
    vector< string > case_names {
        "plain c++"
    };

    vector< string > header_line {
        "grid size",
        "64x64",
        "128x128",
        "256x256",
        "512x512",
        "1Kx1K",
        "2Kx2K"
    };

    TestResults results{ case_names, header_line };

    for( auto dims : image_dims ) {

        const auto w = dims.width;
        const auto h = dims.height;

        TestExecutorConvolution2D<T, KERNEL_DIM> e( results, w, h, NUM_TRIALS, false, gen_low, gen_high );

        e.addTestCase( make_shared< TestCaseConvolution2D_baseline<T, KERNEL_DIM> > ( case_names[0], w, h ) );

        e.execute( print_diag );
    }

    std::stringstream ss;

    results.printHTML( ss );

    return ss.str();
}

} //namespace Convolution2D

#ifdef __EMSCRIPTEN__

string testConvolution2D()
{
    return Convolution2D::testSuitePerType<float, 5> ( true, -1.0, 1.0 );
}

EMSCRIPTEN_BINDINGS( convolution_2d_module ) {
    emscripten::function( "testConvolution2D", &testConvolution2D );
}

#else

int main( int argc, char* argv[] )
{
    const bool print_diag = (argc == 2);

    cout << "conv 2d 5x5\n\n";
    cout << Convolution2D::testSuitePerType<float, 5> ( print_diag, -1.0, 1.0 );
    cout << "\n\n";

    return 0;
}

#endif
