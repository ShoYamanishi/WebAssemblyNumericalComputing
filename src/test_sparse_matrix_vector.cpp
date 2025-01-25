#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <emscripten/bind.h>
#endif

#include <sstream>

#include "test_case_with_time_measurements.h"
#include "test_pattern_generation.h"

namespace SparseMatrixVector {

template<class T>
class TestCaseSPMV :public TestCaseWithTimeMeasurements {

static constexpr double RMS_PASSING_THRESHOLD = 1.0e-6;

protected:
    const int m_M;
    const int m_N;
    const int m_num_nonzero_elems;
    int*      m_csr_row_ptrs;
    int*      m_csr_columns;
    T*        m_csr_values;
    T*        m_csr_vector;
    T*        m_output_vector;

public:

    TestCaseSPMV(
        const string& case_name,
        const int     M,
        const int     N,
        const int     num_nonzero_elems
    )
        :TestCaseWithTimeMeasurements { case_name         }
        ,m_M                          { M                 }
        ,m_N                          { N                 }
        ,m_num_nonzero_elems          { num_nonzero_elems }
        ,m_csr_row_ptrs               { nullptr           }
        ,m_csr_columns                { nullptr           }
        ,m_csr_values                 { nullptr           }
        ,m_csr_vector                 { nullptr           }
        ,m_output_vector              { nullptr           }
    {
        static_assert( is_same< float,T >::value ||  is_same< double,T >::value );
    }

    virtual ~TestCaseSPMV()
    {
        ;
    }

    virtual void compareTruth( const T* const baseline )
    {
        auto rms = getRMSDiffTwoVectors( getOutputVector(), baseline, m_M );

        this->setRMS( rms );

        setTrueFalse( (rms < RMS_PASSING_THRESHOLD) ? true : false );
    }

    virtual void setInitialStates( int* csr_row_ptrs, int* csr_columns, T* csr_values, T* csr_vector, T* output_vector )
    {
        m_csr_row_ptrs  = csr_row_ptrs;
        m_csr_columns   = csr_columns;
        m_csr_values    = csr_values;
        m_csr_vector    = csr_vector;
        m_output_vector = output_vector;
    }

    virtual T* getOutputVector()
    {
        return m_output_vector;
    }

    virtual void run() = 0;
};


template<class T>
class TestCaseSPMV_baseline : public TestCaseSPMV<T> {

  public:
    TestCaseSPMV_baseline( const string& case_name, const int M, const int N, const int num_nonzero_elems )
        :TestCaseSPMV<T>{ case_name, M, N, num_nonzero_elems }
    {
        ;
    }

    virtual ~TestCaseSPMV_baseline()
    {
        ;
    }

    virtual void run()
    {
        for ( int i = 0; i < this->m_M; i++ ) {

            this->m_output_vector[ i ] = 0.0;

            for ( int j = this->m_csr_row_ptrs[i]; j < this->m_csr_row_ptrs[i+1]; j++ ) {

                this->m_output_vector[i] += ( this->m_csr_values[j] * this->m_csr_vector[ this->m_csr_columns[j] ] ) ;
            }
        }
    }
};


template <class T>
class TestExecutorSPMV : public TestExecutor {

  protected:

    const int             m_M;
    const int             m_N;
    const int             m_num_nonzero_elems;
    default_random_engine m_e;
    int*                  m_csr_row_ptrs;
    int*                  m_csr_columns;
    T*                    m_csr_values;
    T*                    m_csr_vector;
    T*                    m_output_vector;
    T*                    m_output_vector_baseline;

  public:

    TestExecutorSPMV(
        TestResults& results,
        const int  M,
        const int  N,
        const int  num_nonzero_elems,
        const int  num_trials,
        const bool repeatable,
        const T    low,
        const T    high
    )
        :TestExecutor             { results, num_nonzero_elems, num_trials }
        ,m_M                      { M }
        ,m_N                      { N }
        ,m_num_nonzero_elems      { num_nonzero_elems }
        ,m_e                      { static_cast<unsigned int>( repeatable ? 0 : chrono::system_clock::now().time_since_epoch().count() ) }
        ,m_csr_row_ptrs           { new int [ M + 1 ] }
        ,m_csr_columns            { new int [ num_nonzero_elems ] }
        ,m_csr_values             { new T   [ num_nonzero_elems ] }
        ,m_csr_vector             { new T   [ N ] }
        ,m_output_vector          { new T   [ M ] }
        ,m_output_vector_baseline { new T   [ M ] }
    {
        memset( m_output_vector,          0, sizeof(T) * m_M );
        memset( m_output_vector_baseline, 0, sizeof(T) * m_M );

        generateCSR( m_M, m_N, m_num_nonzero_elems, low, high, m_e, m_csr_row_ptrs, m_csr_columns, m_csr_values, m_csr_vector) ;
    }

    virtual ~TestExecutorSPMV()
    {
        delete[] m_csr_row_ptrs;
        delete[] m_csr_columns;
        delete[] m_csr_values;
        delete[] m_csr_vector;
        delete[] m_output_vector;
        delete[] m_output_vector_baseline;
    }

    void prepareForRun ( const int test_case, const int num )
    {
        memset( m_output_vector, 0,  sizeof(T) * m_M );

        auto t = dynamic_pointer_cast< TestCaseSPMV<T> >( this->m_test_cases[ test_case ] );
        t->setInitialStates( m_csr_row_ptrs, m_csr_columns, m_csr_values, m_csr_vector, m_output_vector );
    }

    void cleanupAfterBatchRuns ( const int test_case )
    {
        auto t = dynamic_pointer_cast< TestCaseSPMV<T> >( this->m_test_cases[ test_case ] );

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
    float  ratio;
};

static struct matrix_dim matrix_dims[]={

      {     256,     256, 0.1  }
    , {     512,     512, 0.1  }
    , {    1024,    1024, 0.1  }
    , {    2048,    2048, 0.1  }
};

template<class T>
string testSuitePerType( const bool print_diag, const T gen_low, const T gen_high )
{
    vector< string > case_names {
        "plain c++"
    };

    vector< string > header_line {
        "matrix size & occupancy",
        "256x256, 0.1",
        "512x512, 0.1",
        "1Kx1K, 0.1",
        "2Kx2K, 0.1"
    };

    TestResults results{ case_names, header_line };

    for( auto& dims : matrix_dims ) {

        const auto M = dims.M;
        const auto N = dims.N;

        const int num_nonzero_elems = (int)( ((float)(M*N))*dims.ratio );

        TestExecutorSPMV<T> e( results, M, N, num_nonzero_elems, NUM_TRIALS, false, gen_low, gen_high );

        e.addTestCase( make_shared< TestCaseSPMV_baseline<T> > ( case_names[0], M, N, num_nonzero_elems ) );
        e.execute( print_diag );
    }

    std::stringstream ss;

    results.printHTML( ss );

    return ss.str();
}

} // namespace SparseMatrixVector

#ifdef __EMSCRIPTEN__

string testSparseMatrixVectorFloat()
{
    return SparseMatrixVector::testSuitePerType<float> ( true, -1.0, 1.0 );
}

string testSparseMatrixVectorDouble()
{
    return SparseMatrixVector::testSuitePerType<double> ( true, -1.0, 1.0 );
}

EMSCRIPTEN_BINDINGS( saxpy_module ) {
    emscripten::function( "testSparseMatrixVectorFloat", &testSparseMatrixVectorFloat  );
    emscripten::function( "testSparseMatrixVectorDouble",&testSparseMatrixVectorDouble );
}

#else

int main( int argc, char* argv[] )
{
    const bool print_diag = (argc == 2);

    cout << "sparse mul mat * vec (float)\n\n";
    cout << SparseMatrixVector::testSuitePerType<float> ( print_diag, -1.0, 1.0 );
    cout << "\n\n";

    cout << "sparse mul mat * vec (double)\n\n";
    cout << SparseMatrixVector::testSuitePerType<double> ( print_diag, -1.0, 1.0 );
    cout << "\n\n";

    return 0;
}

#endif
