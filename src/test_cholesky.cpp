#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <emscripten/bind.h>
#endif

#include <sstream>

#include "test_pattern_generation.h"

#include "test_case_cholesky.h"
#include "test_case_cholesky_baseline.h"
#include "test_case_cholesky_eigen3.h"

namespace Cholesky {

template <class T, bool IS_COL_MAJOR>
class TestExecutorCholesky : public TestExecutor {

  protected:

    // Ax = b A is PD and lower triangular, i.e. the upper part is missing.

    const int             m_dim;
    default_random_engine m_e;

    T*                    m_A;
    T*                    m_b;
    T*                    m_L_baseline;
    T*                    m_x_baseline;

  public:

    TestExecutorCholesky(
        TestResults& results,
        const int    dim,
        const T      condition_num,
        const T      val_low,
        const T      val_high,
        const int    num_trials,
        const bool   repeatable
    )
        :TestExecutor   { results, dim * dim, num_trials }
        ,m_dim          { dim }
        ,m_e            { static_cast<unsigned int>( repeatable ? 0 : chrono::system_clock::now().time_since_epoch().count() ) }
        ,m_A            { new T [ (dim + 1) * dim / 2 ] }
        ,m_b            { new T [ dim ]                 }
        ,m_L_baseline   { new T [ (dim + 1) * dim / 2 ] }
        ,m_x_baseline   { new T [ dim ]                 }
    {
        generateRandomPDLowerMat<T, IS_COL_MAJOR>( m_A, m_dim, condition_num, m_e );

        fillArrayWithRandomValues<T>( m_e, m_b, m_dim, val_low, val_high );
    }

    virtual ~TestExecutorCholesky()
    {
        delete[] m_A;
        delete[] m_b;
        delete[] m_L_baseline;
        delete[] m_x_baseline;
    }

    void prepareForRun ( const int test_case, const int num )
    {
        auto t = dynamic_pointer_cast< TestCaseCholesky<T,IS_COL_MAJOR> >( this->m_test_cases[ test_case ] );
        t->setInitialStates( m_A, m_b );
    }

    void cleanupAfterBatchRuns ( const int test_case )
    {
        auto t = dynamic_pointer_cast< TestCaseCholesky<T,IS_COL_MAJOR> >( this->m_test_cases[ test_case ] );
        if ( test_case == 0 ) {
            memcpy( m_L_baseline, t->getOutputPointer_L(), sizeof(T) * (m_dim+1) * m_dim / 2 );
            memcpy( m_x_baseline, t->getOutputPointer_x(), sizeof(T) * m_dim                 );
        }

        t->compareTruth( m_L_baseline, m_x_baseline );
    }
};

static const size_t NUM_TRIALS = 10;

static int matrix_dims[]={ 64, 128, 256, 512 };

template<class T, bool IS_COL_MAJOR>
string testSuitePerType( const bool print_diag, const T condition_num, const T gen_low, const T gen_high ) {

    vector< string > case_names {
        "plain c++ column-cholesky",
        "plain c++ submatrix-cholesky"
    };

    vector< string > header_line {
        "matrix size",
        "64x64",
        "128x128",
        "256x256",
        "512x512"
    };

    if constexpr (IS_COL_MAJOR) {     
        case_names.push_back( "Eigen3 LLT" );
    }

    TestResults results{ case_names, header_line };

    for( auto& dim : matrix_dims ) {
      
        TestExecutorCholesky<T, IS_COL_MAJOR> e( results, dim, condition_num, gen_low, gen_high, NUM_TRIALS, false );

        e.addTestCase( make_shared< TestCaseCholesky_baseline  <T, IS_COL_MAJOR> > ( case_names[0], dim, false ) );
        e.addTestCase( make_shared< TestCaseCholesky_baseline  <T, IS_COL_MAJOR> > ( case_names[1], dim, true  ) );

        if constexpr (IS_COL_MAJOR) {

            e.addTestCase( make_shared< TestCaseCholesky_eigen3  <T, IS_COL_MAJOR> > ( case_names[2], dim ) );
        }

        e.execute( print_diag );
    }

    std::stringstream ss;

    results.printHTML( ss );

    return ss.str();
}

} // namespace Cholesky

#ifdef __EMSCRIPTEN__

string testCholeskyFloatColMajor()
{
    return Cholesky::testSuitePerType<float, true  > ( true, 10.0, -1.0, 1.0 );
}

string testCholeskyFloatRowMajor()
{
    return Cholesky::testSuitePerType<float, false > ( true, 10.0, -1.0, 1.0 );
}

string testCholeskyDoubleColMajor()
{
    return Cholesky::testSuitePerType<double, true  > ( true, 10.0, -1.0, 1.0 );
}

string testCholeskyDoubleRowMajor()
{
    return Cholesky::testSuitePerType<double, false > ( true, 10.0, -1.0, 1.0 );
}

EMSCRIPTEN_BINDINGS( saxpy_module ) {
    emscripten::function( "testCholeskyFloatColMajor",  &testCholeskyFloatColMajor  );
    emscripten::function( "testCholeskyFloatRowMajor",  &testCholeskyFloatRowMajor  );
    emscripten::function( "testCholeskyDoubleColMajor", &testCholeskyDoubleColMajor );
    emscripten::function( "testCholeskyDoubleRowMajor", &testCholeskyDoubleRowMajor );
}

#else

int main( int argc, char* argv[] )
{
    const bool print_diag = (argc == 2);

    cout << "cholesky (float, col-major)\n\n";
    cout << Cholesky::testSuitePerType<float,  true  > ( print_diag, 10.0, -1.0, 1.0 );
    cout << "\n\n";

    cout << "cholesky (float, row-major)\n\n";
    cout << Cholesky::testSuitePerType<float,  false > ( print_diag, 10.0, -1.0, 1.0 );
    cout << "\n\n";

    cout << "cholesky (double, col-major)\n\n";
    cout << Cholesky::testSuitePerType<double, true  > ( print_diag, 10.0, -1.0, 1.0 );
    cout << "\n\n";

    cout << "cholesky (double, row-major)\n\n";
    cout << Cholesky::testSuitePerType<double, false > ( print_diag, 10.0, -1.0, 1.0 );
    cout << "\n\n";

    return 0;
}

#endif
