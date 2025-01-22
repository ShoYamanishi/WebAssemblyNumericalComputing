#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <emscripten/bind.h>
#endif

#include "test_lcp_pattern_generator.h"
#include "test_case_lcp.h"
#include "test_case_lcp_lemke_baseline.h"
#include "test_case_lcp_lemke_neon.h"

#include <sstream>
#include <algorithm>

template <class T, bool IS_COL_MAJOR>
class TestExecutorLCP : public TestExecutor {

  protected:

    LCPPatternGenerator< T, IS_COL_MAJOR> m_generator;
    const int                             m_dim;
    T*                                    m_M;
    T*                                    m_q;

  public:

    TestExecutorLCP(
        TestResults&             results,
        const int                dim,
        const T                  condition_num,
        const T                  val_low,
        const T                  val_high,
        const int                num_trials,
        const bool               repeatable,
        const LCPTestPatternType p_type,
        const std::string        test_pattern_path
    )
        :TestExecutor{ results, dim*dim, num_trials }
        ,m_generator { repeatable, val_low, val_high, test_pattern_path }
        ,m_dim       { dim }
        ,m_M         { new T [ dim * dim ] }
        ,m_q         { new T [ dim ]       }
    {
        m_generator.generateTestPattern( m_dim, m_M, m_q, p_type, condition_num );
    }

    virtual ~TestExecutorLCP()
    {
        delete[] m_M;
        delete[] m_q;
    }

    void prepareForRun ( const int test_case, const int num )
    {
        auto t = dynamic_pointer_cast< TestCaseLCP<T,IS_COL_MAJOR> >( this->m_test_cases[ test_case ] );
        t->setInitialStates( m_M, m_q );
    }

    void cleanupAfterBatchRuns ( const int test_case )
    {
        auto t = dynamic_pointer_cast< TestCaseLCP<T,IS_COL_MAJOR> >( this->m_test_cases[ test_case ] );

        t->checkOutput();
    }
};

static const size_t NUM_TRIALS         = 10;
static const int    MAX_NUM_PIVOTS     = 10000;
//static const int    MAX_NUM_ITERATIONS = 10000;
static const float  EPSILON            = 1.0e-15;
static const bool   REPEATABLE         = false;
//static const int    NUM_PGS_PER_SM     = 10;
//static const float  OMEGA              = 1.0;

int matrix_dims[] = { 64, 128, 256, 512 };

template<class T, bool IS_COL_MAJOR>
string testSuitePerType (
    const bool               print_diag,
    const T                  condition_num,
    const T                  gen_low,
    const T                  gen_high,
    const LCPTestPatternType p_type,
    const std::string        test_pattern_path
) {
    vector< string > case_names {
    };

    vector< string > header_line {
        "matrix size",
        "64x64",
        "128x128",
        "256x256",
        "512x512"
    };

    if constexpr ( !IS_COL_MAJOR ) {
        case_names.push_back( "plain c++" );
        case_names.push_back( "NEON" );
    }

    TestResults results{ case_names, header_line };

    for( auto& dim : matrix_dims ) {
      
        TestExecutorLCP<T, IS_COL_MAJOR> e( results, dim, condition_num, gen_low, gen_high, NUM_TRIALS, REPEATABLE, p_type, test_pattern_path );

        if constexpr ( !IS_COL_MAJOR ) {

            e.addTestCase( make_shared< TestCaseLCP_lemke_baseline<T, IS_COL_MAJOR> >( case_names[0], dim, condition_num, MAX_NUM_PIVOTS, (T)EPSILON, p_type ) );
            e.addTestCase( make_shared< TestCaseLCP_lemke_neon    <T, IS_COL_MAJOR> >( case_names[1], dim, condition_num, MAX_NUM_PIVOTS, (T)EPSILON, p_type ) );
        }

        e.execute( print_diag );
    }

    std::stringstream ss;

    results.printHTML( ss );

    return ss.str();
}

static const std::string test_pattern_path( "test_patterns/" );

#ifdef __EMSCRIPTEN__

string testLCPFloatMu02()
{
    return testSuitePerType< float, false >( true, 0.0, -1.0, 1.0, LCP_REAL_NONSYMMETRIC_MU02, test_pattern_path );
}

string testLCPDoubleMu02()
{
    return testSuitePerType< double, false >( true, 0.0, -1.0, 1.0, LCP_REAL_NONSYMMETRIC_MU02, test_pattern_path );
}

string testLCPFloatMu08()
{
    return testSuitePerType< float, false  >( true, 0.0, -1.0, 1.0, LCP_REAL_NONSYMMETRIC_MU08, test_pattern_path );
}

string testLCPDoubleMu08()
{
    return testSuitePerType< double, false >( true, 0.0, -1.0, 1.0, LCP_REAL_NONSYMMETRIC_MU08, test_pattern_path );
}

string testLCPFloatSymmetric()
{
    return testSuitePerType< float, false  >( true, 0.0, -1.0, 1.0, LCP_REAL_SYMMETRIC, test_pattern_path );
}

string testLCPDoubleSymmetric()
{
    return testSuitePerType< double, false >( true, 0.0, -1.0, 1.0, LCP_REAL_SYMMETRIC, test_pattern_path );
}

EMSCRIPTEN_BINDINGS( saxpy_module ) {
    emscripten::function( "testLCPFloatMu02",       &testLCPFloatMu02       );
    emscripten::function( "testLCPDoubleMu02",      &testLCPDoubleMu02      );
    emscripten::function( "testLCPFloatMu08",       &testLCPFloatMu08       );
    emscripten::function( "testLCPDoubleMu08",      &testLCPDoubleMu08      );
    emscripten::function( "testLCPFloatSymmetric",  &testLCPFloatSymmetric  );
    emscripten::function( "testLCPDoubleSymmetric", &testLCPDoubleSymmetric );
}

#else

int main( int argc, char* argv[] )
{
    const bool print_diag = (argc == 2);


/*
    cout << "lcp (float, random symmetric cond num 1.0)\n\n";
    cout << testSuitePerType<float,  false  > ( print_diag,            1.0, -1.0, 1.0, LCP_RANDOM_DIAGONALLY_DOMINANT_SYMMETRIC, test_pattern_path );
    cout << "\n\n";

    cout << "lcp (float, random symmetric cond num 10.0)\n\n";
    cout << testSuitePerType<float,  false  > ( print_diag,           10.0, -1.0, 1.0, LCP_RANDOM_DIAGONALLY_DOMINANT_SYMMETRIC, test_pattern_path );
    cout << "\n\n";

    cout << "lcp (float, random symmetric cond num 100.0)\n\n";
    cout << testSuitePerType<float,  false  > ( print_diag,          100.0, -1.0, 1.0, LCP_RANDOM_DIAGONALLY_DOMINANT_SYMMETRIC, test_pattern_path );
    cout << "\n\n";

    cout << "lcp (float, random symmetric cond num 1000.0)\n\n";
    cout << testSuitePerType<float,  false  > ( print_diag,         1000.0, -1.0, 1.0, LCP_RANDOM_DIAGONALLY_DOMINANT_SYMMETRIC, test_pattern_path );
    cout << "\n\n";

    cout << "lcp (float, random symmetric cond num 10000.0)\n\n";
    cout << testSuitePerType<float,  false  > ( print_diag,        10000.0, -1.0, 1.0, LCP_RANDOM_DIAGONALLY_DOMINANT_SYMMETRIC, test_pattern_path );
    cout << "\n\n";

    cout << "lcp (float, random symmetric cond num 100000.0)\n\n";
    cout << testSuitePerType<float,  false  > ( print_diag,       100000.0, -1.0, 1.0, LCP_RANDOM_DIAGONALLY_DOMINANT_SYMMETRIC, test_pattern_path );
    cout << "\n\n";

    cout << "lcp (double, random symmetric cond num 1.0)\n\n";
    cout << testSuitePerType<double,  false  > ( print_diag,            1.0, -1.0, 1.0, LCP_RANDOM_DIAGONALLY_DOMINANT_SYMMETRIC, test_pattern_path );
    cout << "\n\n";

    cout << "lcp (double, random symmetric cond num 10.0)\n\n";
    cout << testSuitePerType<double,  false  > ( print_diag,           10.0, -1.0, 1.0, LCP_RANDOM_DIAGONALLY_DOMINANT_SYMMETRIC, test_pattern_path );
    cout << "\n\n";

    cout << "lcp (double, random symmetric cond num 100.0)\n\n";
    cout << testSuitePerType<double,  false  > ( print_diag,          100.0, -1.0, 1.0, LCP_RANDOM_DIAGONALLY_DOMINANT_SYMMETRIC, test_pattern_path );
    cout << "\n\n";

    cout << "lcp (double, random symmetric cond num 1000.0)\n\n";
    cout << testSuitePerType<double,  false  > ( print_diag,         1000.0, -1.0, 1.0, LCP_RANDOM_DIAGONALLY_DOMINANT_SYMMETRIC, test_pattern_path );
    cout << "\n\n";

    cout << "lcp (double, random symmetric cond num 10000.0)\n\n";
    cout << testSuitePerType<double,  false  > ( print_diag,        10000.0, -1.0, 1.0, LCP_RANDOM_DIAGONALLY_DOMINANT_SYMMETRIC, test_pattern_path );
    cout << "\n\n";

    cout << "lcp (double, random symmetric cond num 100000.0)\n\n";
    cout << testSuitePerType<double,  false  > ( print_diag,       100000.0, -1.0, 1.0, LCP_RANDOM_DIAGONALLY_DOMINANT_SYMMETRIC, test_pattern_path );
    cout << "\n\n";

    cout << "lcp (float, random skey-symmetric cond num 1.0)\n\n";
    cout << testSuitePerType<float,  false  > ( print_diag,           1.0, -1.0, 1.0, LCP_RANDOM_DIAGONALLY_DOMINANT_SKEWSYMMETRIC, test_pattern_path );
    cout << "\n\n";

    cout << "lcp (float, random skey-symmetric cond num 10.0)\n\n";
    cout << testSuitePerType<float,  false  > ( print_diag,          10.0, -1.0, 1.0, LCP_RANDOM_DIAGONALLY_DOMINANT_SKEWSYMMETRIC, test_pattern_path );
    cout << "\n\n";

    cout << "lcp (float, random skey-symmetric cond num 100.0)\n\n";
    cout << testSuitePerType<float,  false  > ( print_diag,         100.0, -1.0, 1.0, LCP_RANDOM_DIAGONALLY_DOMINANT_SKEWSYMMETRIC, test_pattern_path );
    cout << "\n\n";

    cout << "lcp (float, random skey-symmetric cond num 1000.0)\n\n";
    cout << testSuitePerType<float,  false  > ( print_diag,        1000.0, -1.0, 1.0, LCP_RANDOM_DIAGONALLY_DOMINANT_SKEWSYMMETRIC, test_pattern_path );
    cout << "\n\n";

    cout << "lcp (float, random skey-symmetric cond num 10000.0)\n\n";
    cout << testSuitePerType<float,  false  > ( print_diag,       10000.0, -1.0, 1.0, LCP_RANDOM_DIAGONALLY_DOMINANT_SKEWSYMMETRIC, test_pattern_path );
    cout << "\n\n";

    cout << "lcp (float, random skey-symmetric cond num 100000.0)\n\n";
    cout << testSuitePerType<float,  false  > ( print_diag,      100000.0, -1.0, 1.0, LCP_RANDOM_DIAGONALLY_DOMINANT_SKEWSYMMETRIC, test_pattern_path );
    cout << "\n\n";

    cout << "lcp (double, random skey-symmetric cond num 1.0)\n\n";
    cout << testSuitePerType<double,  false  > ( print_diag,           1.0, -1.0, 1.0, LCP_RANDOM_DIAGONALLY_DOMINANT_SKEWSYMMETRIC, test_pattern_path );
    cout << "\n\n";

    cout << "lcp (double, random skey-symmetric cond num 10.0)\n\n";
    cout << testSuitePerType<double,  false  > ( print_diag,          10.0, -1.0, 1.0, LCP_RANDOM_DIAGONALLY_DOMINANT_SKEWSYMMETRIC, test_pattern_path );
    cout << "\n\n";

    cout << "lcp (double, random skey-symmetric cond num 100.0)\n\n";
    cout << testSuitePerType<double,  false  > ( print_diag,         100.0, -1.0, 1.0, LCP_RANDOM_DIAGONALLY_DOMINANT_SKEWSYMMETRIC, test_pattern_path );
    cout << "\n\n";

    cout << "lcp (double, random skey-symmetric cond num 1000.0)\n\n";
    cout << testSuitePerType<double,  false  > ( print_diag,        1000.0, -1.0, 1.0, LCP_RANDOM_DIAGONALLY_DOMINANT_SKEWSYMMETRIC, test_pattern_path );
    cout << "\n\n";

    cout << "lcp (double, random skey-symmetric cond num 10000.0)\n\n";
    cout << testSuitePerType<double,  false  > ( print_diag,       10000.0, -1.0, 1.0, LCP_RANDOM_DIAGONALLY_DOMINANT_SKEWSYMMETRIC, test_pattern_path );
    cout << "\n\n";

    cout << "lcp (double, random skey-symmetric cond num 100000.0)\n\n";
    cout << testSuitePerType<double,  false  > ( print_diag,      100000.0, -1.0, 1.0, LCP_RANDOM_DIAGONALLY_DOMINANT_SKEWSYMMETRIC, test_pattern_path );
    cout << "\n\n";
*/

    cout << "lcp (float, real non-symmetric, mu=0.2)\n\n";
    cout << testSuitePerType<float,  false  > ( print_diag, 0.0, -1.0, 1.0, LCP_REAL_NONSYMMETRIC_MU02, test_pattern_path );
    cout << "\n\n";

    cout << "lcp (double, real non-symmetric, mu=0.2)\n\n";
    cout << testSuitePerType<double,  false > ( print_diag, 0.0, -1.0, 1.0, LCP_REAL_NONSYMMETRIC_MU02, test_pattern_path );
    cout << "\n\n";

    cout << "lcp (float, real non-symmetric, mu=0.8)\n\n";
    cout << testSuitePerType<float,  false  > ( print_diag, 0.0, -1.0, 1.0, LCP_REAL_NONSYMMETRIC_MU08, test_pattern_path );
    cout << "\n\n";

    cout << "lcp (double, real non-symmetric, mu=0.8)\n\n";
    cout << testSuitePerType<double,  false > ( print_diag, 0.0, -1.0, 1.0, LCP_REAL_NONSYMMETRIC_MU08, test_pattern_path );
    cout << "\n\n";

    cout << "lcp (float, real symmetric)\n\n";
    cout << testSuitePerType<float,  false  > ( print_diag, 0.0, -1.0, 1.0, LCP_REAL_SYMMETRIC, test_pattern_path );
    cout << "\n\n";

    cout << "lcp (double, real symmetric)\n\n";
    cout << testSuitePerType<double,  false > ( print_diag, 0.0, -1.0, 1.0, LCP_REAL_SYMMETRIC, test_pattern_path );
    cout << "\n\n";

    return 0;
}

#endif
