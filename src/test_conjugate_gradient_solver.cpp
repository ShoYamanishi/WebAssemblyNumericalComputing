#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <emscripten/bind.h>
#endif

#include <arm_neon.h>
#include <sstream>

#include "test_case_with_time_measurements.h"
#include "test_pattern_generation.h"

template< class T, bool IS_COL_MAJOR >
class TestCaseConjugateGradientSolver : public TestCaseWithTimeMeasurements  {

static constexpr double RMS_PASSING_THRESHOLD = 1.0e-4;

protected:

    const int m_dim;
    const int m_max_iteration;
    int       m_iterations;
    T*        m_A;
    T*        m_b;
    T*        m_x;
    T*        m_Ap;
    T*        m_r;
    T*        m_p;
    const T   m_epsilon;
    const int m_condition_num;

public:

    TestCaseConjugateGradientSolver( 
        const string& case_name,
        const int     dim,
        const int     max_iteration, 
        const T       epsilon, 
        const int     condition_num
    )
        :TestCaseWithTimeMeasurements{ case_name     }
        ,m_dim                       { dim           }
        ,m_max_iteration             { max_iteration }
        ,m_iterations                { 0             }
        ,m_A                         { nullptr       }
        ,m_b                         { nullptr       }
        ,m_x                         { nullptr       }
        ,m_Ap                        { nullptr       }
        ,m_r                         { nullptr       }
        ,m_p                         { nullptr       }
        ,m_epsilon                   { epsilon       }
        ,m_condition_num             { condition_num }
    {
     	static_assert( is_same< float,T >::value || is_same< double,T >::value );

        m_x  = new T[ dim ];
        m_Ap = new T[ dim ];
        m_r  = new T[ dim ];
        m_p  = new T[ dim ];
    }

    virtual ~TestCaseConjugateGradientSolver()
    {
        delete[] m_x;
        delete[] m_Ap;
        delete[] m_r;
        delete[] m_p;
    }

    virtual void compareTruth()
    {
        // calculate Ax and compare it with b.
        for ( int i = 0; i < m_dim ; i++ ) {
            m_Ap[i] = 0.0;
            for ( int j = 0; j < m_dim ; j++ ) {
                m_Ap[i] += (m_A[i * m_dim + j] * m_x[j]);
            }
        }

        auto rms = getRMSDiffTwoVectors( m_Ap, m_b, m_dim );
        this->setRMS( rms );

        setTrueFalse( (rms < RMS_PASSING_THRESHOLD) ? true : false );
    }

    virtual void setInitialStates( T* A,T* b )
    {
        m_A = A;
        m_b = b;
    }

    virtual void run() = 0;
};


template< class T, bool IS_COL_MAJOR >
class TestCaseConjugateGradientSolver_baseline : public TestCaseConjugateGradientSolver< T, IS_COL_MAJOR > {

  public:

    TestCaseConjugateGradientSolver_baseline(
        const string& case_name,
        const int     dim,
        const int     max_iteration,
        const T       epsilon,
        const int     condition_num
    )
        :TestCaseConjugateGradientSolver< T, IS_COL_MAJOR >( case_name, dim, max_iteration, epsilon, condition_num )
    {
        ;
    }

    virtual ~TestCaseConjugateGradientSolver_baseline()
    {
        ;
    }

    virtual void run()
    {
        memset( this->m_x, 0, sizeof(T) * this->m_dim );

        for ( int i = 0; i < this->m_dim ; i++ ) {

            this->m_Ap[i] = 0.0;

            for ( int j = 0; j < this->m_dim ; j++ ) {

                this->m_Ap[i] += ( this->m_A[ i * this->m_dim + j ] * this->m_x[j] );
            }
            this->m_r[i] = this->m_b[i] - this->m_Ap[i];
            this->m_p[i] = this->m_r[i];
        }

        T max_abs_r = 0.0;

        for ( int i = 0; i < this->m_dim ; i++ ) {

            max_abs_r = max( max_abs_r , fabs(this->m_r[i]) );
        }
        
        if ( max_abs_r < this->m_epsilon ) {
            this->m_iterations = 0;
            return;
        }

        for( this->m_iterations = 1; this->m_iterations <= this->m_max_iteration; this->m_iterations++ ) {

            for ( int i = 0; i < this->m_dim ; i++ ) {

                this->m_Ap[i] = 0.0;

                for ( int j = 0; j < this->m_dim ; j++ ) {

                    this->m_Ap[i] += ( this->m_A[ i * this->m_dim + j ] * this->m_p[j] );
                }
            }

            T rtr = 0.0;
            T pAp = 0.0;

            for ( int i = 0; i < this->m_dim ; i++ ) {

                rtr += ( this->m_r[i] * this->m_r[i]  );
                pAp += ( this->m_p[i] * this->m_Ap[i] );

            }

            const T alpha = rtr / pAp;

            T rtr2 = 0.0;

            for ( int i = 0; i < this->m_dim ; i++ ) {

                this->m_x[i] = this->m_x[i] + alpha * this->m_p[i];
                this->m_r[i] = this->m_r[i] - alpha * this->m_Ap[i];

                rtr2 += ( this->m_r[i] * this->m_r[i] );
            }

            max_abs_r = 0.0;
            for ( int i = 0; i < this->m_dim ; i++ ) {

                max_abs_r = max( max_abs_r , fabs(this->m_r[i]) );
            }
        
            if ( max_abs_r < this->m_epsilon ) {
                return;
            }

            const T beta = rtr2 / rtr;

            for ( int i = 0; i < this->m_dim ; i++ ) {

                this->m_p[i] = this->m_r[i] + beta * this->m_p[i];
            }
        }
    }
};


template<class T, bool IS_COL_MAJOR>
class TestExecutorConjugateGradientSolver : public TestExecutor {

  protected:
    const int             m_dim;
    default_random_engine m_e;
    T*                    m_A;
    T*                    m_b;

  public:

    TestExecutorConjugateGradientSolver (
        TestResults& results,
        const int    dim,
        const T      condition_num,
        const T      val_low,
        const T      val_high,
        const int    num_trials,
        const bool   repeatable
    )
        :TestExecutor{ results, dim * dim, num_trials }
        ,m_dim       { dim }
        ,m_e         { static_cast<unsigned int>( repeatable ? 0 : chrono::system_clock::now().time_since_epoch().count() ) }
        ,m_A         { new T [ dim * dim ] }
        ,m_b         { new T [ dim ]       }
    {

        generateRandomPDMat<T, IS_COL_MAJOR>( m_A, m_dim, condition_num, m_e );

        fillArrayWithRandomValues( m_e, m_b, m_dim, val_low, val_high );
    }

    void prepareForRun( const int test_case, const int num )
    {
        auto t = dynamic_pointer_cast< TestCaseConjugateGradientSolver<T,IS_COL_MAJOR> >( this->m_test_cases[ test_case ] );

        t->setInitialStates( m_A, m_b );
    }

    void cleanupAfterBatchRuns ( const int test_case )
    {
        auto t = dynamic_pointer_cast< TestCaseConjugateGradientSolver<T,IS_COL_MAJOR> >( this->m_test_cases[ test_case ] );

        t->compareTruth();
    }

    virtual ~TestExecutorConjugateGradientSolver()
    {
        delete[] m_A;
        delete[] m_b;
    }
};

static const size_t NUM_TRIALS    = 10;
static const int    MAX_ITERATION = 1000;
static const double EPSILON       = 1.0e-8;

int matrix_dims[]={ 64, 128, 256, 512 };

template<class T, bool IS_COL_MAJOR>
string testSuitePerType (
    const bool    print_diag,
    const T       condition_num,
    const T       gen_low,
    const T       gen_high
) {
    vector< string > case_names {
        "plain c++"
    };

    vector< string > header_line {
        "matrix size",
        "64x64",
        "128x128",
        "256x256",
        "512x512"
    };

    TestResults results{ case_names, header_line };

    for( auto& dim : matrix_dims ) {

        TestExecutorConjugateGradientSolver<T, IS_COL_MAJOR> e( results, dim, condition_num, gen_low, gen_high, NUM_TRIALS, false );

        e.addTestCase( make_shared< TestCaseConjugateGradientSolver_baseline<T, IS_COL_MAJOR> > ( case_names[0], dim, MAX_ITERATION, (const T)EPSILON, condition_num ) );
        e.execute( print_diag );
    }

    std::stringstream ss;

    results.printHTML( ss );

    return ss.str();
}

#ifdef __EMSCRIPTEN__

string testConjugateGradientFloatCond10()
{
    return testSuitePerType< float, false >( true, 10.0, -1.0, 1.0 );
}

string testConjugateGradientFloatCond1000()
{
    return testSuitePerType< float, false >( true, 1000.0, -1.0, 1.0 );
}

string testConjugateGradientFloatCond100000()
{
    return testSuitePerType< float, false >( true, 100000.0, -1.0, 1.0 );
}

string testConjugateGradientDoubleCond10()
{
    return testSuitePerType< double, false >( true, 10.0, -1.0, 1.0 );
}

string testConjugateGradientDoubleCond1000()
{
    return testSuitePerType< double, false >( true, 1000.0, -1.0, 1.0 );
}

string testConjugateGradientDoubleCond100000()
{
    return testSuitePerType< double, false >( true, 100000.0, -1.0, 1.0 );
}

EMSCRIPTEN_BINDINGS( saxpy_module ) {

    emscripten::function( "testConjugateGradientFloatCond10",     &testConjugateGradientFloatCond10     );
    emscripten::function( "testConjugateGradientFloatCond1000",   &testConjugateGradientFloatCond1000   );
    emscripten::function( "testConjugateGradientFloatCond100000", &testConjugateGradientFloatCond100000 );

    emscripten::function( "testConjugateGradientDoubleCond10",    &testConjugateGradientDoubleCond10    );
    emscripten::function( "testConjugateGradientDoubleCond1000",  &testConjugateGradientDoubleCond1000  );
    emscripten::function( "testConjugateGradientDoubleCond100000",&testConjugateGradientDoubleCond100000);
}

#else

int main( int argc, char* argv[] )
{
    const bool print_diag = (argc == 2);

    cout << "conjugate gradient (float, cond num 10.0)\n\n";
    cout << testSuitePerType<float, false > ( print_diag,      10.0, -1.0, 1.0 );
    cout << "\n\n";

    cout << "conjugate gradient (float, cond num 1000.0)\n\n";
    cout << testSuitePerType<float, false > ( print_diag,    1000.0, -1.0, 1.0 );
    cout << "\n\n";

    cout << "conjugate gradient (float, cond num 100000.0)\n\n";
    cout << testSuitePerType<float, false > ( print_diag,  100000.0, -1.0, 1.0 );
    cout << "\n\n";

    cout << "conjugate gradient (double, cond num 10.0)\n\n";
    cout << testSuitePerType<double, false > ( print_diag,      10.0, -1.0, 1.0 );
    cout << "\n\n";

    cout << "conjugate gradient (double, cond num 1000.0)\n\n";
    cout << testSuitePerType<double, false > ( print_diag,    1000.0, -1.0, 1.0 );
    cout << "\n\n";

    cout << "conjugate gradient (double, cond num 100000.0)\n\n";
    cout << testSuitePerType<double, false > ( print_diag,  100000.0, -1.0, 1.0 );
    cout << "\n\n";

    return 0;
}

#endif
