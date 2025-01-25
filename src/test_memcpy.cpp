#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <emscripten/bind.h>
#endif

#include <iostream>
#include <sstream>
#include <memory>
#include "test_case_with_time_measurements.h"
#include "test_pattern_generation.h"

namespace Memcpy {

template<class T>
class TestCaseMemcpy : public TestCaseWithTimeMeasurements {

protected:

    const T*     m_in;
          T*     m_out;
    const size_t m_num_elements;

public:

    TestCaseMemcpy( const string& case_name, const size_t num_elements )
        :TestCaseWithTimeMeasurements { case_name }
        ,m_num_elements{ num_elements }
    {
        ;
    }

    virtual ~TestCaseMemcpy()
    {
        ;
    }

    virtual void compareTruth( const T* truth )
    {
        for ( size_t i = 0; i < m_num_elements; i++ ) {

            if ( m_out[ i ] != truth[ i ] ) {

                setTrueFalse( false );
            }
        }

        setTrueFalse( true );
    }

    virtual void setIn ( const T* const in  ){ m_in  = in;  }
    virtual void setOut(       T* const out ){ m_out = out; }

    virtual T*   getOut( ){ return m_out; }

    virtual void run() = 0;
};


template<class T>
class TestCaseMemcpy_baseline : public TestCaseMemcpy<T> {

public:

    TestCaseMemcpy_baseline( const string& case_name, const size_t num_elements )
        :TestCaseMemcpy<T>{ case_name, num_elements }
    {
    }

    virtual ~TestCaseMemcpy_baseline(){
        ;
    }

    void run()
    {
        for ( size_t i = 0; i < this->m_num_elements ; i++ ) {
//            if (i % 1 == 0) {
//                __builtin_prefetch (&(this->m_out[i+1]), 1, 0);
//                __builtin_prefetch (&(this->m_in [i+1]), 0, 0);
//            }
            this->m_out[i] = this->m_in[i];
        }
    }
};

template<class T>
class TestCaseMemcpy_memcpy : public TestCaseMemcpy<T> {

  public:
    TestCaseMemcpy_memcpy( const string& case_name, const size_t num_elements )
        :TestCaseMemcpy<T>{ case_name, num_elements }
    {
        ;
    }

    virtual ~TestCaseMemcpy_memcpy()
    {
        ;
    }

    void run()
    {
        memcpy( this->m_out, this->m_in, sizeof(T) * this->m_num_elements );
    }
};


template <class T>
class TestExecutorMemcpy : public TestExecutor {

  protected:

    const bool            m_repeatable;
    default_random_engine m_e;
    T*                    m_in;
    T*                    m_out;
    T*                    m_out_baseline;

  public:

    TestExecutorMemcpy(
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
        ,m_in          { new T[ num_elements ] }
        ,m_out         { new T[ num_elements ] }
        ,m_out_baseline{ new T[ num_elements ] }
    {
        fillArrayWithRandomValues( m_e, m_in, m_num_elements, min_val, max_val );

        memset( m_out,          0, m_num_elements * sizeof(T) );
        memset( m_out_baseline, 0, m_num_elements * sizeof(T) );
    }

    virtual ~TestExecutorMemcpy()
    {
        delete[] m_in;
        delete[] m_out;
        delete[] m_out_baseline;
    }

    void cleanupAfterBatchRuns ( const int test_case )
    {
        auto t = dynamic_pointer_cast< TestCaseMemcpy< T > >( this->m_test_cases[ test_case ] );

        if ( test_case == 0 ) {

            memcpy( m_out_baseline, t->getOut(), sizeof(T) * m_num_elements );
        }

        t->compareTruth( m_out_baseline );
    }

    void prepareForRun ( const int test_case, const int num ) {

        memset( m_out, 0, sizeof(T)*m_num_elements );
        auto t = dynamic_pointer_cast<TestCaseMemcpy<T>>( this->m_test_cases[ test_case ] );
        t->setIn ( m_in  );
        t->setOut( m_out );
    }
};

static const size_t NUM_TRIALS = 10;

static size_t nums_elements[] = { 256, 4*1024, 64*1024, 1024*1024, 16*1024*1024 };

template<class T>
string testSuitePerType (
    const bool print_diag
) {
    vector< string > case_names {
        "plain c++",
        "memcpy()"
    };

    vector< string > header_line {
        "length in bytes",
        "1K",
        "16K",
        "256K",
        "4M",
        "64M"
    };

    TestResults results{ case_names, header_line };

    for ( auto n : nums_elements ) {

        TestExecutorMemcpy< T > e( results, n, NUM_TRIALS, false , INT_MIN, INT_MAX );

        e.addTestCase( make_shared< TestCaseMemcpy_baseline< T > > ( case_names[ 0 ], n ) );
        e.addTestCase( make_shared< TestCaseMemcpy_memcpy  < T > > ( case_names[ 1 ], n ) );

        e.execute( print_diag );
    }

    std::stringstream ss;

    results.printHTML( ss );

    return ss.str();
}

} // namespace Memcpy

#ifdef __EMSCRIPTEN__

string testMemcpy()
{
    return Memcpy::testSuitePerType< int >( true );
}

EMSCRIPTEN_BINDINGS( memcpy_module ) {
    emscripten::function( "testMemcpy", &testMemcpy );
}

#else

int main( int argc, char* argv[] )
{
    const bool print_diag = (argc == 2);

    cout << "memcpy (bytes)\n\n";
    cout << Memcpy::testSuitePerType< int >( print_diag );
    cout << "\n\n";

    return 0;
}

#endif
