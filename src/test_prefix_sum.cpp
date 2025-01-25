#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <emscripten/bind.h>
#endif

#include <functional>
#include <iostream>
#include <sstream>
#include <iterator>
#include <numeric>
#include <vector>

#include "test_case_with_time_measurements.h"
#include "test_pattern_generation.h"

namespace PrefixSum {

template<class T>
class TestCasePrefixSum : public TestCaseWithTimeMeasurements {

static constexpr double DIST_PASSING_THRESHOLD = 1.0e-6;

protected:
    const size_t m_num_elements;
    const T*     m_in;
          T*     m_out;
    double       m_dist_from_baseline;

public:

    TestCasePrefixSum( const string& case_name, const size_t num_elements )
        :TestCaseWithTimeMeasurements { case_name }
        ,m_num_elements               { num_elements }
        ,m_dist_from_baseline         { 0.0 }
    {
        ;
    }

    virtual ~TestCasePrefixSum()
    {
        ;
    }

    void calculateDistance( const T* baseline )
    {
        double sum = 0.0;

        m_out = getOut();
        for ( size_t i = 0; i < m_num_elements; i++ ) {

            sum += fabs( (double)( m_out[i] - baseline[i] ) );
        }

        const auto dist = sum / ( double )( m_num_elements );
        this->setDist( dist );

        this->setTrueFalse( (dist < DIST_PASSING_THRESHOLD) ? true : false );
    }


    virtual void setIn  ( const T* const in  ){ m_in  = in;  }
    virtual void setOut (       T* const out ){ m_out = out; }

    virtual T* getOut() { return m_out; }

    virtual void copyBackOut() { ; }

    virtual void run() = 0;
};


template<class T>
class TestCasePrefixSum_baseline : public TestCasePrefixSum<T> {

protected:
    const int m_factor_loop_unrolling;

public:
    TestCasePrefixSum_baseline( const string& case_name, const size_t num_elements, const int factor_loop_unrolling )
        :TestCasePrefixSum<T>   { case_name, num_elements }
        ,m_factor_loop_unrolling{ factor_loop_unrolling }
    {
        assert(    factor_loop_unrolling == 1
                || factor_loop_unrolling == 2
                || factor_loop_unrolling == 4
                || factor_loop_unrolling == 8 );
    }

    virtual ~TestCasePrefixSum_baseline()
    {
        ;
    }

    void run()
    {
        if ( m_factor_loop_unrolling == 1 ) {

            T sum = 0;

            for ( size_t i = 0; i < this->m_num_elements ; i++ ) {

                this->m_out[ i ] = this->m_in[ i ] + sum ;

                sum = this->m_out[ i ];
            }
        }
        else if ( m_factor_loop_unrolling == 2 ) {

            T sum = 0;
            size_t i;
            for ( i = 0; i < this->m_num_elements - 2 ; i+= 2 ) {

                this->m_out[ i     ] = this->m_in[ i ] + sum ;
                this->m_out[ i + 1 ] = this->m_in[ i ] + this->m_in[ i + 1 ] + sum ;

                sum = this->m_out[i+1];
            }

            for ( ; i < this->m_num_elements ; i++ ) {

                this->m_out[ i ] = this->m_in[ i ] + sum ;

                sum = this->m_out[ i ];
            }
        }
        else if ( m_factor_loop_unrolling == 4 ) {

            T sum = 0;
            size_t i;
            for ( i = 0; i < this->m_num_elements - 4 ; i+= 4 ) {

                this->m_out[i  ] = this->m_in[i] + sum ;
                this->m_out[i+1] = this->m_in[i] + this->m_in[i+1] + sum ;
                this->m_out[i+2] = this->m_in[i] + this->m_in[i+1] + this->m_in[i+2] + sum ;
                this->m_out[i+3] = this->m_in[i] + this->m_in[i+1] + this->m_in[i+2] + this->m_in[i+3] + sum ;

                sum = this->m_out[i+3];
            }
            for ( ; i < this->m_num_elements ; i++ ) {

                this->m_out[i] = this->m_in[i] + sum ;

                sum = this->m_out[i];
            }

        }
        else if ( m_factor_loop_unrolling == 8 ) {

            T sum = 0;
            size_t i;
            for ( i = 0; i < this->m_num_elements - 8 ; i+= 8 ) {

                this->m_out[i  ] = this->m_in[i] + sum ;
                this->m_out[i+1] = this->m_in[i] + this->m_in[i+1] + sum ;
                this->m_out[i+2] = this->m_in[i] + this->m_in[i+1] + this->m_in[i+2] + sum ;
                this->m_out[i+3] = this->m_in[i] + this->m_in[i+1] + this->m_in[i+2] + this->m_in[i+3] + sum ;
                this->m_out[i+4] = this->m_in[i] + this->m_in[i+1] + this->m_in[i+2] + this->m_in[i+3]
                                 + this->m_in[i+4] + sum ;

                this->m_out[i+5] = this->m_in[i] + this->m_in[i+1] + this->m_in[i+2] + this->m_in[i+3]
                                 + this->m_in[i+4] + this->m_in[i+5] + sum ;

                this->m_out[i+6] = this->m_in[i] + this->m_in[i+1] + this->m_in[i+2] + this->m_in[i+3]
                                 + this->m_in[i+4] + this->m_in[i+5] + this->m_in[i+6] + sum ;

                this->m_out[i+7] = this->m_in[i] + this->m_in[i+1] + this->m_in[i+2] + this->m_in[i+3]
                                 + this->m_in[i+4] + this->m_in[i+5] + this->m_in[i+6] + this->m_in[i+7] + sum ;

                sum = this->m_out[i+7];
            }
            for ( ; i < this->m_num_elements ; i++ ) {

                this->m_out[i] = this->m_in[i] + sum ;

                sum = this->m_out[i];
            }
        }
    }
};


template<class T>
class TestCasePrefixSum_stdcpp : public TestCasePrefixSum<T> {

  protected:
    vector<T> m_in_vector;
    vector<T> m_out_vector;

  public:
    TestCasePrefixSum_stdcpp( const string& case_name, const size_t num_elements )
        :TestCasePrefixSum<T>{ case_name, num_elements }
    {
        ;
    }

    virtual ~TestCasePrefixSum_stdcpp()
    {
        ;
    }

    virtual void run() {

        std::inclusive_scan( m_in_vector.begin(), m_in_vector.end(), m_out_vector.begin() );
    }

    virtual void setIn( const T* const in  )
    {
        TestCasePrefixSum<T>::setIn( in );

        m_in_vector.clear();

        for ( int i = 0; i < this->m_num_elements; i++ ) {

            m_in_vector.push_back( this->m_in[ i ] );
        }
    }

    virtual void setOut( T* const out  )
    {
        TestCasePrefixSum<T>::setOut( out );
        m_out_vector.clear();

        for ( int i = 0; i < this->m_num_elements; i++ ) {

            m_out_vector.push_back( 0.0 );
        }
    }

    virtual T* getOut()
    {
        for ( int i = 0; i < this->m_num_elements; i++ ) {

            this->m_out[ i ] = m_out_vector[ i ];
        }

        return TestCasePrefixSum<T>::getOut();
    }
};

template <class T>
class TestExecutorPrefixSum : public TestExecutor {

  protected:

    const bool            m_repeatable;
    default_random_engine m_e;
    T*                    m_in;
    T*                    m_out;
    T*                    m_out_baseline;

  public:

    TestExecutorPrefixSum( 
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
        ,m_in          { new T[num_elements] }
        ,m_out         { new T[num_elements] }
        ,m_out_baseline{ new T[num_elements] }
    {
        fillArrayWithRandomValues( m_e, m_in,  m_num_elements, min_val, max_val );
    }

    virtual ~TestExecutorPrefixSum()
    {
        delete[] m_in;
        delete[] m_out;
        delete[] m_out_baseline;
    }

    void cleanupAfterBatchRuns ( const int test_case )
    {
        auto t = dynamic_pointer_cast< TestCasePrefixSum <T> >( this->m_test_cases[ test_case ] );

        if ( test_case == 0 ) {

            memcpy( m_out_baseline , t->getOut(), sizeof(T)*m_num_elements );
        }

        t->calculateDistance( m_out_baseline ); 
    }

    void prepareForRun ( const int test_case, const int num )
    {
        auto t = dynamic_pointer_cast< TestCasePrefixSum<T> >( this->m_test_cases[ test_case ] );
        memset( m_out, 0, sizeof(T) * this->m_num_elements );
        t->setIn     ( m_in  );
        t->setOut    ( m_out );
    }
};

static const size_t NUM_TRIALS = 100;

static size_t nums_elements[]{ 32, 128, 512, 2*1024, 8* 1024, 32*1024, 128*1024, 512*1024 };

template<class T>
string testSuitePerType ( const bool print_diag )
{
    vector< string > case_names {

        "plain c++ loop unrolled order 1",
        "plain c++ loop unrolled order 2",
        "plain c++ loop unrolled order 4",
        "plain c++ loop unrolled order 8",
        "std::inclusive_scan()",
    };

    vector< string > header_line {
        "vector length",
        "32",
        "128",
        "512",
        "2K",
        "8K",
        "32K",
        "128K",
        "512K"
    };

    TestResults results{ case_names, header_line };

    for( auto num_elements : nums_elements ) {

        TestExecutorPrefixSum<T> e( results, num_elements, NUM_TRIALS, false, 0, 10 );

        e.addTestCase( make_shared< TestCasePrefixSum_baseline <T> > ( case_names[0], num_elements, 1 ) );
        e.addTestCase( make_shared< TestCasePrefixSum_baseline <T> > ( case_names[1], num_elements, 2 ) );
        e.addTestCase( make_shared< TestCasePrefixSum_baseline <T> > ( case_names[2], num_elements, 4 ) );
        e.addTestCase( make_shared< TestCasePrefixSum_baseline <T> > ( case_names[3], num_elements, 8 ) );
        e.addTestCase( make_shared< TestCasePrefixSum_stdcpp   <T> > ( case_names[4], num_elements    ) );

        e.execute( print_diag );
    }

    std::stringstream ss;

    results.printHTML( ss );

    return ss.str();
}

} // namespace PrefixSum

#ifdef __EMSCRIPTEN__

string testPrefixSum()
{
    return PrefixSum::testSuitePerType<int>( true );
}

EMSCRIPTEN_BINDINGS( prefix_sum_module ) {
    emscripten::function( "testPrefixSum", &testPrefixSum );
}

#else

int main(int argc, char* argv[])
{
    const bool print_diag = (argc == 2);

    cout << "prefix sum (int)\n\n";
    cout << PrefixSum::testSuitePerType<int>( print_diag );
    cout << "\n\n";

    return 0;
}

#endif
