#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <emscripten/bind.h>
#endif

#include <sstream>
#include "test_case_with_time_measurements.h"
#include "test_pattern_generation.h"

#include <boost/sort/spreadsort/spreadsort.hpp>
#include <boost/sort/sort.hpp>

namespace Sort {

template<class T>
class TestCaseRadixSort : public TestCaseWithTimeMeasurements {

protected:

    const size_t m_num_elements;
    T*           m_array;
    bool         m_calculation_correct;

public:

    TestCaseRadixSort( const string& case_name, const size_t num_elements )
        :TestCaseWithTimeMeasurements { case_name }
        ,m_num_elements               { num_elements }
        ,m_calculation_correct        { false }
    {
        ;
    }

    virtual ~TestCaseRadixSort()
    {
        ;
    }

    void checkResult( const T* truth )
    {
        T* out = getArray();

        setTrueFalse( true );       

        for ( size_t i = 0; i < m_num_elements; i++ ) {

            if ( out[i] != truth[i] ) {
                setTrueFalse( false );
            }
        }
    }

    virtual void setArray( T* const a ) { m_array = a; }
    virtual T*   getArray() { return m_array; }

    virtual void run() = 0;
};


template<class T>
class TestCaseRadixSort_baseline : public TestCaseRadixSort<T> {

public:
    TestCaseRadixSort_baseline( const string& case_name, const size_t num_elements )
        :TestCaseRadixSort<T>{ case_name, num_elements }
    {
        ;
    }

    virtual ~TestCaseRadixSort_baseline()
    {
        ;
    }

    void run()
    {
        std::sort( this->m_array, this->m_array + this->m_num_elements );
    }
};

template<class T>
class TestCaseRadixSort_boost_spread_sort : public TestCaseRadixSort<T> {

public:

    TestCaseRadixSort_boost_spread_sort( const string& case_name, const size_t num_elements )
        :TestCaseRadixSort<T>{ case_name, num_elements }
    {
        ;
    }

    virtual ~TestCaseRadixSort_boost_spread_sort()
    {
        ;
    }

    void run()
    {
        if constexpr ( is_same<float, T>::value || is_same<double, T>::value ) {

            boost::sort::spreadsort::float_sort ( this->m_array, this->m_array + this->m_num_elements );
        }
        else {
            boost::sort::spreadsort::integer_sort( this->m_array, this->m_array + this->m_num_elements );
        }
    }
};

template<class T>
class TestCaseRadixSort_boost_sample_sort : public TestCaseRadixSort<T> {

private:
    size_t m_num_threads;

public:
    TestCaseRadixSort_boost_sample_sort( const string& case_name, const size_t num_element, const size_t num_threads )
        :TestCaseRadixSort<T>{ case_name, num_element }
        ,m_num_threads       { num_threads }
    {
        ;
    }

    virtual ~TestCaseRadixSort_boost_sample_sort()
    {
        ;
    }

    void run()
    {
        boost::sort::block_indirect_sort( this->m_array, this->m_array + this->m_num_elements , m_num_threads );
    }
};

template <class T>
class TestExecutorRadixSort : public TestExecutor {

  protected:

    const int             m_num_elements;
    const bool            m_repeatable;
    default_random_engine m_e;
    T*                    m_array_original;
    T*                    m_array_sorted_baseline;
    T*                    m_array_sorted;

  public:

    TestExecutorRadixSort(
        TestResults& results,
        const int    num_elements,
        const int    num_trials,
        const bool   repeatable,
        const T      min_val,
        const T      max_val
    )
        :TestExecutor            { results, num_elements, num_trials }
        ,m_num_elements          { num_elements }
        ,m_repeatable            { repeatable }
        ,m_e                     { static_cast<unsigned int>( repeatable ? 0 : chrono::system_clock::now().time_since_epoch().count() ) }
        ,m_array_original        { new T[ num_elements ] }
        ,m_array_sorted_baseline { new T[ num_elements ] }
        ,m_array_sorted          { new T[ num_elements ] }
    {
        fillArrayWithRandomValues( m_e, m_array_original,  m_num_elements, min_val, max_val );
    }

    virtual ~TestExecutorRadixSort()
    {
        delete[] m_array_original;
        delete[] m_array_sorted_baseline;
        delete[] m_array_sorted;
    }

    void cleanupAfterBatchRuns ( const int test_case )
    {
        auto t = dynamic_pointer_cast< TestCaseRadixSort <T> >( this->m_test_cases[ test_case ] );

        if ( test_case == 0 ) {

            memcpy( m_array_sorted_baseline , t->getArray(), sizeof(T)*m_num_elements );
        }
        t->checkResult(  m_array_sorted_baseline );
    }

    void prepareForRun ( const int test_case, const int num )
    {
        auto t = dynamic_pointer_cast< TestCaseRadixSort<T> >( this->m_test_cases[ test_case ] );

        memcpy( m_array_sorted, m_array_original, sizeof(T) * this->m_num_elements );

        t->setArray( m_array_sorted );
    }
};

static const size_t NUM_TRIALS = 10;

static size_t nums_elements[]{ 32, 128, 512, 2*1024, 8*1024, 32*1024, 128*1024 };

template<class T>
string testSuitePerType ( const bool print_diag )
{
    vector< string > case_names {

        "std::sort()",
        "boost::sort::spreadsort()",
        "boost::sort::block_indirect_sort()"
    };

    vector< string > header_line {

        "number of elements",
        "32",
        "128",
        "512",
        "2K",
        "8K",
        "32K",
        "128K"
    };

    TestResults results{ case_names, header_line };

    for( auto num_elements : nums_elements ) {

        TestExecutorRadixSort<T> e( results, num_elements, NUM_TRIALS, false, static_cast<T>(INT_MIN), static_cast<T>(INT_MAX) );

        e.addTestCase( make_shared< TestCaseRadixSort_baseline          <T> > ( case_names[0], num_elements    ) );
        e.addTestCase( make_shared< TestCaseRadixSort_boost_spread_sort <T> > ( case_names[1], num_elements    ) );
        e.addTestCase( make_shared< TestCaseRadixSort_boost_sample_sort <T> > ( case_names[2], num_elements, 1 ) );

        e.execute( print_diag );
    }

    std::stringstream ss;

    results.printHTML( ss );

    return ss.str();
}

} // namespace Sort

#ifdef __EMSCRIPTEN__

string testSortInt()
{
    return Sort::testSuitePerType<int> ( true );
}

string testSortFloat()
{
    return Sort::testSuitePerType<float> (true );
}

EMSCRIPTEN_BINDINGS( sort_module ) {
    emscripten::function( "testSortInt", &testSortInt );
    emscripten::function( "testSortFloat", &testSortFloat );
}

#else

int main( int argc, char* argv[] )
{
    const bool print_diag = (argc == 2);

    cout << "sort (int)\n\n";
    cout << Sort::testSuitePerType<int> ( print_diag );
    cout << "\n\n";

    cout << "sort (float)\n\n";
    cout << Sort::testSuitePerType<float> ( print_diag );
    cout << "\n\n";

    return 0;
}

#endif
