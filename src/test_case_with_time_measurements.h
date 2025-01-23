#ifndef __TEST_CASE_TIME_MEASUREMENTS_H__
#define __TEST_CASE_TIME_MEASUREMENTS_H__

#include <type_traits>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <random>
#include <thread>
#include <vector>
#include <map>
#include <math.h>
#include <assert.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

using namespace std;

class TestResults {

public:
    TestResults( const vector< string >& case_names, const vector< string >& header_line ) 
        :m_case_names { case_names }
        ,m_header_line{ header_line }
    {
        for ( int i = 0; i < m_case_names.size(); i++ ) {

            m_times.emplace_back();
        }
    }

    ~TestResults()
    {
        ;
    }

    void addNumElems( const int num_elems )
    {
        m_nums_elems.push_back( num_elems );
    }

    void addTime( const int index, const double time )
    {
        m_times[index].push_back( time );
    }

    void printCSV( ostream& os )
    {
        for ( int j = 0; j < m_header_line.size(); j++ ) {
            if ( j > 0 ) {
                os << ", ";
            }
            os << m_header_line[ j ];
        }

        os << "\n";

        for ( int i = 0; i < m_case_names.size(); i++ ) {

            os << m_case_names[ i ];
            for ( int j = 0; j < m_nums_elems.size(); j++ ) {

                os << ", " << m_times[ i ][ j ] * 1000.0; // milliseconds

            }
            os << "\n";
        }
    }

    void printHTML( ostream& os )
    {
        printHTMLTableHeader( os );
        os << "\n";

        for ( int i = 0; i < m_case_names.size(); i++ ) {

            printHTMLTableRow( i, os );
            os << "\n";
        }
    }

    void printHTMLTableHeader( ostream& os )
    {
        os << "<tr>";
        for ( int j = 0; j < m_header_line.size(); j++ ) {
            os << "<th>";
            os << m_header_line[ j ];
            os << "</th>";
        }
        os << "</tr>";
    }

    void printHTMLTableRow( const int i, ostream& os )
    {
        os << "<tr>";
        os << "<td>";
        os << m_case_names[i];
        os << "</td>";      

        for ( int j = 0; j < m_times[i].size(); j++ ) {

            os << "<td>";

            const auto v = m_times[ i ][ j ];
            os << std::setprecision(3);
            os << v * 1000.0;

            os << "</td>";
        }
        os << "</tr>";

    }

protected:
    vector< string >           m_case_names;
    vector< string >           m_header_line;
    vector< int >              m_nums_elems;
    vector< vector< double > > m_times;
};

template<class T>
static inline void static_type_guard()
{
    static_assert(
           is_same< short,T >::value
        || is_same< int,  T >::value 
        || is_same< long, T >::value
        || is_same< float,T >::value
        || is_same< double,T >::value  );
}

template<class T>
static inline void static_type_guard_real()
{
    static_assert( is_same< float,T >::value || is_same< double,T >::value );
}

class TestCaseWithTimeMeasurements {

  protected:
    bool           m_verification_true_false;
    double         m_verification_rms;
    double         m_verification_dist;

    string         m_case_name;
    vector<double> m_measured_times;
    double         m_mean_time;
    double         m_stddev_time;

  public:

    TestCaseWithTimeMeasurements( const string& case_name )
        :m_verification_true_false { false     }
        ,m_verification_rms        { 0.0       }
        ,m_verification_dist       { 0.0       }
        ,m_case_name               { case_name }
        ,m_mean_time               { 0.0       }
        ,m_stddev_time             { 0.0       }
    {
        ;
    }

    virtual ~TestCaseWithTimeMeasurements()
    {
        ;
    }

    string caseName() const
    {
        return m_case_name;
    }

    double meanTime() const
    {
        return m_mean_time;
    }

    double stdDev() const
    {
        return m_stddev_time;
    }

    bool VerificationTrue() const
    {
        return m_verification_true_false;
    }

    void addTime( const double microseconds )
    {
        m_measured_times.push_back( microseconds );
    }

    void setTrueFalse( const bool t )
    {
        m_verification_true_false = t;
    }

    void setRMS( const double rms )
    {
        m_verification_rms = rms;
    }

    void setDist( const double dist )
    {
        m_verification_dist = dist;
    }

    void calculateMeanStddevOfTime()
    {
        m_mean_time = 0.0;

        const double len = m_measured_times.size();

        for ( auto v : m_measured_times ) {

            m_mean_time += v;
        }

        m_mean_time /= len;

        m_stddev_time = 0.0;

        for ( auto v : m_measured_times ) {

            const double diff = v - m_mean_time;
            const double sq   = diff * diff;
            m_stddev_time += sq;
        }

        m_stddev_time /= ( len - 1 );
    }

    virtual void run() = 0;

    virtual void prologue() {}
    virtual void epilogue() {}
};


class TestExecutor {

  protected:
    vector< shared_ptr< TestCaseWithTimeMeasurements > > m_test_cases;

    TestResults& m_results;
    const int    m_num_elements;
    const int    m_num_trials;

  public:
    TestExecutor( TestResults& results, const int num_elements, const int num_trials )
        :m_results     { results      }
        ,m_num_elements{ num_elements }
        ,m_num_trials  { num_trials   }
    {
        ;
    }

    virtual ~TestExecutor()
    {
        ;
    }

    void addTestCase( shared_ptr< TestCaseWithTimeMeasurements>&& c )
    {
        m_test_cases.emplace_back( c );
    }

    virtual void prepareForBatchRuns   ( const int test_case ){;}
    virtual void cleanupAfterBatchRuns ( const int test_case ){;}
    virtual void prepareForRun         ( const int test_case, const int num ){;}
    virtual void cleanupAfterRun       ( const int test_case, const int num ){;}

    void execute( const bool print_diag )
    {
        m_results.addNumElems( m_num_elements );

        for ( int i = 0; i < m_test_cases.size(); i++ ) {

            auto test_case = m_test_cases[ i ];

            prepareForBatchRuns( i );

            for ( int j = 0; j < m_num_trials + 1; j++ ) {

                prepareForRun( i, j );

                test_case->prologue();

                auto time_begin = chrono::high_resolution_clock::now();        

                test_case->run();

                auto time_end = chrono::high_resolution_clock::now();        

                test_case->epilogue();

                cleanupAfterRun( i, j );

                chrono::duration<double> time_diff = time_end - time_begin;

                if ( j > 0 ) {

                    // discard the first run.
                    test_case->addTime( time_diff.count() );
                }

            }

            cleanupAfterBatchRuns(i);

            test_case->calculateMeanStddevOfTime();

            if ( print_diag ) {

                if ( test_case->VerificationTrue() ) {

                    cerr << "test case: ["    << test_case->caseName() << "]\t";
                    cerr << "average time: [" << test_case->meanTime() << "]\t";
                    cerr << "stddev: ["       << test_case->stdDev()   << "]\n";
                }
                else {
                    cerr << "test case: [" << test_case->caseName() << "] FAILED\n";
                }
            }
            m_results.addTime( i, test_case->meanTime() );
        }
    }
};


template<class T>
inline double getRMSDiffTwoVectors( const T* const a, const T* const b, const size_t num )
{
    static_assert( is_same<float, T>::value || is_same<double, T>::value );

    double diff_sum = 0.0;
    for ( int i = 0; i < num; i++ ) {

        double diff = a[i] - b[i];

        diff_sum += (diff*diff);
    }
    double rms = sqrt(diff_sum / ((double)num));
    return rms;
}


template<class T>
inline double getDistTwoVectors( const T* const a, const T* const b, const size_t num )
{
    static_assert( is_same<float, T>::value || is_same<double, T>::value );

    double diff_sum = 0.0;
    for ( int i = 0; i < num; i++ ) {

        double diff = a[i] - b[i];

        diff_sum += (diff*diff);
    }
    return sqrt(diff_sum);
}


template<class T>
inline bool equalWithinTolerance( const T& v1, const T& v2, const T& tolerance )
{
    const T d = fabs(v1 - v2);

    if ( d <= tolerance ) {

        return true;
    }
    else {
        return ( 2.0*d / (fabs(v1) + fabs(v2)) )  < tolerance;
    }
}

template<class T>
inline T align_up( const T v, const T align )
{
    return ( (v + align - 1) / align ) * align;
}

static inline size_t index_row_major( const size_t x, const size_t y, const size_t width )
{
    return y * width + x;
}

template< bool IS_COL_MAJOR >
static inline int linear_index_mat(const int i, const int j, const int M, const int N )
{
    if constexpr ( IS_COL_MAJOR ) {

        return j * N + i;
    }
    else {
        return i * M + j;
    }
}

// indexing lower-diagonal matrix in column-major:
//
//     0   1   2   3 <= j 
//   +---+-----------+
// 0 | 0 |           |
//   +---+---+-------+
// 1 | 1 | 4 |       |
//   +---+---+---+---+
// 2 | 2 | 5 | 7 |   |
//   +---+---+---+---+
// 3 | 3 | 6 | 8 | 9 |
//   +---+---+---+---+
// ^
// i
//
//
// indexing lower-diagonal matrix in row-major:
//
//     0   1   2   3 <= j
//   +---+-----------+
// 0 | 0 |           |
//   +---+---+-------+
// 1 | 1 | 2 |       |
//   +---+---+---+---+
// 2 | 3 | 4 | 5 |   |
//   +---+---+---+---+
// 3 | 6 | 7 | 8 | 9 |
//   +---+---+---+---+
// ^
// i

template<bool IS_COL_MAJOR>
static inline int lower_mat_index( const int i, const int j, const int dim )
{
    assert ( i >= j );

    if constexpr ( IS_COL_MAJOR ) {

        const int num_elems = ( dim + 1 ) * dim / 2;
        const int i_rev = ( dim - 1 ) - i;
        const int j_rev = ( dim - 1 ) - j;

        return num_elems - 1 - ( j_rev * ( j_rev + 1 ) /2 + i_rev );
    }
    else {
        return ( i + 1 ) * i / 2 + j;
    }
}

#endif /*__TEST_CASE_TIME_MEASUREMENTS_H__*/
