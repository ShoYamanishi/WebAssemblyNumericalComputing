#ifndef __TEST_CASE_CHOLESKY_H__
#define __TEST_CASE_CHOLESKY_H__

#include "test_case_with_time_measurements.h"

namespace Cholesky {

template<class T, bool IS_COL_MAJOR>
class TestCaseCholesky : public TestCaseWithTimeMeasurements {

static constexpr double RMS_PASSING_THRESHOLD = 1.0e-4;

protected:
    const int m_dim;
    const int m_num_elements;
    T*        m_L;
    T*        m_x;
    T*        m_y;
    T*        m_b;

public:

    TestCaseCholesky( const string& case_name, const int dim )
        :TestCaseWithTimeMeasurements { case_name                }
        ,m_dim                        { dim                      }
        ,m_num_elements               { ( dim + 1 ) * dim / 2    }
        ,m_L                          { new T [ m_num_elements ] }
        ,m_x                          { new T [ dim ]            }
        ,m_y                          { new T [ dim ]            }
        ,m_b                          { new T [ dim ]            }
    {
        ;
    }

    virtual ~TestCaseCholesky()
    {
        delete[] m_L;
        delete[] m_x;
        delete[] m_y;
        delete[] m_b;
    }

    virtual void compareTruth( const T* const L_baseline, const T* const x_baseline )
    {
        static_assert( is_same< float,T >::value || is_same< double,T >::value );

//        auto rms_on_matrix = getRMSDiffTwoVectors( getOutputPointer_L(), L_baseline, m_num_elements );
        auto rms_on_x      = getRMSDiffTwoVectors( getOutputPointer_x(), x_baseline, m_dim          );

        T accumm = 0.0;
        T* L = getOutputPointer_L();
        T* x = getOutputPointer_x();

        for ( int i = 0; i < m_dim; i++ ) {

            T sum = 0.0;

            for (int j = 0; j < m_dim; j++ ) {

                if ( i >= j ) {
                    sum += ( L[ lower_mat_index<IS_COL_MAJOR>( i, j, m_dim ) ] * x[j] );
                }
                else {
                    sum += ( L[ lower_mat_index<IS_COL_MAJOR>( j, i, m_dim ) ] * x[j] );
                }
            }
            accumm += fabs( sum - m_b[i] );
        }
//        auto residual_diff = accumm / ( m_dim * m_dim );

        this->setRMS( rms_on_x );

        setTrueFalse( (rms_on_x < RMS_PASSING_THRESHOLD) ? true : false );
    }

    virtual void setInitialStates( T* A, T* b )
    {
        memcpy( m_L, A, sizeof(T) * m_num_elements );
        memcpy( m_b, b, sizeof(T) * m_dim );
    }

    virtual T* getOutputPointer_L()
    {
        return m_L;
    }

    virtual T* getOutputPointer_x()
    {
        return m_x;
    }

    virtual void run() = 0;
};

} // namespace Cholesky

#endif /*__TEST_CASE_CHOLESKY_H__*/
