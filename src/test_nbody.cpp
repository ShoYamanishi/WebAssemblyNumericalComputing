#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <emscripten/bind.h>
#endif

#include <type_traits>
#include <chrono>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <random>
#include <thread>
#include <vector>
#include <assert.h>
#include <arm_neon.h>

#include "test_case_with_time_measurements.h"

#include "nbody_elements.h"
#include "nbody_elements_impl.h"

template< class T>
class TestCaseNBody : public TestCaseWithTimeMeasurements {

  protected:
    const T  EPSILON = 1.0e-5;
    const T  COEFF_G = 9.8;

    const size_t m_num_elements;
    const T      m_delta_t;
    const T      m_tolerance;

  public:

    TestCaseNBody(
        const string& case_name,
        const size_t  num_elements,
        const T       delta_t,
        const T       tolerance
    )
        :TestCaseWithTimeMeasurements { case_name }
        ,m_num_elements               { num_elements }
        ,m_delta_t                    { delta_t }
        ,m_tolerance                  { tolerance }
    {
         static_type_guard_real<T>();
    }

    virtual ~TestCaseNBody()
    {
        ;
    }

    virtual void compareTruth( const NBodyElem<T>* const baseline )
    {
        for ( size_t i = 0; i < m_num_elements; i++ ) {

            if ( ! getParticleAt( i ).equalWithinTolerance( baseline[i], m_tolerance ) ) {

                this->setTrueFalse( false );
                return;
            }
        }
        this->setTrueFalse( true );
    }

    virtual void setInitialStates( const NBodyElem<T>* const aos ) = 0;
    virtual NBodyElem<T> getParticleAt( const size_t i ) = 0;
    virtual void run() = 0;
};


template<class T>
class TestCaseNBody_baselineSOA : public TestCaseNBody<T> {

  protected:

    NBodySOA<T>      m_soa;
    VelocityElem<T>* m_v_saved;

    virtual void inline bodyBodyInteraction(

        T& a0x,        T& a0y,        T& a0z,
        const T p0x,   const T p0y,   const T p0z,
        const T p1x,   const T p1y,   const T p1z,
        const T mass1,
        const T epsilon
    ) {
        const T dx = p1x - p0x;
        const T dy = p1y - p0y;
        const T dz = p1z - p0z;

        const T dist_sqr = dx*dx + dy*dy + dz*dz + epsilon; 

        T inv_dist;

        // vDSP's rsqrt. No noticeable difference in speed.
        //const int num_1 = 1;
        //vvrsqrtf( &inv_dist, &dist_sqr, &num_1 ); 

        inv_dist = 1.0 / sqrtf( dist_sqr );

        const T inv_dist_cube = inv_dist * inv_dist * inv_dist;
        const T s = mass1 * inv_dist_cube;

        a0x += (dx * s);
        a0y += (dy * s);
        a0z += (dz * s);
    }

  public:

    virtual void setInitialStates( const NBodyElem<T>* const aos )
    {
        for ( size_t i = 0; i < this->m_num_elements ; i++ ) {

            m_soa.set(aos[i], i);
            m_v_saved[i] = aos[i].m_v;
        }
    }

    virtual NBodyElem<T> getParticleAt( const size_t i )
    {
        NBodyElem<T> e;
        m_soa.get(e, i);
        return e;
    }

    TestCaseNBody_baselineSOA(
        const string& case_name,
        const size_t  num_elements, 
        const T       delta_t, 
        const T       tolerance
    )
        :TestCaseNBody< T >{ case_name, num_elements, delta_t, tolerance }
        ,m_soa             { num_elements }
        ,m_v_saved         { new VelocityElem<T>[ num_elements ] }
    {
        ;
    }

    virtual ~TestCaseNBody_baselineSOA()
    {
        delete[] m_v_saved;
    }

    virtual void run()
    {
        // reset the velocities.
        for ( size_t i = 0; i < this->m_num_elements ; i++ ) {

            NBodyElem<T> e;            
            m_soa.get( e, i );
            e.m_v = m_v_saved[i].m_v;
            m_soa.set( e, i );
        }

        memset( m_soa.m_ax, 0, sizeof(float)*this->m_num_elements ); 
        memset( m_soa.m_ay, 0, sizeof(float)*this->m_num_elements ); 
        memset( m_soa.m_az, 0, sizeof(float)*this->m_num_elements ); 

        if ( m_soa.m_p0_is_active ) { // take out 'if' out of the for loop.

            for ( size_t i = 0; i < this->m_num_elements ; i++ ) {

                for ( size_t j = 0; j < this->m_num_elements ; j++ ) {

                    // NOTES:

                    // manual loop unrolling does not make it fater.
                    // Therefore, let clang take care of optimization
                    // ex. 12.6 secs @ 32K * 32K with manual loop unrolling
                    // while 9.5 secs without.
                    if ( i != j ) {
                        bodyBodyInteraction(
                            m_soa.m_ax  [i  ], m_soa.m_ay [i  ], m_soa.m_az [i  ],
                            m_soa.m_p0x [i  ], m_soa.m_p0y[i  ], m_soa.m_p0z[i  ],
                            m_soa.m_p0x [j  ], m_soa.m_p0y[j  ], m_soa.m_p0z[j  ],
                            m_soa.m_mass[j  ], this->EPSILON                      );
                    }
                }

                m_soa.m_vx[i]   += ( m_soa.m_ax[i] * m_soa.m_mass[i] * this->COEFF_G * this->m_delta_t );
                m_soa.m_vy[i]   += ( m_soa.m_ay[i] * m_soa.m_mass[i] * this->COEFF_G * this->m_delta_t );
                m_soa.m_vz[i]   += ( m_soa.m_az[i] * m_soa.m_mass[i] * this->COEFF_G * this->m_delta_t );

                m_soa.m_p1x[i] = m_soa.m_p0x[i] + m_soa.m_vx[i] * this->m_delta_t;
                m_soa.m_p1y[i] = m_soa.m_p0y[i] + m_soa.m_vy[i] * this->m_delta_t;
                m_soa.m_p1z[i] = m_soa.m_p0z[i] + m_soa.m_vz[i] * this->m_delta_t;
            }
            // m_soa.m_p0_is_active = false; commenting out for the test cases
        }
        else {
            assert(true); // this should never be called in the test cases.

            for ( size_t i = 0; i < this->m_num_elements ; i++ ) {

                for ( size_t j = 0; j < this->m_num_elements ; j++ ) {

                    bodyBodyInteraction(
                        m_soa.m_ax  [i], m_soa.m_ay [i], m_soa.m_az [i],
                        m_soa.m_p1x [i], m_soa.m_p1y[i], m_soa.m_p1z[i],
                        m_soa.m_p1x [j], m_soa.m_p1y[j], m_soa.m_p1z[j],
                        m_soa.m_mass[j], this->EPSILON                   );
                }

                m_soa.m_vx[i] += ( m_soa.m_ax[i] * m_soa.m_mass[i] * this->COEFF_G * this->m_delta_t );
                m_soa.m_vy[i] += ( m_soa.m_ay[i] * m_soa.m_mass[i] * this->COEFF_G * this->m_delta_t );
                m_soa.m_vz[i] += ( m_soa.m_az[i] * m_soa.m_mass[i] * this->COEFF_G * this->m_delta_t );

                m_soa.m_p0x[i] = m_soa.m_p1x[i] + m_soa.m_vx[i] * this->m_delta_t;
                m_soa.m_p0y[i] = m_soa.m_p1y[i] + m_soa.m_vy[i] * this->m_delta_t;
                m_soa.m_p0z[i] = m_soa.m_p1z[i] + m_soa.m_vz[i] * this->m_delta_t;
            }
            m_soa.m_p0_is_active = true;
        }
    }
};


template<class T>
class TestCaseNBody_SOA_NEON : public TestCaseNBody_baselineSOA<T> {

protected:

    const size_t m_factor_loop_unrolling;

    inline float32x4_t sqrt_f32( float32x4_t v ) {

        float32x4_t rough    = vrsqrteq_f32( v );
        float32x4_t refined1 = vmulq_f32( vrsqrtsq_f32( vmulq_f32( rough, rough ), v ), rough );
        float32x4_t refined2 = vmulq_f32( vrsqrtsq_f32( vmulq_f32( refined1, refined1 ), v ), refined1 );
        // float32x4_t refined3 = vmulq_f32( vrsqrtsq_f32( vmulq_f32( refined2, refined2 ), v ), refined2 );
        return refined2;

    }

    virtual void inline bodyBodyInteraction_neon(

        T& a0x,       T& a0y,       T& a0z,
        const T p0x, const T p0y, const T p0z,
        const T* p1x, const T* p1y, const T* p1z,
        const T* mass1,
        const float32x4_t& qw_epsilon
    ) {
        const float32x4_t qw_p0x = { p0x, p0x, p0x, p0x };
        const float32x4_t qw_p0y = { p0y, p0y, p0y, p0y };
        const float32x4_t qw_p0z = { p0z, p0z, p0z, p0z };

        const float32x4_t qw_p1x = vld1q_f32( p1x );
        const float32x4_t qw_p1y = vld1q_f32( p1y );
        const float32x4_t qw_p1z = vld1q_f32( p1z );

        const float32x4_t qw_mass1 = vld1q_f32( mass1 );

        const float32x4_t qw_dx = vsubq_f32( qw_p1x, qw_p0x );
        const float32x4_t qw_dy = vsubq_f32( qw_p1y, qw_p0y );
        const float32x4_t qw_dz = vsubq_f32( qw_p1z, qw_p0z );

        const float32x4_t qw_dxdx = vmulq_f32( qw_dx, qw_dx );
        const float32x4_t qw_dydy = vmulq_f32( qw_dy, qw_dy );
        const float32x4_t qw_dzdz = vmulq_f32( qw_dz, qw_dz );

        const float32x4_t qw_subsum_1 = vaddq_f32( qw_dxdx, qw_dydy );
        const float32x4_t qw_subsum_2 = vaddq_f32( qw_dzdz, qw_epsilon );

        const float32x4_t qw_dist_sqr = vaddq_f32( qw_subsum_1, qw_subsum_2 );
        const float32x4_t qw_inv_dist = sqrt_f32( qw_dist_sqr );
        const float32x4_t qw_inv_dist_cube = vmulq_f32( vmulq_f32( qw_inv_dist, qw_inv_dist ), qw_inv_dist ); 
        const float32x4_t qw_s = vmulq_f32( qw_mass1, qw_inv_dist_cube );

        const float32x4_t qw_dxs = vmulq_f32( qw_dx, qw_s );
        const float32x4_t qw_dys = vmulq_f32( qw_dy, qw_s );
        const float32x4_t qw_dzs = vmulq_f32( qw_dz, qw_s );

        a0x += ( qw_dxs[0] + qw_dxs[1] + qw_dxs[2] + qw_dxs[3] );
        a0y += ( qw_dys[0] + qw_dys[1] + qw_dys[2] + qw_dys[3] );
        a0z += ( qw_dzs[0] + qw_dzs[1] + qw_dzs[2] + qw_dzs[3] );
    }

  public:

    TestCaseNBody_SOA_NEON(
        const string& case_name,
        const size_t  num_elements,
        const size_t  factor_loop_unrolling,
        const T       delta_t,
        const T       tolerance
    )
        :TestCaseNBody_baselineSOA<T>{ case_name,  num_elements, delta_t, tolerance }
        ,m_factor_loop_unrolling     { factor_loop_unrolling }
    {
        ;
    }

    virtual ~TestCaseNBody_SOA_NEON()
    {
        ;
    }

    virtual void run()
    {
        // reset the velocities.
        for ( size_t i = 0; i < this->m_num_elements ; i++ ) {

            NBodyElem<T> e;            
            this->m_soa.get( e, i );
            e.m_v = this->m_v_saved[i].m_v;
            this->m_soa.set( e, i );
        }

        memset( this->m_soa.m_ax, 0, sizeof(float)*this->m_num_elements ); 
        memset( this->m_soa.m_ay, 0, sizeof(float)*this->m_num_elements ); 
        memset( this->m_soa.m_az, 0, sizeof(float)*this->m_num_elements ); 

        calc_block( 0, this->m_num_elements );
    }

    virtual void inline bodyBodyInteractionGuarded4LanesP0IsActive( int i, int j, const float32x4_t& qw_epsilon )
    {
        if ( i < j || j+3 < i ) {
            bodyBodyInteraction_neon(
                this->m_soa.m_ax    [i  ],  this->m_soa.m_ay   [i  ],  this->m_soa.m_az   [i  ],
                this->m_soa.m_p0x   [i  ],  this->m_soa.m_p0y  [i  ],  this->m_soa.m_p0z  [i  ],
                &(this->m_soa.m_p0x [j  ]), &(this->m_soa.m_p0y[j  ]), &(this->m_soa.m_p0z[j  ]),
                &(this->m_soa.m_mass[j  ]), qw_epsilon           );
        }

        else {
            if ( i != j ) {
                this->bodyBodyInteraction(
                    this->m_soa.m_ax  [i  ], this->m_soa.m_ay [i  ], this->m_soa.m_az [i  ],
                    this->m_soa.m_p0x [i  ], this->m_soa.m_p0y[i  ], this->m_soa.m_p0z[i  ],
                    this->m_soa.m_p0x [j  ], this->m_soa.m_p0y[j  ], this->m_soa.m_p0z[j  ],
                    this->m_soa.m_mass[j  ], this->EPSILON                      );
            }
            if ( i != j+1 ) {
                this->bodyBodyInteraction(
                    this->m_soa.m_ax  [i  ], this->m_soa.m_ay [i  ], this->m_soa.m_az [i  ],
                    this->m_soa.m_p0x [i  ], this->m_soa.m_p0y[i  ], this->m_soa.m_p0z[i  ],
                    this->m_soa.m_p0x [j+1], this->m_soa.m_p0y[j+1], this->m_soa.m_p0z[j+1],
                    this->m_soa.m_mass[j+1], this->EPSILON                      );
            }
            if ( i != j+2 ) {
                this->bodyBodyInteraction(
                    this->m_soa.m_ax  [i  ], this->m_soa.m_ay [i  ], this->m_soa.m_az [i  ],
                    this->m_soa.m_p0x [i  ], this->m_soa.m_p0y[i  ], this->m_soa.m_p0z[i  ],
                    this->m_soa.m_p0x [j+2], this->m_soa.m_p0y[j+2], this->m_soa.m_p0z[j+2],
                    this->m_soa.m_mass[j+2], this->EPSILON                      );
            }
            if ( i != j+3 ) {
                this->bodyBodyInteraction(
                    this->m_soa.m_ax  [i  ], this->m_soa.m_ay [i  ], this->m_soa.m_az [i  ],
                    this->m_soa.m_p0x [i  ], this->m_soa.m_p0y[i  ], this->m_soa.m_p0z[i  ],
                    this->m_soa.m_p0x [j+3], this->m_soa.m_p0y[j+3], this->m_soa.m_p0z[j+3],
                    this->m_soa.m_mass[j+3], this->EPSILON                      );
            }
        }
    }

    virtual void inline bodyBodyInteractionGuarded4LanesP1IsActive( int i, int j, const float32x4_t& qw_epsilon ) {

        if ( i < j || j+3 < i ) {
            bodyBodyInteraction_neon(
                this->m_soa.m_ax    [i  ],  this->m_soa.m_ay   [i  ],  this->m_soa.m_az   [i  ],
                this->m_soa.m_p1x   [i  ],  this->m_soa.m_p1y  [i  ],  this->m_soa.m_p1z  [i  ],
                &(this->m_soa.m_p1x [j  ]), &(this->m_soa.m_p1y[j  ]), &(this->m_soa.m_p1z[j  ]),
                &(this->m_soa.m_mass[j  ]), qw_epsilon           );
        }
        else {
            if ( i != j ) {
                this->bodyBodyInteraction(
                    this->m_soa.m_ax  [i  ], this->m_soa.m_ay [i  ], this->m_soa.m_az [i  ],
                    this->m_soa.m_p1x [i  ], this->m_soa.m_p1y[i  ], this->m_soa.m_p1z[i  ],
                    this->m_soa.m_p1x [j  ], this->m_soa.m_p1y[j  ], this->m_soa.m_p1z[j  ],
                    this->m_soa.m_mass[j  ], this->EPSILON                      );
            }
            if ( i != j+1 ) {
                this->bodyBodyInteraction(
                    this->m_soa.m_ax  [i  ], this->m_soa.m_ay [i  ], this->m_soa.m_az [i  ],
                    this->m_soa.m_p1x [i  ], this->m_soa.m_p1y[i  ], this->m_soa.m_p1z[i  ],
                    this->m_soa.m_p1x [j+1], this->m_soa.m_p1y[j+1], this->m_soa.m_p1z[j+1],
                    this->m_soa.m_mass[j+1], this->EPSILON                      );
            }
            if ( i != j+2 ) {
                this->bodyBodyInteraction(
                    this->m_soa.m_ax  [i  ], this->m_soa.m_ay [i  ], this->m_soa.m_az [i  ],
                    this->m_soa.m_p1x [i  ], this->m_soa.m_p1y[i  ], this->m_soa.m_p1z[i  ],
                    this->m_soa.m_p1x [j+2], this->m_soa.m_p1y[j+2], this->m_soa.m_p1z[j+2],
                    this->m_soa.m_mass[j+2], this->EPSILON                      );
            }
            if ( i != j+3 ) {
                this->bodyBodyInteraction(
                    this->m_soa.m_ax  [i  ], this->m_soa.m_ay [i  ], this->m_soa.m_az [i  ],
                    this->m_soa.m_p1x [i  ], this->m_soa.m_p1y[i  ], this->m_soa.m_p1z[i  ],
                    this->m_soa.m_p1x [j+3], this->m_soa.m_p1y[j+3], this->m_soa.m_p1z[j+3],
                    this->m_soa.m_mass[j+3], this->EPSILON                      );
            }
        }
    }

    virtual void calc_block( const int elem_begin, const int elem_end_past_one )
    {
        const float32x4_t qw_epsilon{
            this->EPSILON, 
            this->EPSILON, 
            this->EPSILON, 
            this->EPSILON
        }; // used by bodyBodyInteraction_neon()

        if ( this->m_soa.m_p0_is_active ) { // take out 'if' out of the loop.

            for ( int i = elem_begin; i < elem_end_past_one ; i++ ) {

                if ( m_factor_loop_unrolling == 1 ) {

                    for ( int j = 0; j < this->m_num_elements ; j += 4 ) {

                        bodyBodyInteractionGuarded4LanesP0IsActive( i, j, qw_epsilon );
                    }
                }
                else if ( m_factor_loop_unrolling == 2 ) {

                    for ( size_t j = 0; j < this->m_num_elements ; j += 8 ) {

                        bodyBodyInteractionGuarded4LanesP0IsActive( i, j,   qw_epsilon );
                        bodyBodyInteractionGuarded4LanesP0IsActive( i, j+4, qw_epsilon );
                    }
                }
                else if ( m_factor_loop_unrolling == 4 ) {

                    for ( size_t j = 0; j < this->m_num_elements ; j += 16 ) {

                        bodyBodyInteractionGuarded4LanesP0IsActive( i, j,    qw_epsilon );
                        bodyBodyInteractionGuarded4LanesP0IsActive( i, j+ 4, qw_epsilon );
                        bodyBodyInteractionGuarded4LanesP0IsActive( i, j+ 8, qw_epsilon );
                        bodyBodyInteractionGuarded4LanesP0IsActive( i, j+12, qw_epsilon );
                    }
                }
                else if ( m_factor_loop_unrolling == 8 ) {

                    for ( size_t j = 0; j < this->m_num_elements ; j += 32 ) {

                        bodyBodyInteractionGuarded4LanesP0IsActive( i, j,    qw_epsilon );
                        bodyBodyInteractionGuarded4LanesP0IsActive( i, j+ 4, qw_epsilon );
                        bodyBodyInteractionGuarded4LanesP0IsActive( i, j+ 8, qw_epsilon );
                        bodyBodyInteractionGuarded4LanesP0IsActive( i, j+12, qw_epsilon );
                        bodyBodyInteractionGuarded4LanesP0IsActive( i, j+16, qw_epsilon );
                        bodyBodyInteractionGuarded4LanesP0IsActive( i, j+20, qw_epsilon );
                        bodyBodyInteractionGuarded4LanesP0IsActive( i, j+24, qw_epsilon );
                        bodyBodyInteractionGuarded4LanesP0IsActive( i, j+28, qw_epsilon );
                    }
                }

                this->m_soa.m_vx[i]   += ( this->m_soa.m_ax[i] * this->m_soa.m_mass[i] * this->COEFF_G * this->m_delta_t );
                this->m_soa.m_vy[i]   += ( this->m_soa.m_ay[i] * this->m_soa.m_mass[i] * this->COEFF_G * this->m_delta_t );
                this->m_soa.m_vz[i]   += ( this->m_soa.m_az[i] * this->m_soa.m_mass[i] * this->COEFF_G * this->m_delta_t );

                this->m_soa.m_p1x[i] = this->m_soa.m_p0x[i] + this->m_soa.m_vx[i] * this->m_delta_t;
                this->m_soa.m_p1y[i] = this->m_soa.m_p0y[i] + this->m_soa.m_vy[i] * this->m_delta_t;
                this->m_soa.m_p1z[i] = this->m_soa.m_p0z[i] + this->m_soa.m_vz[i] * this->m_delta_t;
            }
            // this->soa_.p0_is_active_ = false;
        }
        else {
            assert(true);// this should never be called in the test cases.

            for ( size_t i = elem_begin; i < elem_end_past_one ; i++ ) {

                if ( m_factor_loop_unrolling == 1 ) {

                    for ( size_t j = 0; j < this->m_num_elements ; j += 4 ) {

                        bodyBodyInteractionGuarded4LanesP1IsActive( i, j, qw_epsilon );
                    }
                }
                else if ( m_factor_loop_unrolling == 2 ) {

                    for ( size_t j = 0; j < this->m_num_elements ; j += 8 ) {

                        bodyBodyInteractionGuarded4LanesP1IsActive( i, j,   qw_epsilon );
                        bodyBodyInteractionGuarded4LanesP1IsActive( i, j+4, qw_epsilon );
                    }

                }
                else if ( m_factor_loop_unrolling == 4 ) {

                    for ( size_t j = 0; j < this->m_num_elements ; j += 16 ) {

                        bodyBodyInteractionGuarded4LanesP1IsActive( i, j,    qw_epsilon );
                        bodyBodyInteractionGuarded4LanesP1IsActive( i, j+ 4, qw_epsilon );
                        bodyBodyInteractionGuarded4LanesP1IsActive( i, j+ 8, qw_epsilon );
                        bodyBodyInteractionGuarded4LanesP1IsActive( i, j+12, qw_epsilon );
                    }
                }
                else if ( m_factor_loop_unrolling == 8 ) {

                    for ( size_t j = 0; j < this->m_num_elements ; j += 32 ) {

                        bodyBodyInteractionGuarded4LanesP1IsActive( i, j,    qw_epsilon );
                        bodyBodyInteractionGuarded4LanesP1IsActive( i, j+ 4, qw_epsilon );
                        bodyBodyInteractionGuarded4LanesP1IsActive( i, j+ 8, qw_epsilon );
                        bodyBodyInteractionGuarded4LanesP1IsActive( i, j+12, qw_epsilon );
                        bodyBodyInteractionGuarded4LanesP1IsActive( i, j+16, qw_epsilon );
                        bodyBodyInteractionGuarded4LanesP1IsActive( i, j+20, qw_epsilon );
                        bodyBodyInteractionGuarded4LanesP1IsActive( i, j+24, qw_epsilon );
                        bodyBodyInteractionGuarded4LanesP1IsActive( i, j+28, qw_epsilon );
                    }
                }

                this->m_soa.m_vx[i]   += ( this->m_soa.m_ax[i] * this->m_soa.m_mass[i] * this->COEFF_G * this->m_delta_t );
                this->m_soa.m_vy[i]   += ( this->m_soa.m_ay[i] * this->m_soa.m_mass[i] * this->COEFF_G * this->m_delta_t );
                this->m_soa.m_vz[i]   += ( this->m_soa.m_az[i] * this->m_soa.m_mass[i] * this->COEFF_G * this->m_delta_t );

                this->m_soa.m_p1x[i] = this->m_soa.m_p0x[i] + this->m_soa.m_vx[i] * this->m_delta_t;
                this->m_soa.m_p1y[i] = this->m_soa.m_p0y[i] + this->m_soa.m_vy[i] * this->m_delta_t;
                this->m_soa.m_p1z[i] = this->m_soa.m_p0z[i] + this->m_soa.m_vz[i] * this->m_delta_t;
            }
            this->m_soa.m_p0_is_active = true;
        }
    }
};

template<class T>
class TestCaseNBody_baselineAOS : public TestCaseNBody<T> {

    NBodyElem<T>*    m_aos;
    VelocityElem<T>* m_v_saved;
    bool             m_p0_is_active;

  public:
    virtual void inline bodyBodyInteraction_P0toP1( NBodyElem<T>& particle_i, const NBodyElem<T>& particle_j )
    {
        const T dx = particle_j.m_p0.x - particle_i.m_p0.x;
        const T dy = particle_j.m_p0.y - particle_i.m_p0.y;
        const T dz = particle_j.m_p0.z - particle_i.m_p0.z;

        const T dist_sqr = dx*dx + dy*dy + dz*dz + this->EPSILON; 

        T inv_dist;

        // vDSP's rsqrt. No noticeable difference in speed.
        // const int num_1 = 1;
        //vvrsqrtf( &inv_dist, &dist_sqr, &num_1 ); 

        inv_dist = 1.0 / sqrtf( dist_sqr);

        const T inv_dist_cube = inv_dist * inv_dist * inv_dist;
        const T s = particle_j.m_am.w * inv_dist_cube;

        particle_i.m_am.x += (dx * s);
        particle_i.m_am.y += (dy * s);
        particle_i.m_am.z += (dz * s);
    }

    virtual void inline bodyBodyInteraction_P1toP0( NBodyElem<T>& particle_i, const NBodyElem<T>& particle_j )
    {
        const T dx = particle_j.m_p1.x - particle_i.m_p1.x;
        const T dy = particle_j.m_p1.y - particle_i.m_p1.y;
        const T dz = particle_j.m_p1.z - particle_i.m_p1.z;

        const T dist_sqr = dx*dx + dy*dy + dz*dz + this->EPSILON; 

        T inv_dist;

        // vDSP's rsqrt. No noticeable difference in speed.
        // const int num_1 = 1;
        //vvrsqrtf( &inv_dist, &dist_sqr, &num_1 ); 

        inv_dist = 1.0 / sqrtf( dist_sqr);

        const T inv_dist_cube = inv_dist * inv_dist * inv_dist;
        const T s = particle_j.m_am.w * inv_dist_cube;

        particle_i.m_am.x += (dx * s);
        particle_i.m_am.y += (dy * s);
        particle_i.m_am.z += (dz * s);
    }


  public:

    TestCaseNBody_baselineAOS( const string& case_name, const size_t num_elements, const T delta_t, const T tolerance )
        :TestCaseNBody<T>{ case_name, num_elements, delta_t, tolerance }
        ,m_aos           { new NBodyElem<T>   [num_elements] }
        ,m_v_saved       { new VelocityElem<T>[num_elements] }
        ,m_p0_is_active  { true }
    {
        ;
    }

    virtual ~TestCaseNBody_baselineAOS()
    {
        delete[] m_aos;
        delete[] m_v_saved;
    }

    virtual void setInitialStates( const NBodyElem<T>* const src_array )
    {
        for ( size_t i = 0; i < this->m_num_elements ; i++ ) {

             m_aos[i]     = src_array[i];
             m_v_saved[i] = src_array[i].m_v;
        }
    }

    virtual NBodyElem<T> getParticleAt( const size_t i )
    {
        return m_aos[i];
    }

    virtual void run()
    {
        // reset the velocity after every iteration.
        for ( size_t i = 0; i < this->m_num_elements ; i++ ) {
             m_aos[i].m_v = m_v_saved[i].m_v;
        }

        if ( m_p0_is_active ) { // take out 'if' out of the loop.

            for ( size_t i = 0; i < this->m_num_elements ; i++ ) {

                auto& particle_i = m_aos[i];

                particle_i.m_am.x = 0.0;
                particle_i.m_am.y = 0.0;
                particle_i.m_am.z = 0.0;

                for ( size_t j = 0; j < this->m_num_elements ; j ++ ) {
                    if ( i != j ) {
                        const auto& particle_j = m_aos[j];
                   
                        bodyBodyInteraction_P0toP1( particle_i, particle_j );
                    }
                }

                particle_i.m_v.x += ( particle_i.m_am.x * particle_i.m_am.w * this->COEFF_G * this->m_delta_t );
                particle_i.m_v.y += ( particle_i.m_am.y * particle_i.m_am.w * this->COEFF_G * this->m_delta_t );
                particle_i.m_v.z += ( particle_i.m_am.z * particle_i.m_am.w * this->COEFF_G * this->m_delta_t );

                particle_i.m_p1.x = particle_i.m_p0.x + particle_i.m_v.x * this->m_delta_t;
                particle_i.m_p1.y = particle_i.m_p0.y + particle_i.m_v.y * this->m_delta_t;
                particle_i.m_p1.z = particle_i.m_p0.z + particle_i.m_v.z * this->m_delta_t;
            }

            // m_p0_is_active = false;
        }
        else {
            assert(true);// this should never be called in the tests.

            for ( size_t i = 0; i < this->m_num_elements ; i++ ) {

                auto& particle_i = m_aos[i];

                particle_i.m_am.x = 0.0;
                particle_i.m_am.y = 0.0;
                particle_i.m_am.z = 0.0;

                for ( size_t j = 0; j < this->m_num_elements ; j ++ ) {
                    if ( i != j ) {
                        const auto& particle_j = m_aos[j];

                        bodyBodyInteraction_P1toP0( particle_i, particle_j );
                     }
                }

                particle_i.m_v.x += ( particle_i.m_am.x * particle_i.m_am.w * this->COEFF_G * this->m_delta_t );
                particle_i.m_v.y += ( particle_i.m_am.y * particle_i.m_am.w * this->COEFF_G * this->m_delta_t );
                particle_i.m_v.z += ( particle_i.m_am.z * particle_i.m_am.w * this->COEFF_G * this->m_delta_t );

                particle_i.m_p0.x = particle_i.m_p1.x + particle_i.m_v.x * this->m_delta_t;
                particle_i.m_p0.y = particle_i.m_p1.y + particle_i.m_v.y * this->m_delta_t;
                particle_i.m_p0.z = particle_i.m_p1.z + particle_i.m_v.z * this->m_delta_t;
            }
            m_p0_is_active = true;
        }
    }
};


template <class T>
class TestExecutorNBody : public TestExecutor {

  protected:

    const bool            m_repeatable;
    default_random_engine m_e;

    NBodyElem<T>*         m_particles;
    NBodyElem<T>*         m_particles_baseline;

  public:
    TestExecutorNBody(
        TestResults& results,
        const int    num_elements,
        const int    num_trials,
        const bool   repeatable
    )
        :TestExecutor        { results, num_elements, num_trials }
        ,m_repeatable        { repeatable }
        ,m_e                 { static_cast<unsigned int>( repeatable ? 0 : chrono::system_clock::now().time_since_epoch().count() ) }
        ,m_particles         { nullptr }
        ,m_particles_baseline{ nullptr }
    {
        m_particles = new NBodyElem<T>[ num_elements ];

        for ( size_t i = 0 ; i < num_elements; i++ ) {

            auto& p = m_particles[i];
            p.setRandomInitialState( m_e );
        }

        m_particles_baseline = new NBodyElem<T>[ num_elements ];
    }

    virtual ~TestExecutorNBody()
    {
        delete[] m_particles;
        delete[] m_particles_baseline;

    }

    void cleanupAfterBatchRuns ( const int test_case )
    {
        auto t = dynamic_pointer_cast< TestCaseNBody<T> >( this->m_test_cases[ test_case ] );

        if ( test_case == 0 ) {
            for ( int j = 0; j < m_num_elements; j++ ) {

                m_particles_baseline[j] = t->getParticleAt(j);
            }
        }

        t->compareTruth( m_particles_baseline );
    }

    void prepareForRun ( const int test_case, const int num )
    {
        auto t = dynamic_pointer_cast< TestCaseNBody<T> >( this->m_test_cases[ test_case ] );
        t->setInitialStates( m_particles );
    }
};

static const size_t NUM_TRIALS = 10;
static const float  TIMESTEP   = 0.1;
static const float  TOLERANCE  = 0.01;

size_t nums_elements[]{ 32, 64, 128, 256, 512, 1024, 2*1024 };

template<class T>
string testSuitePerType ( const bool print_diag, const T delta_t, const T tolerance )
{
    vector< string > case_names {
        "plain c++ AoS",
        "plain c++ SoA",
        "NEON SoA loop unrolled order 1",
        "NEON SoA loop unrolled order 2",
        "NEON SoA loop unrolled order 4",
        "NEON SoA loop unrolled order 8"
    };

    vector< string > header_line {
        "number of bodies",
        "32",
        "64",
        "128",
        "256",
        "512",
        "1K",
        "2K"
    };

    TestResults results{ case_names, header_line };

    const int neon_num_lanes = ( is_same<float, T>::value )? 4 : 2;

    for( auto num_elements : nums_elements ) {

        TestExecutorNBody<T> e( results, num_elements, NUM_TRIALS, false );

        e.addTestCase( make_shared< TestCaseNBody_baselineAOS <T> > ( case_names[0], num_elements, delta_t, tolerance ) );
        e.addTestCase( make_shared< TestCaseNBody_baselineSOA <T> > ( case_names[1], num_elements, delta_t, tolerance ) );
        e.addTestCase( make_shared< TestCaseNBody_SOA_NEON <T> > ( case_names[2], num_elements, 1, delta_t, tolerance ) );
        e.addTestCase( make_shared< TestCaseNBody_SOA_NEON <T> > ( case_names[3], num_elements, 2, delta_t, tolerance ) );
        if ( num_elements >= 4 * neon_num_lanes ) {

            e.addTestCase( make_shared< TestCaseNBody_SOA_NEON <T> > ( case_names[4], num_elements, 4, delta_t, tolerance ) );
        }
        if ( num_elements >= 8 * neon_num_lanes ) {

            e.addTestCase( make_shared< TestCaseNBody_SOA_NEON <T> > ( case_names[5], num_elements, 8, delta_t, tolerance ) );
        }

        e.execute( print_diag );
    }

    std::stringstream ss;

    results.printHTML( ss );

    return ss.str();
}

#ifdef __EMSCRIPTEN__

string testNBody()
{
    return testSuitePerType<float> ( true, TIMESTEP, TOLERANCE );
}

EMSCRIPTEN_BINDINGS( nbody_module ) {
    emscripten::function( "testNBody", &testNBody );
}

#else

int main( int argc, char* argv[] )
{
    const bool print_diag = (argc == 2);

    cout << "N-body\n\n";
    cout << testSuitePerType<float> ( print_diag, TIMESTEP, TOLERANCE );
    cout << "\n\n";
    return 0;
}

#endif
