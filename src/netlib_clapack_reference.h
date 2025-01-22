#include <string>
//      CLAPACK-3.2.1/INCLUDE/f2c.h.

typedef long int integer;
typedef float real;
typedef double doublereal;
typedef long int logical;

// from CLAPACK-3.2.1/BLAS/SRC/saxpy.c.

/* Subroutine */ int saxpy_(integer *n, real *sa, real *sx, integer *incx, 
        real *sy, integer *incy)
{
    /* System generated locals */
    integer i__1;

    /* Local variables */
    integer i__, m, ix, iy, mp1;

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*     SAXPY constant times a vector plus a vector. */
/*     uses unrolled loop for increments equal to one. */
/*     jack dongarra, linpack, 3/11/78. */
/*     modified 12/3/93, array(1) declarations changed to array(*) */


/*     .. Local Scalars .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
    /* Parameter adjustments */
    --sy;
    --sx;

    /* Function Body */
    if (*n <= 0) {
        return 0;
    }
    if (*sa == 0.f) {
        return 0;
    }
    if (*incx == 1 && *incy == 1) {
        goto L20;
    }

/*        code for unequal increments or equal increments */
/*          not equal to 1 */

    ix = 1;
    iy = 1;
    if (*incx < 0) {
        ix = (-(*n) + 1) * *incx + 1;
    }
    if (*incy < 0) {
        iy = (-(*n) + 1) * *incy + 1;
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
        sy[iy] += *sa * sx[ix];
        ix += *incx;
        iy += *incy;
/* L10: */
    }
    return 0;

/*        code for both increments equal to 1 */


/*        clean-up loop */

L20:
    m = *n % 4;
    if (m == 0) {
        goto L40;
    }
    i__1 = m;
    for (i__ = 1; i__ <= i__1; ++i__) {
        sy[i__] += *sa * sx[i__];
/* L30: */
    }
    if (*n < 4) {
        return 0;
    }
L40:
    mp1 = m + 1;
    i__1 = *n;
    for (i__ = mp1; i__ <= i__1; i__ += 4) {
        sy[i__] += *sa * sx[i__];
        sy[i__ + 1] += *sa * sx[i__ + 1];
        sy[i__ + 2] += *sa * sx[i__ + 2];
        sy[i__ + 3] += *sa * sx[i__ + 3];
/* L50: */
    }
    return 0;
} /* saxpy_ */



// from CLAPACK-3.2.1/BLAS/SRC/daxpy.c.

/* Subroutine */ int daxpy_(integer *n, doublereal *da, doublereal *dx, 
        integer *incx, doublereal *dy, integer *incy)
{
    /* System generated locals */
    integer i__1;

    /* Local variables */
    integer i__, m, ix, iy, mp1;

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*     constant times a vector plus a vector. */
/*     uses unrolled loops for increments equal to one. */
/*     jack dongarra, linpack, 3/11/78. */
/*     modified 12/3/93, array(1) declarations changed to array(*) */


/*     .. Local Scalars .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
    /* Parameter adjustments */
    --dy;
    --dx;

    /* Function Body */
    if (*n <= 0) {
        return 0;
    }
    if (*da == 0.) {
        return 0;
    }
    if (*incx == 1 && *incy == 1) {
        goto L20;
    }

/*        code for unequal increments or equal increments */
/*          not equal to 1 */

    ix = 1;
    iy = 1;
    if (*incx < 0) {
        ix = (-(*n) + 1) * *incx + 1;
    }
    if (*incy < 0) {
        iy = (-(*n) + 1) * *incy + 1;
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
        dy[iy] += *da * dx[ix];
        ix += *incx;
        iy += *incy;
/* L10: */
    }
    return 0;

/*        code for both increments equal to 1 */


/*        clean-up loop */

L20:
    m = *n % 4;
    if (m == 0) {
        goto L40;
    }
    i__1 = m;
    for (i__ = 1; i__ <= i__1; ++i__) {
        dy[i__] += *da * dx[i__];
/* L30: */
    }
    if (*n < 4) {
        return 0;
    }
L40:
    mp1 = m + 1;
    i__1 = *n;
    for (i__ = mp1; i__ <= i__1; i__ += 4) {
        dy[i__] += *da * dx[i__];
        dy[i__ + 1] += *da * dx[i__ + 1];
        dy[i__ + 2] += *da * dx[i__ + 2];
        dy[i__ + 3] += *da * dx[i__ + 3];
/* L50: */
    }
    return 0;
} /* daxpy_ */


// from CLAPACK-3.2.1/BLAS/SRC/sdot.c.


doublereal sdot_(integer *n, real *sx, integer *incx, real *sy, integer *incy)
{
    /* System generated locals */
    integer i__1;
    real ret_val;

    /* Local variables */
    integer i__, m, ix, iy, mp1;
    real stemp;

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*     forms the dot product of two vectors. */
/*     uses unrolled loops for increments equal to one. */
/*     jack dongarra, linpack, 3/11/78. */
/*     modified 12/3/93, array(1) declarations changed to array(*) */


/*     .. Local Scalars .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
    /* Parameter adjustments */
    --sy;
    --sx;

    /* Function Body */
    stemp = 0.f;
    ret_val = 0.f;
    if (*n <= 0) {
        return ret_val;
    }
    if (*incx == 1 && *incy == 1) {
        goto L20;
    }

/*        code for unequal increments or equal increments */
/*          not equal to 1 */

    ix = 1;
    iy = 1;
    if (*incx < 0) {
        ix = (-(*n) + 1) * *incx + 1;
    }
    if (*incy < 0) {
        iy = (-(*n) + 1) * *incy + 1;
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
        stemp += sx[ix] * sy[iy];
        ix += *incx;
        iy += *incy;
/* L10: */
    }
    ret_val = stemp;
    return ret_val;

/*        code for both increments equal to 1 */


/*        clean-up loop */

L20:
    m = *n % 5;
    if (m == 0) {
        goto L40;
    }
    i__1 = m;
    for (i__ = 1; i__ <= i__1; ++i__) {
        stemp += sx[i__] * sy[i__];
/* L30: */
    }
    if (*n < 5) {
        goto L60;
    }
L40:
    mp1 = m + 1;
    i__1 = *n;
    for (i__ = mp1; i__ <= i__1; i__ += 5) {
        stemp = stemp + sx[i__] * sy[i__] + sx[i__ + 1] * sy[i__ + 1] + sx[
                i__ + 2] * sy[i__ + 2] + sx[i__ + 3] * sy[i__ + 3] + sx[i__ + 
                4] * sy[i__ + 4];
/* L50: */
    }
L60:
    ret_val = stemp;
    return ret_val;
} /* sdot_ */


// from CLAPACK-3.2.1/BLAS/SRC/ddot.c.

doublereal ddot_(integer *n, doublereal *dx, integer *incx, doublereal *dy, 
        integer *incy)
{
    /* System generated locals */
    integer i__1;
    doublereal ret_val;

    /* Local variables */
    integer i__, m, ix, iy, mp1;
    doublereal dtemp;

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*     forms the dot product of two vectors. */
/*     uses unrolled loops for increments equal to one. */
/*     jack dongarra, linpack, 3/11/78. */
/*     modified 12/3/93, array(1) declarations changed to array(*) */


/*     .. Local Scalars .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
    /* Parameter adjustments */
    --dy;
    --dx;

    /* Function Body */
    ret_val = 0.;
    dtemp = 0.;
    if (*n <= 0) {
        return ret_val;
    }
    if (*incx == 1 && *incy == 1) {
        goto L20;
    }

/*        code for unequal increments or equal increments */
/*          not equal to 1 */

    ix = 1;
    iy = 1;
    if (*incx < 0) {
        ix = (-(*n) + 1) * *incx + 1;
    }
    if (*incy < 0) {
        iy = (-(*n) + 1) * *incy + 1;
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
        dtemp += dx[ix] * dy[iy];
        ix += *incx;
        iy += *incy;
/* L10: */
    }
    ret_val = dtemp;
    return ret_val;

/*        code for both increments equal to 1 */


/*        clean-up loop */

L20:
    m = *n % 5;
    if (m == 0) {
        goto L40;
    }
    i__1 = m;
    for (i__ = 1; i__ <= i__1; ++i__) {
        dtemp += dx[i__] * dy[i__];
/* L30: */
    }
    if (*n < 5) {
        goto L60;
    }
L40:
    mp1 = m + 1;
    i__1 = *n;
    for (i__ = mp1; i__ <= i__1; i__ += 5) {
        dtemp = dtemp + dx[i__] * dy[i__] + dx[i__ + 1] * dy[i__ + 1] + dx[
                i__ + 2] * dy[i__ + 2] + dx[i__ + 3] * dy[i__ + 3] + dx[i__ + 
                4] * dy[i__ + 4];
/* L50: */
    }
L60:
    ret_val = dtemp;
    return ret_val;
} /* ddot_ */



/* Subroutine */ int xerbla_(char *srname, integer *info)
{
    return 0;
}


logical lsame_(char *ca, char *cb)
{
    /* System generated locals */
    logical ret_val;

    /* Local variables */
    integer inta, intb, zcode;


/*  -- LAPACK auxiliary routine (version 3.1) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  LSAME returns .TRUE. if CA is the same letter as CB regardless of */
/*  case. */

/*  Arguments */
/*  ========= */

/*  CA      (input) CHARACTER*1 */

/*  CB      (input) CHARACTER*1 */
/*          CA and CB specify the single characters to be compared. */

/* ===================================================================== */

/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */

/*     Test if the characters are equal */

    ret_val = *(unsigned char *)ca == *(unsigned char *)cb;
    if (ret_val) {
        return ret_val;
    }

/*     Now test for equivalence if both characters are alphabetic. */

    zcode = 'Z';

/*     Use 'Z' rather than 'A' so that ASCII can be detected on Prime */
/*     machines, on which ICHAR returns a value with bit 8 set. */
/*     ICHAR('A') on Prime machines returns 193 which is the same as */
/*     ICHAR('A') on an EBCDIC machine. */

    inta = *(unsigned char *)ca;
    intb = *(unsigned char *)cb;

    if (zcode == 90 || zcode == 122) {

/*        ASCII is assumed - ZCODE is the ASCII code of either lower or */
/*        upper case 'Z'. */

        if (inta >= 97 && inta <= 122) {
            inta += -32;
        }
        if (intb >= 97 && intb <= 122) {
            intb += -32;
        }

    } else if (zcode == 233 || zcode == 169) {

/*        EBCDIC is assumed - ZCODE is the EBCDIC code of either lower or */
/*        upper case 'Z'. */

        if ( (inta >= 129 && inta <= 137) || (inta >= 145 && inta <= 153) || ( inta >= 162 && inta <= 169) ) {
               
            inta += 64;
        }
        if ( (intb >= 129 && intb <= 137) || (intb >= 145 && intb <= 153) || ( intb  >= 162 && intb <= 169) ) {
               
            intb += 64;
        }

    } else if (zcode == 218 || zcode == 250) {

/*        ASCII is assumed, on Prime machines - ZCODE is the ASCII code */
/*        plus 128 of either lower or upper case 'Z'. */

        if (inta >= 225 && inta <= 250) {
            inta += -32;
        }
        if (intb >= 225 && intb <= 250) {
            intb += -32;
        }
    }
    ret_val = inta == intb;

/*     RETURN */

/*     End of LSAME */

    return ret_val;
} /* lsame_ */




/* Subroutine */ int sgemv_(char *trans, integer *m, integer *n, real *alpha, 
        real *a, integer *lda, real *x, integer *incx, real *beta, real *y, 
        integer *incy)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2;

    /* Local variables */
    integer i__, j, ix, iy, jx, jy, kx, ky, info;
    real temp;
    integer lenx, leny;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int xerbla_(char *, integer *);

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  SGEMV  performs one of the matrix-vector operations */

/*     y := alpha*A*x + beta*y,   or   y := alpha*A'*x + beta*y, */

/*  where alpha and beta are scalars, x and y are vectors and A is an */
/*  m by n matrix. */

/*  Arguments */
/*  ========== */

/*  TRANS  - CHARACTER*1. */
/*           On entry, TRANS specifies the operation to be performed as */
/*           follows: */

/*              TRANS = 'N' or 'n'   y := alpha*A*x + beta*y. */

/*              TRANS = 'T' or 't'   y := alpha*A'*x + beta*y. */

/*              TRANS = 'C' or 'c'   y := alpha*A'*x + beta*y. */

/*           Unchanged on exit. */

/*  M      - INTEGER. */
/*           On entry, M specifies the number of rows of the matrix A. */
/*           M must be at least zero. */
/*           Unchanged on exit. */

/*  N      - INTEGER. */
/*           On entry, N specifies the number of columns of the matrix A. */
/*           N must be at least zero. */
/*           Unchanged on exit. */

/*  ALPHA  - REAL            . */
/*           On entry, ALPHA specifies the scalar alpha. */
/*           Unchanged on exit. */

/*  A      - REAL             array of DIMENSION ( LDA, n ). */
/*           Before entry, the leading m by n part of the array A must */
/*           contain the matrix of coefficients. */
/*           Unchanged on exit. */

/*  LDA    - INTEGER. */
/*           On entry, LDA specifies the first dimension of A as declared */
/*           in the calling (sub) program. LDA must be at least */
/*           max( 1, m ). */
/*           Unchanged on exit. */

/*  X      - REAL             array of DIMENSION at least */
/*           ( 1 + ( n - 1 )*abs( INCX ) ) when TRANS = 'N' or 'n' */
/*           and at least */
/*           ( 1 + ( m - 1 )*abs( INCX ) ) otherwise. */
/*           Before entry, the incremented array X must contain the */
/*           vector x. */
/*           Unchanged on exit. */

/*  INCX   - INTEGER. */
/*           On entry, INCX specifies the increment for the elements of */
/*           X. INCX must not be zero. */
/*           Unchanged on exit. */

/*  BETA   - REAL            . */
/*           On entry, BETA specifies the scalar beta. When BETA is */
/*           supplied as zero then Y need not be set on input. */
/*           Unchanged on exit. */

/*  Y      - REAL             array of DIMENSION at least */
/*           ( 1 + ( m - 1 )*abs( INCY ) ) when TRANS = 'N' or 'n' */
/*           and at least */
/*           ( 1 + ( n - 1 )*abs( INCY ) ) otherwise. */
/*           Before entry with BETA non-zero, the incremented array Y */
/*           must contain the vector y. On exit, Y is overwritten by the */
/*           updated vector y. */

/*  INCY   - INTEGER. */
/*           On entry, INCY specifies the increment for the elements of */
/*           Y. INCY must not be zero. */
/*           Unchanged on exit. */


/*  Level 2 Blas routine. */

/*  -- Written on 22-October-1986. */
/*     Jack Dongarra, Argonne National Lab. */
/*     Jeremy Du Croz, Nag Central Office. */
/*     Sven Hammarling, Nag Central Office. */
/*     Richard Hanson, Sandia National Labs. */


/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */

/*     Test the input parameters. */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --x;
    --y;

    /* Function Body */
    std::string N{"N"};
    std::string T{"T"};
    std::string C{"C"};
    std::string SGEMV{"SGEMV"};

    info = 0;
    if (! lsame_( trans, const_cast<char*>(N.c_str())) && ! lsame_(trans, const_cast<char*>(T.c_str()) ) && ! lsame_(trans, const_cast<char*>(C.c_str()) )
            ) {
        info = 1;
    } else if (*m < 0) {
        info = 2;
    } else if (*n < 0) {
        info = 3;
    } else if (*lda < std::max( static_cast<integer>(1), static_cast<integer>(*m))) {
        info = 6;
    } else if (*incx == 0) {
        info = 8;
    } else if (*incy == 0) {
        info = 11;
    }
    if (info != 0) {
        xerbla_( const_cast<char*>( SGEMV.c_str() ), &info);
        return 0;
    }

/*     Quick return if possible. */

    if (*m == 0 || *n == 0 || ( *alpha == 0.f && *beta == 1.f ) ) {
        return 0;
    }

/*     Set  LENX  and  LENY, the lengths of the vectors x and y, and set */
/*     up the start points in  X  and  Y. */

    if (lsame_(trans, const_cast<char*>(N.c_str()))) {
        lenx = *n;
        leny = *m;
    } else {
        lenx = *m;
        leny = *n;
    }
    if (*incx > 0) {
        kx = 1;
    } else {
        kx = 1 - (lenx - 1) * *incx;
    }
    if (*incy > 0) {
        ky = 1;
    } else {
        ky = 1 - (leny - 1) * *incy;
    }

/*     Start the operations. In this version the elements of A are */
/*     accessed sequentially with one pass through A. */

/*     First form  y := beta*y. */

    if (*beta != 1.f) {
        if (*incy == 1) {
            if (*beta == 0.f) {
                i__1 = leny;
                for (i__ = 1; i__ <= i__1; ++i__) {
                    y[i__] = 0.f;
/* L10: */
                }
            } else {
                i__1 = leny;
                for (i__ = 1; i__ <= i__1; ++i__) {
                    y[i__] = *beta * y[i__];
/* L20: */
                }
            }
        } else {
            iy = ky;
            if (*beta == 0.f) {
                i__1 = leny;
                for (i__ = 1; i__ <= i__1; ++i__) {
                    y[iy] = 0.f;
                    iy += *incy;
/* L30: */
                }
            } else {
                i__1 = leny;
                for (i__ = 1; i__ <= i__1; ++i__) {
                    y[iy] = *beta * y[iy];
                    iy += *incy;
/* L40: */
                }
            }
        }
    }
    if (*alpha == 0.f) {
        return 0;
    }
    if (lsame_(trans, const_cast<char*>(N.c_str()))) {

/*        Form  y := alpha*A*x + y. */

        jx = kx;
        if (*incy == 1) {
            i__1 = *n;
            for (j = 1; j <= i__1; ++j) {
                if (x[jx] != 0.f) {
                    temp = *alpha * x[jx];
                    i__2 = *m;
                    for (i__ = 1; i__ <= i__2; ++i__) {
                        y[i__] += temp * a[i__ + j * a_dim1];
/* L50: */
                    }
                }
                jx += *incx;
/* L60: */
            }
        } else {
            i__1 = *n;
            for (j = 1; j <= i__1; ++j) {
                if (x[jx] != 0.f) {
                    temp = *alpha * x[jx];
                    iy = ky;
                    i__2 = *m;
                    for (i__ = 1; i__ <= i__2; ++i__) {
                        y[iy] += temp * a[i__ + j * a_dim1];
                        iy += *incy;
/* L70: */
                    }
                }
                jx += *incx;
/* L80: */
            }
        }
    } else {

/*        Form  y := alpha*A'*x + y. */

        jy = ky;
        if (*incx == 1) {
            i__1 = *n;
            for (j = 1; j <= i__1; ++j) {
                temp = 0.f;
                i__2 = *m;
                for (i__ = 1; i__ <= i__2; ++i__) {
                    temp += a[i__ + j * a_dim1] * x[i__];
/* L90: */
                }
                y[jy] += *alpha * temp;
                jy += *incy;
/* L100: */
            }
        } else {
            i__1 = *n;
            for (j = 1; j <= i__1; ++j) {
                temp = 0.f;
                ix = kx;
                i__2 = *m;
                for (i__ = 1; i__ <= i__2; ++i__) {
                    temp += a[i__ + j * a_dim1] * x[ix];
                    ix += *incx;
/* L110: */
                }
                y[jy] += *alpha * temp;
                jy += *incy;
/* L120: */
            }
        }
    }

    return 0;

/*     End of SGEMV . */

} /* sgemv_ */



/* Subroutine */ int dgemv_(char *trans, integer *m, integer *n, doublereal *
        alpha, doublereal *a, integer *lda, doublereal *x, integer *incx, 
        doublereal *beta, doublereal *y, integer *incy)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2;

    /* Local variables */
    integer i__, j, ix, iy, jx, jy, kx, ky, info;
    doublereal temp;
    integer lenx, leny;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int xerbla_(char *, integer *);

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DGEMV  performs one of the matrix-vector operations */

/*     y := alpha*A*x + beta*y,   or   y := alpha*A'*x + beta*y, */

/*  where alpha and beta are scalars, x and y are vectors and A is an */
/*  m by n matrix. */

/*  Arguments */
/*  ========== */

/*  TRANS  - CHARACTER*1. */
/*           On entry, TRANS specifies the operation to be performed as */
/*           follows: */

/*              TRANS = 'N' or 'n'   y := alpha*A*x + beta*y. */

/*              TRANS = 'T' or 't'   y := alpha*A'*x + beta*y. */

/*              TRANS = 'C' or 'c'   y := alpha*A'*x + beta*y. */

/*           Unchanged on exit. */

/*  M      - INTEGER. */
/*           On entry, M specifies the number of rows of the matrix A. */
/*           M must be at least zero. */
/*           Unchanged on exit. */

/*  N      - INTEGER. */
/*           On entry, N specifies the number of columns of the matrix A. */
/*           N must be at least zero. */
/*           Unchanged on exit. */

/*  ALPHA  - DOUBLE PRECISION. */
/*           On entry, ALPHA specifies the scalar alpha. */
/*           Unchanged on exit. */

/*  A      - DOUBLE PRECISION array of DIMENSION ( LDA, n ). */
/*           Before entry, the leading m by n part of the array A must */
/*           contain the matrix of coefficients. */
/*           Unchanged on exit. */

/*  LDA    - INTEGER. */
/*           On entry, LDA specifies the first dimension of A as declared */
/*           in the calling (sub) program. LDA must be at least */
/*           max( 1, m ). */
/*           Unchanged on exit. */

/*  X      - DOUBLE PRECISION array of DIMENSION at least */
/*           ( 1 + ( n - 1 )*abs( INCX ) ) when TRANS = 'N' or 'n' */
/*           and at least */
/*           ( 1 + ( m - 1 )*abs( INCX ) ) otherwise. */
/*           Before entry, the incremented array X must contain the */
/*           vector x. */
/*           Unchanged on exit. */

/*  INCX   - INTEGER. */
/*           On entry, INCX specifies the increment for the elements of */
/*           X. INCX must not be zero. */
/*           Unchanged on exit. */

/*  BETA   - DOUBLE PRECISION. */
/*           On entry, BETA specifies the scalar beta. When BETA is */
/*           supplied as zero then Y need not be set on input. */
/*           Unchanged on exit. */

/*  Y      - DOUBLE PRECISION array of DIMENSION at least */
/*           ( 1 + ( m - 1 )*abs( INCY ) ) when TRANS = 'N' or 'n' */
/*           and at least */
/*           ( 1 + ( n - 1 )*abs( INCY ) ) otherwise. */
/*           Before entry with BETA non-zero, the incremented array Y */
/*           must contain the vector y. On exit, Y is overwritten by the */
/*           updated vector y. */

/*  INCY   - INTEGER. */
/*           On entry, INCY specifies the increment for the elements of */
/*           Y. INCY must not be zero. */
/*           Unchanged on exit. */


/*  Level 2 Blas routine. */

/*  -- Written on 22-October-1986. */
/*     Jack Dongarra, Argonne National Lab. */
/*     Jeremy Du Croz, Nag Central Office. */
/*     Sven Hammarling, Nag Central Office. */
/*     Richard Hanson, Sandia National Labs. */


/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */

/*     Test the input parameters. */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --x;
    --y;

    /* Function Body */
    std::string N{"N"};
    std::string T{"T"};
    std::string C{"C"};
    std::string DGEMV{"DGEMV"};


    info = 0;
    if (! lsame_(trans, const_cast<char*>(N.c_str())) && ! lsame_(trans, const_cast<char*>(T.c_str()) ) && ! lsame_(trans, const_cast<char*>( C.c_str()))
            ) {
        info = 1;
    } else if (*m < 0) {
        info = 2;
    } else if (*n < 0) {
        info = 3;
    } else if (*lda < std::max( static_cast<integer>(1), static_cast<integer>(*m))) {
        info = 6;
    } else if (*incx == 0) {
        info = 8;
    } else if (*incy == 0) {
        info = 11;
    }
    if (info != 0) {
        xerbla_( const_cast<char*>( DGEMV.c_str() ), &info);
        return 0;
    }

/*     Quick return if possible. */

    if (*m == 0 || *n == 0 || ( *alpha == 0. && *beta == 1.) ) {
        return 0;
    }

/*     Set  LENX  and  LENY, the lengths of the vectors x and y, and set */
/*     up the start points in  X  and  Y. */

    if (lsame_(trans, const_cast<char*>(N.c_str()))) {
        lenx = *n;
        leny = *m;
    } else {
        lenx = *m;
        leny = *n;
    }
    if (*incx > 0) {
        kx = 1;
    } else {
        kx = 1 - (lenx - 1) * *incx;
    }
    if (*incy > 0) {
        ky = 1;
    } else {
        ky = 1 - (leny - 1) * *incy;
    }

/*     Start the operations. In this version the elements of A are */
/*     accessed sequentially with one pass through A. */

/*     First form  y := beta*y. */

    if (*beta != 1.) {
        if (*incy == 1) {
            if (*beta == 0.) {
                i__1 = leny;
                for (i__ = 1; i__ <= i__1; ++i__) {
                    y[i__] = 0.;
/* L10: */
                }
            } else {
                i__1 = leny;
                for (i__ = 1; i__ <= i__1; ++i__) {
                    y[i__] = *beta * y[i__];
/* L20: */
                }
            }
        } else {
            iy = ky;
            if (*beta == 0.) {
                i__1 = leny;
                for (i__ = 1; i__ <= i__1; ++i__) {
                    y[iy] = 0.;
                    iy += *incy;
/* L30: */
                }
            } else {
                i__1 = leny;
                for (i__ = 1; i__ <= i__1; ++i__) {
                    y[iy] = *beta * y[iy];
                    iy += *incy;
/* L40: */
                }
            }
        }
    }
    if (*alpha == 0.) {
        return 0;
    }
    if (lsame_(trans, const_cast<char*>(N.c_str()))) {

/*        Form  y := alpha*A*x + y. */

        jx = kx;
        if (*incy == 1) {
            i__1 = *n;
            for (j = 1; j <= i__1; ++j) {
                if (x[jx] != 0.) {
                    temp = *alpha * x[jx];
                    i__2 = *m;
                    for (i__ = 1; i__ <= i__2; ++i__) {
                        y[i__] += temp * a[i__ + j * a_dim1];
/* L50: */
                    }
                }
                jx += *incx;
/* L60: */
            }
        } else {
            i__1 = *n;
            for (j = 1; j <= i__1; ++j) {
                if (x[jx] != 0.) {
                    temp = *alpha * x[jx];
                    iy = ky;
                    i__2 = *m;
                    for (i__ = 1; i__ <= i__2; ++i__) {
                        y[iy] += temp * a[i__ + j * a_dim1];
                        iy += *incy;
/* L70: */
                    }
                }
                jx += *incx;
/* L80: */
            }
        }
    } else {

/*        Form  y := alpha*A'*x + y. */

        jy = ky;
        if (*incx == 1) {
            i__1 = *n;
            for (j = 1; j <= i__1; ++j) {
                temp = 0.;
                i__2 = *m;
                for (i__ = 1; i__ <= i__2; ++i__) {
                    temp += a[i__ + j * a_dim1] * x[i__];
/* L90: */
                }
                y[jy] += *alpha * temp;
                jy += *incy;
/* L100: */
            }
        } else {
            i__1 = *n;
            for (j = 1; j <= i__1; ++j) {
                temp = 0.;
                ix = kx;
                i__2 = *m;
                for (i__ = 1; i__ <= i__2; ++i__) {
                    temp += a[i__ + j * a_dim1] * x[ix];
                    ix += *incx;
/* L110: */
                }
                y[jy] += *alpha * temp;
                jy += *incy;
/* L120: */
            }
        }
    }

    return 0;

/*     End of DGEMV . */

} /* dgemv_ */
