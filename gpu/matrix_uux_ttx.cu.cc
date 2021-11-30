#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include <iostream>
#include <math.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/work_sharder.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "matrix_uux_ttx.h"
using namespace tensorflow;
#define COMPLEX_CONJUGATE cconj
#define MAXIMUM max
#define MINIMUM min
#define CPUDevice Eigen::ThreadPoolDevice
#define GPUDevice Eigen::GpuDevice
#define DEFAULT_BLOCK_SIZE 32
__device__ double SQH = 0.70710676908493;
__device__ complex128 CZERO = complex128(0.0, 0.0);


__device__ double signn (double x, double y);

__device__ double signvecc (double x, double y);
template <typename T>
__device__ void sxxxxx (const double* p, double nss, T* phi);
template <typename T>
__device__ void ixxxxx (const double* p, double fmass, double nhel, double nsf, T* fi);
template <typename T>
__device__ void oxxxxx (const double* p, double fmass, double nhel, double nsf, T* fo);
template <typename T>
__device__ void vxxxxx (const double* p, double vmass, double nhel, double nsv, T* eps);
template <typename T>
__device__ void _ix_massive (const double* p, double fmass, double nsf, double nh, T* out_final);
template <typename T>
__device__ void _ix_massive_pp_nonzero (const double* p, double fmass, double nsf, double nh, int ip, int im, double pp, T* v);
template <typename T>
__device__ void _ix_massless (const double* p, double nhel, double nsf, double nh, T* out_final);
template <typename T>
__device__ void _ix_massless_sqp0p3_zero (const double* p, double nhel, T& out_final);
template <typename T>
__device__ void _ix_massless_sqp0p3_nonzero (const double* p, double nh, double sqp0p3, T& out_final);
template <typename T>
__device__ void _ix_massless_nh_one (T* chi, T* v);
template <typename T>
__device__ void _ix_massless_nh_not_one (T* chi, T* v);
template <typename T>
__device__ void _ox_massive (const double* p, double fmass, double nhel, double nsf, double nh, T* out_final);
template <typename T>
__device__ void _ox_massive_pp_zero (double fmass, double nsf, int ip, int im, T* v);
template <typename T>
__device__ void _ox_massive_pp_nonzero (const double* p, double fmass, double nsf, double nh, double pp, T* v);
template <typename T>
__device__ void _ox_massless (const double* p, double nhel, double nsf, double nh, T* out_final);
template <typename T>
__device__ void _vx_BRST_check (const double* p, double vmass, T* out_final);
template <typename T>
__device__ void _vx_BRST_check_massless (const double* p, T* out_final);
template <typename T>
__device__ void _vx_BRST_check_massive (const double* p, double vmass, T* out_final);
template <typename T>
__device__ void _vx_no_BRST_check (const double* p, double vmass, double nhel, double nsv, double hel0, double nsvahl, double pp, double pt, T* out_final);
template <typename T>
__device__ void _vx_no_BRST_check_massive (const double* p, double vmass, double nhel, double hel0, double nsvahl, double pp, double pt, T* out_final);
template <typename T>
__device__ void _vx_no_BRST_check_massive_pp_zero (double nhel, double nsvahl, T* v);
template <typename T>
__device__ void _vx_no_BRST_check_massive_pp_nonzero (const double* p, double vmass, double nhel, double hel0, double nsvahl, double pp, double pt, T* out_final);
template <typename T>
__device__ void _vx_no_BRST_check_massive_pp_nonzero_pt_nonzero (const double* p, double nhel, double hel0, double nsvahl, double pp, double pt, double emp, T* v);
template <typename T>
__device__ void _vx_no_BRST_check_massive_pp_nonzero_pt_zero (const double* p, double nhel, double nsvahl, T* v);
template <typename T>
__device__ void _vx_no_BRST_check_massless (const double* p, double nhel, double nsv, T* out_final);
template <typename T>
__device__ void _vx_no_BRST_check_massless_pt_nonzero (const double* p, double nhel, double nsv, double pp, double pt, T* v);
template <typename T>
__device__ void _vx_no_BRST_check_massless_pt_zero (const double* p, double nhel, double nsv, T* v);
template <typename T>
__device__ void FFV1_0 (T* F1, T* F2, T* V3, T COUP, T& out_final);
template <typename T>
__device__ void FFV1P0_3 (T* F1, T* F2, T COUP, double M3_, double W3_, T* V3);
template <typename T>
__global__ void matrix (const double* all_ps, const double* hel, const double* mdl_MT, const double* mdl_WT, const T* GC_11, double* out_final, const int nevents);
__device__ COMPLEX_TYPE cconj(COMPLEX_TYPE a) {
    return COMPLEX_TYPE(a.real(), -a.imag());
}

__device__ COMPLEX_TYPE operator+(const COMPLEX_TYPE& a, const COMPLEX_TYPE& b) {
    return COMPLEX_TYPE(a.real() + b.real(), a.imag() + b.imag());
}

__device__ COMPLEX_TYPE operator-(const COMPLEX_TYPE& a, const COMPLEX_TYPE& b) {
    return COMPLEX_TYPE(a.real() - b.real(), a.imag() - b.imag());
}

__device__ COMPLEX_TYPE operator*(const COMPLEX_TYPE& a, const COMPLEX_TYPE& b) {
    return COMPLEX_TYPE(a.real() * b.real() - a.imag() * b.imag(), a.imag() * b.real() + a.real() * b.imag());
}

__device__ COMPLEX_TYPE operator/(const COMPLEX_TYPE& a, const COMPLEX_TYPE& b) {
    double norm = b.real() * b.real() + b.imag() * b.imag();
    return COMPLEX_TYPE((a.real() * b.real() + a.imag() * b.imag())/norm, (a.imag() * b.real() - a.real() * b.imag())/norm);
}

__device__ COMPLEX_TYPE operator-(const COMPLEX_TYPE& a) {
    return COMPLEX_TYPE(-a.real(), -a.imag());
}

__device__ COMPLEX_TYPE operator*(const COMPLEX_TYPE& a, const double& b) {
    return COMPLEX_TYPE(a.real() * b, a.imag() * b);
}

__device__ COMPLEX_TYPE operator*(const double& a, const COMPLEX_TYPE& b) {
    return b * a;
}

__device__ COMPLEX_TYPE operator/(const COMPLEX_TYPE& a, const double& b) {
    return COMPLEX_TYPE(a.real() / b, a.imag() / b);
}



__device__ double signn (double x, double y) {
    int signn = 0;
    y >= 0 ? signn = 1 : signn = -1;
    return x * signn;
}


__device__ double signvecc (double x, double y) {
    return signn(x, y);
}

template <typename T>
__device__ void sxxxxx (const double* p, double nss, T* phi) {
        /*

    
        Defines a scalar wavefunction. Input momenta have shape (num events, 4).

    
    

    
        Parameters

    
        ----------

    
            p: tf.Tensor, scalar boson four-momenta of shape=(None,4)

    
            nss: tf.Tensor, final|initial state of shape=(), values=(+1|-1)

    
    

    
        Returns

    
        -------

    
            phi: tf.Tensor, scalar wavefunction of shape=(3,None)

    
        */
    
    T v0 = T(p[0]*nss,p[3]*nss);
    T v1 = T(p[1]*nss,p[2]*nss);
    T v = T(1.0,0.0);
    phi[0] = v0;
    phi[1] = v1;
    phi[2] = v;
    
    
}

template <typename T>
__device__ void ixxxxx (const double* p, double fmass, double nhel, double nsf, T* fi) {
        /*

    
        Defines an inflow fermion wavefunction. Input momenta have shape

    
        (num events, 4).

    
    

    
        Parameters

    
        ----------

    
            p: tf.Tensor, fermion four-momenta of shape=(None,4)

    
            fmass: tf.Tensor, fermion mass of shape=()

    
            nhel: tf.Tensor, fermion helicity of shape=()

    
            nsf: tf.Tensor, particle|anti-particle of shape=(), values=(+1|-1)

    
    

    
        Returns

    
        -------

    
            |fi>: tf.Tensor, fermion wavefunction of shape=(6,None)

    
        */
    
    T v0 = T(-p[0]*nsf,-p[3]*nsf);
    T v1 = T(-p[1]*nsf,-p[2]*nsf);
    
    
    
    double nh = nhel*nsf;
    
    const bool massive = fmass!=0;
    T v[4];
    if (massive) {
        _ix_massive(p,fmass,nsf,nh,v);
    }
    else {
        _ix_massless(p,nhel,nsf,nh,v);
    }
    
    fi[0] = v0;
    fi[1] = v1;
    for (int it1 = 0; it1 < 4; it1++) {
        fi[2 + it1] = v[it1];
    }
    
    
}

template <typename T>
__device__ void oxxxxx (const double* p, double fmass, double nhel, double nsf, T* fo) {
        /*

    
        Defines an outgoing fermion wavefunction. Input momenta have shape

    
        (num events, 4).

    
    

    
        Parameters

    
        ----------

    
            p: tf.Tensor, fermion four-momenta of shape=(None,4)

    
            fmass: tf.Tensor, fermion mass of shape=()

    
            nhel: tf.Tensor, fermion helicity of shape=()

    
            nsf: tf.Tensor, particle|anti-particle of shape=(), values=(+1|-1)

    
    

    
        Returns

    
        -------

    
             <fo|: tf.Tensor, fermion wavefunction of shape=(6,None)

    
        */
    
    T v0 = T(p[0]*nsf,p[3]*nsf);
    T v1 = T(p[1]*nsf,p[2]*nsf);
    
    double nh = nhel*nsf;
    
    const bool massive = fmass!=0;
    T v[4];
    if (massive) {
        _ox_massive(p,fmass,nhel,nsf,nh,v);
    }
    else {
        _ox_massless(p,nhel,nsf,nh,v);
    }
    
    fo[0] = v0;
    fo[1] = v1;
    for (int it1 = 0; it1 < 4; it1++) {
        fo[2 + it1] = v[it1];
    }
    
    
}

template <typename T>
__device__ void vxxxxx (const double* p, double vmass, double nhel, double nsv, T* eps) {
        /*

    
        Defines a vector wavefunction. nhel=4 is for checking BRST.

    
        Input momenta have shape (num events, 4).

    
    

    
        Parameters

    
        ----------

    
            p: tf.Tensor, vector boson four-momenta of shape=(None,4)

    
            vmass: tf.Tensor, boson mass of shape=()

    
            nhel: tf.Tensor, boson helicity of shape=(), 0 is forbidden if vmass=0.0

    
            nsv: tf.Tensor, final|initial state of shape=(), values=(+1|-1)

    
    

    
        Returns

    
        -------

    
            epsilon^{mu(v)}: tf.Tensor, vector wavefunction of shape=(6,None)

    
        */
    
    T v0 = T(p[0]*nsv,p[3]*nsv);
    T v1 = T(p[1]*nsv,p[2]*nsv);
    
    double pt2 = p[1]*p[1]+p[2]*p[2];
    double pp = MINIMUM(p[0],sqrt(pt2+p[3]*p[3]));
    double pt = MINIMUM(pp,sqrt(pt2));
    
    double hel0 = 1-abs(nhel);
    double nsvahl = nsv*abs(nhel);
    
    const bool BRST = nhel==4;
    T v[4];
    if (BRST) {
        _vx_BRST_check(p,vmass,v);
    }
    else {
        _vx_no_BRST_check(p,vmass,nhel,nsv,hel0,nsvahl,pp,pt,v);
    }
    
    eps[0] = v0;
    eps[1] = v1;
    for (int it1 = 0; it1 < 4; it1++) {
        eps[2 + it1] = v[it1];
    }
    
    
}

template <typename T>
__device__ void _ix_massive (const double* p, double fmass, double nsf, double nh, T* out_final) {
        /*

    
        Parameters

    
        ----------

    
            p: tf.Tensor, four-momenta of shape=(None,4)

    
            fmass: tf.Tensor, fermion mass of shape=()

    
            nsf: tf.Tensor, particle|anti-particle of shape=()

    
            nh: tf.Tensor, helicity times particle|anti-particle of shape=()

    
    

    
        Returns

    
        -------

    
            tf.Tensor, of shape=(None,4) and dtype DTYPECOMPLEX

    
        */
    
    double pp = MINIMUM(p[0],sqrt(p[1]*p[1]+p[2]*p[2]+p[3]*p[3]));
    int ip = (int)((int)(1+nh)/2);
    int im = (int)((int)(1-nh)/2);
    
    const bool cond = pp==0;
    if (cond) {
        _ox_massive_pp_zero(fmass,nsf,im,ip,out_final);
    }
    else {
        _ix_massive_pp_nonzero(p,fmass,nsf,nh,ip,im,pp,out_final);
    }
    
}

template <typename T>
__device__ void _ix_massive_pp_nonzero (const double* p, double fmass, double nsf, double nh, int ip, int im, double pp, T* v) {
        /*

    
        Parameters

    
        ----------

    
            p: tf.Tensor, four-momenta of shape=(None,4)

    
            fmass: tf.Tensor, fermion mass of shape=()

    
            nsf: tf.Tensor, particle|anti-particle of shape=()

    
            nh: tf.Tensor, helicity times particle|anti-particle of shape=()

    
            ip: tf.Tensor, positive nh projector of shape=() and dtype DTYPEINT

    
            im: tf.Tensor, negative nh projector of shape=() and dtype DTYPEINT

    
            pp: tf.Tensor, minimum of energy|three-momentum modulus of shape=(None)

    
    

    
        Returns

    
        -------

    
            tf.Tensor, of shape=(None,4) and dtype DTYPECOMPLEX

    
        */
    
    double sf[] = {(1+nsf+(1-nsf)*nh)*0.5,(1+nsf-(1-nsf)*nh)*0.5};//array of size 2
    double omega[] = {sqrt(p[0]+pp),fmass/(sqrt(p[0]+pp))};//array of size 2
    double sfomeg[] = {sf[0]*omega[ip],sf[1]*omega[im]};//array of size 2
    double pp3 = MAXIMUM(pp+p[3],0.0);
    T chi1;
    if (pp3==0) {
        chi1=T(-nh,0);
    }
    else {
        chi1=T(nh*p[1]/sqrt(2.0*pp*pp3),p[2]/sqrt(2.0*pp*pp3));
    }
    
    T chi2 = T(sqrt(pp3*0.5/pp),0.0);
    T chi[] = {chi2,chi1};//array of size 2
    for (int it1 = 0; it1 < 4; it1++) {
        v[it1] = T(0,0);
    }
    
    v[0] = T(sfomeg[0],0.0)*chi[im];
    v[1] = T(sfomeg[0],0.0)*chi[ip];
    v[2] = T(sfomeg[1],0.0)*chi[im];
    v[3] = T(sfomeg[1],0.0)*chi[ip];
    
}

template <typename T>
__device__ void _ix_massless (const double* p, double nhel, double nsf, double nh, T* out_final) {
        /*

    
        Parameters

    
        ----------

    
            p: tf.Tensor, four-momenta of shape=(None,4)

    
            nhel: tf.Tensor, fermion helicity of shape=()

    
            nsf: tf.Tensor, particle|anti-particle of shape=()

    
            nh: tf.Tensor, helicity times particle|anti-particle of shape=()

    
    

    
        Returns

    
        -------

    
            tf.Tensor, of shape=(None,4) and dtype DTYPECOMPLEX

    
        */
    
    double sqp0p3 = sqrt(MAXIMUM(p[0]+p[3],0.0))*nsf;
    T chi1;
    if (sqp0p3==0) {
        _ix_massless_sqp0p3_zero(p,nhel,chi1);
    }
    else {
        _ix_massless_sqp0p3_nonzero(p,nh,sqp0p3,chi1);
    }
    
    T chi[] = {T(sqp0p3,0.0),chi1};//array of size 2
    if (nh==1) {
        _ix_massless_nh_one(chi,out_final);
    }
    else {
        _ix_massless_nh_not_one(chi,out_final);
    }
    
}

template <typename T>
__device__ void _ix_massless_sqp0p3_zero (const double* p, double nhel, T& out_final) {
        /*

    
        Parameters

    
        ----------

    
            p: tf.Tensor, four-momenta of shape=(None,4)

    
            nhel: tf.Tensor, fermion helicity of shape=()

    
    

    
        Returns

    
        -------

    
            tf.Tensor, of shape=(None) and dtype DTYPECOMPLEX

    
    

    
        Note: this function is the same for input `ixxxxx` and output `oxxxxx`

    
        waveforms

    
        */
    
    out_final = T(-nhel*sqrt(2.0*p[0]),0.0);
}

template <typename T>
__device__ void _ix_massless_sqp0p3_nonzero (const double* p, double nh, double sqp0p3, T& out_final) {
        /*

    
        Parameters

    
        ----------

    
            p: tf.Tensor, four-momenta of shape=(None,4)

    
            nh: tf.Tensor, helicity times particle|anti-particle of shape=()

    
            sqp0p3: tf.Tensor, max(E+pz,0)*nsf of shape=(None)

    
    

    
        Returns

    
        -------

    
            tf.Tensor, of shape=(None) and dtype DTYPECOMPLEX

    
        */
    
    out_final = T(nh*p[1]/sqp0p3,p[2]/sqp0p3);
}

template <typename T>
__device__ void _ix_massless_nh_one (T* chi, T* v) {
        /*

    
        Parameters

    
        ----------

    
            chi: tf.Tensor, of shape=(None,2) and dtype DTYPECOMPLEX

    
    

    
        Returns

    
        -------

    
            tf.Tensor, of shape=(None,4) and dtype DTYPECOMPLEX

    
        */
    
    for (int it1 = 0; it1 < 4; it1++) {
        v[it1] = T(0,0);
    }
    
    v[2] = chi[0];
    v[3] = chi[1];
    v[0] = CZERO;
    v[1] = CZERO;
    
}

template <typename T>
__device__ void _ix_massless_nh_not_one (T* chi, T* v) {
        /*

    
        Parameters

    
        ----------

    
            chi: tf.Tensor, of shape=(None,2) and dtype DTYPECOMPLEX

    
    

    
        Returns

    
        -------

    
            tf.Tensor, of shape=(None,4) and dtype DTYPECOMPLEX

    
        */
    
    for (int it1 = 0; it1 < 4; it1++) {
        v[it1] = T(0,0);
    }
    
    v[0] = chi[1];
    v[1] = chi[0];
    v[2] = CZERO;
    v[3] = CZERO;
    
}

template <typename T>
__device__ void _ox_massive (const double* p, double fmass, double nhel, double nsf, double nh, T* out_final) {
        /*

    
        Parameters

    
        ----------

    
            p: tf.Tensor, four-momenta of shape=(None,4)

    
            fmass: tf.Tensor, fermion mass of shape=()

    
            nhel: tf.Tensor, fermion helicity of shape=()

    
            nsf: tf.Tensor, particle|anti-particle of shape=()

    
            nh: tf.Tensor, helicity times particle|anti-particle of shape=()

    
    

    
        Returns

    
        -------

    
            tf.Tensor, of shape=(None,4) and dtype DTYPECOMPLEX

    
        */
    
    double pp = MINIMUM(p[0],sqrt(p[1]*p[1]+p[2]*p[2]+p[3]*p[3]));
    int ip = (int)(-((int)(1-nh)/2)*nhel);
    int im = (int)((int)(1+nh)/2*nhel);
    
    const bool cond = pp==0;
    if (cond) {
        _ox_massive_pp_zero(fmass,nsf,ip,im,out_final);
    }
    else {
        _ox_massive_pp_nonzero(p,fmass,nsf,nh,pp,out_final);
    }
    
}

template <typename T>
__device__ void _ox_massive_pp_zero (double fmass, double nsf, int ip, int im, T* v) {
        /*

    
        Parameters

    
        ----------

    
            fmass: tf.Tensor, fermion mass of shape=()

    
            nsf: tf.Tensor, particle|anti-particle of shape=()

    
            ip: tf.Tensor, positive nh projector of shape=() and dtype DTYPEINT

    
            im: tf.Tensor, negative nh projector of shape=() and dtype DTYPEINT

    
    

    
        Returns

    
        -------

    
            tf.Tensor, of shape=(1,4) and dtype DTYPECOMPLEX

    
        */
    
    double sqm_ = sqrt(abs(fmass));
    double sqm[] = {sqm_,signn(sqm_,fmass)};//array of size 2
    
    for (int it1 = 0; it1 < 4; it1++) {
        v[it1] = T(0,0);
    }
    
    v[0] = T(im*sqm[abs(im)],0.0);
    v[1] = T(ip*nsf*sqm[abs(im)],0.0);
    v[2] = T(im*nsf*sqm[abs(ip)],0.0);
    v[3] = T(ip*sqm[abs(ip)],0.0);
    
}

template <typename T>
__device__ void _ox_massive_pp_nonzero (const double* p, double fmass, double nsf, double nh, double pp, T* v) {
        /*

    
        Parameters

    
        ----------

    
            p: tf.Tensor, four-momenta of shape=(None,4)

    
            fmass: tf.Tensor, fermion mass of shape=()

    
            nsf: tf.Tensor, particle|anti-particle of shape=()

    
            nh: tf.Tensor, helicity times particle|anti-particle of shape=()

    
            pp: tf.Tensor, minimum of energy|three-momentum modulus of shape=(None)

    
    

    
        Returns

    
        -------

    
            tf.Tensor, of shape=(None,4) and dtype DTYPECOMPLEX

    
        */
    
    double sf[] = {(1+nsf+(1-nsf)*nh)*0.5,(1+nsf-(1-nsf)*nh)*0.5};//array of size 2
    double omega[] = {sqrt(p[0]+pp),fmass/(sqrt(p[0]+pp))};//array of size 2
    int ip = (int)((int)(1+nh)/2);
    int im = (int)((int)(1-nh)/2);
    double sfomeg[] = {sf[0]*omega[ip],sf[1]*omega[im]};//array of size 2
    double pp3 = MAXIMUM(pp+p[3],0.0);
    T chi1;
    if (pp3==0) {
        chi1=T(-nh,0);
    }
    else {
        chi1=T(nh*p[1]/sqrt(2.0*pp*pp3),-p[2]/sqrt(2.0*pp*pp3));
    }
    
    T chi2 = T(sqrt(pp3*0.5/pp),0.0);
    T chi[] = {chi2,chi1};//array of size 2
    for (int it1 = 0; it1 < 4; it1++) {
        v[it1] = T(0,0);
    }
    
    v[0] = T(sfomeg[1],0.0)*chi[im];
    v[1] = T(sfomeg[1],0.0)*chi[ip];
    v[2] = T(sfomeg[0],0.0)*chi[im];
    v[3] = T(sfomeg[0],0.0)*chi[ip];
    
}

template <typename T>
__device__ void _ox_massless (const double* p, double nhel, double nsf, double nh, T* out_final) {
        /*

    
        Parameters

    
        ----------

    
            p: tf.Tensor, four-momenta of shape=(None,4)

    
            nhel: tf.Tensor, fermion helicity of shape=()

    
            nsf: tf.Tensor, particle|anti-particle of shape=()

    
            nh: tf.Tensor, helicity times particle|anti-particle of shape=()

    
    

    
        Returns

    
        -------

    
            tf.Tensor, of shape=(None,4) and dtype DTYPECOMPLEX

    
        */
    
    double sqp0p3 = sqrt(MAXIMUM(p[0]+p[3],0.0))*nsf;
    double mult[] = {1,1,-1,1};//array of size 4
    T chi0;
    double _p[4];
    for (int it1 = 0; it1 <4; it1++) {
        _p[it1] = p[it1] *mult[it1];
    }
    if (sqp0p3==0) {
        _ix_massless_sqp0p3_zero(p,nhel,chi0);
    }
    else {
        _ix_massless_sqp0p3_nonzero(_p,nh,sqp0p3,chi0);
    }
    
    T chi[] = {chi0,T(sqp0p3,0.0)};//array of size 2
    // ongoing fermion has nh inverted wrt the ingoing fermion

    if (nh==1) {
        _ix_massless_nh_not_one(chi,out_final);
    }
    else {
        _ix_massless_nh_one(chi,out_final);
    }
    
}

template <typename T>
__device__ void _vx_BRST_check (const double* p, double vmass, T* out_final) {
        /*

    
        Parameters

    
        ----------

    
            p: tf.Tensor, four-momenta of shape=(None,4)

    
            vmass: tf.Tensor, boson mass of shape=()

    
    

    
        Returns

    
        -------

    
            tf.Tensor, of shape=(None,4) and dtype DTYPECOMPLEX

    
        */
    
    const bool massless = vmass==0;
    if (massless) {
        _vx_BRST_check_massless(p,out_final);
    }
    else {
        _vx_BRST_check_massive(p,vmass,out_final);
    }
    
}

template <typename T>
__device__ void _vx_BRST_check_massless (const double* p, T* out_final) {
        /*

    
        Parameters

    
        ----------

    
            p: tf.Tensor, four-momenta of shape=(None,4)

    
    

    
        Returns

    
        -------

    
            tf.Tensor, of shape=(None,4) and dtype DTYPECOMPLEX

    
        */
    
    for (int it1 = 0; it1 <4; it1++) {
        out_final[it1] = T(p[it1]/p[1]);
    }
    
}

template <typename T>
__device__ void _vx_BRST_check_massive (const double* p, double vmass, T* out_final) {
        /*

    
        Parameters

    
        ----------

    
            p: tf.Tensor, four-momenta of shape=(None,4)

    
            vmass: tf.Tensor, boson mass of shape=()

    
    

    
        Returns

    
        -------

    
            tf.Tensor, of shape=(None,4) and dtype DTYPECOMPLEX

    
        */
    
    for (int it1 = 0; it1 <4; it1++) {
        out_final[it1] = T(p[it1]/vmass);
    }
    
}

template <typename T>
__device__ void _vx_no_BRST_check (const double* p, double vmass, double nhel, double nsv, double hel0, double nsvahl, double pp, double pt, T* out_final) {
        /*

    
        Parameters

    
        ----------

    
            p: tf.Tensor, four-momenta of shape=(None,4)

    
            vmass: tf.Tensor, boson mass of shape=()

    
            nhel: tf.Tensor, boson helicity of shape=()

    
            nsv: tf.Tensor, final|initial state of shape=()

    
            hel0: tf.Tensor, zero helicity of shape=()

    
            nsvahl: tf.Tensor, helicity times particle|anti-particle absolute value

    
                    of shape=()

    
            pp: tf.Tensor, minimum of energy|three-momentum modulus of shape=(None)

    
            pt: tf.Tensor, of shape=(None)

    
    

    
        Returns

    
        -------

    
            tf.Tensor, of shape=(None,4) and dtype DTYPECOMPLEX

    
        */
    
    const bool massive = vmass!=0;
    if (massive) {
        _vx_no_BRST_check_massive(p,vmass,nhel,hel0,nsvahl,pp,pt,out_final);
    }
    else {
        _vx_no_BRST_check_massless(p,nhel,nsv,out_final);
    }
    
}

template <typename T>
__device__ void _vx_no_BRST_check_massive (const double* p, double vmass, double nhel, double hel0, double nsvahl, double pp, double pt, T* out_final) {
        /*

    
        Parameters

    
        ----------

    
            p: tf.Tensor, four-momenta of shape=(None,4)

    
            vmass: tf.Tensor, boson mass of shape=()

    
            nhel: tf.Tensor, boson helicity of shape=()

    
            hel0: tf.Tensor, zero helicity of shape=()

    
            nsvahl: tf.Tensor, helicity times particle|anti-particle absolute value

    
                    of shape=()

    
            pp: tf.Tensor, minimum of energy|three-momentum modulus of shape=(None)

    
            pt: tf.Tensor, of shape=(None)

    
    

    
        Returns

    
        -------

    
            tf.Tensor, of shape=(None,4) and dtype DTYPECOMPLEX

    
        */
    
    const bool cond = pp==0;
    if (cond) {
        _vx_no_BRST_check_massive_pp_zero(nhel,nsvahl,out_final);
    }
    else {
        _vx_no_BRST_check_massive_pp_nonzero(p,vmass,nhel,hel0,nsvahl,pp,pt,out_final);
    }
    
}

template <typename T>
__device__ void _vx_no_BRST_check_massive_pp_zero (double nhel, double nsvahl, T* v) {
        /*

    
        Parameters

    
        ----------

    
            nhel: tf.Tensor, boson helicity of shape=()

    
            nsvahl: tf.Tensor, helicity times particle|anti-particle absolute value

    
                    of shape=()

    
    

    
        Returns

    
        -------

    
            tf.Tensor, of shape=(None,4) and dtype DTYPECOMPLEX

    
        */
    
    double hel0 = 1.0-abs(nhel);
    for (int it1 = 0; it1 < 4; it1++) {
        v[it1] = T(1,0);
    }
    
    v[1] = T(-nhel * SQH, 0.0);
    
    v[2] = T(0.0, nsvahl * SQH);
    
    v[3] = T(hel0, 0.0);
    
    
}

template <typename T>
__device__ void _vx_no_BRST_check_massive_pp_nonzero (const double* p, double vmass, double nhel, double hel0, double nsvahl, double pp, double pt, T* out_final) {
        /*

    
        Parameters

    
        ----------

    
            p: tf.Tensor, four-momenta of shape=(None,4)

    
            vmass: tf.Tensor, boson mass of shape=()

    
            nhel: tf.Tensor, boson helicity of shape=()

    
            hel0: tf.Tensor, zero helicity of shape=()

    
            nsvahl: tf.Tensor, helicity times particle|anti-particle absolute value

    
                    of shape=()

    
            pp: tf.Tensor, minimum of energy|three-momentum modulus of shape=(None)

    
            pt: tf.Tensor, of shape=(None)

    
    

    
        Returns

    
        -------

    
            tf.Tensor, of shape=(None,4) and dtype DTYPECOMPLEX

    
        */
    
    double emp = p[0]/(vmass*pp);
    T v2 = T(hel0*pp/vmass,0.0);
    T v5 = T(hel0*p[3]*emp+nhel*pt/pp*SQH,0);
    const bool condition = pt!=0;
    T v34[2];
    if (condition) {
        _vx_no_BRST_check_massive_pp_nonzero_pt_nonzero(p,nhel,hel0,nsvahl,pp,pt,emp,v34);
    }
    else {
        _vx_no_BRST_check_massive_pp_nonzero_pt_zero(p,nhel,nsvahl,v34);
    }
    
    out_final[0] = v2;
    for (int it1 = 0; it1 < 2; it1++) {
        out_final[1 + it1] = v34[it1];
    }
    out_final[3] = v5;
    
}

template <typename T>
__device__ void _vx_no_BRST_check_massive_pp_nonzero_pt_nonzero (const double* p, double nhel, double hel0, double nsvahl, double pp, double pt, double emp, T* v) {
        /*

    
        Parameters

    
        ----------

    
            p: tf.Tensor, four-momenta of shape=(None,4)

    
            hel0: tf.Tensor, zero helicity of shape=()

    
            nsvahl: tf.Tensor, helicity times particle|anti-particle absolute value

    
                    of shape=()

    
            pp: tf.Tensor, minimum of energy|three-momentum modulus of shape=(None)

    
            pt: tf.Tensor, of shape=(None)

    
            emp: tf.Tensor, of shape=(None)

    
    

    
        Returns

    
        -------

    
            tf.Tensor, of shape=(None,2) and dtype DTYPECOMPLEX

    
        */
    
    for (int it1 = 0; it1 < 2; it1++) {
        v[it1] = T(0,0);
    }
    
    double pzpt = p[3]/(pp*pt)*SQH*nhel;
    v[0] = T(hel0*p[1]*emp-p[1]*pzpt,-nsvahl*p[2]/pt*SQH);
    v[1] = T(hel0*p[2]*emp-p[2]*pzpt,nsvahl*p[1]/pt*SQH);
    
}

template <typename T>
__device__ void _vx_no_BRST_check_massive_pp_nonzero_pt_zero (const double* p, double nhel, double nsvahl, T* v) {
        /*

    
        Parameters

    
        ----------

    
            p: tf.Tensor, four-momenta of shape=(None,4)

    
            nhel: tf.Tensor, boson helicity of shape=()

    
            nsvahl: tf.Tensor, helicity times particle|anti-particle absolute value

    
                    of shape=()

    
    

    
        Returns

    
        -------

    
            tf.Tensor, of shape=(None,2) and dtype DTYPECOMPLEX

    
        */
    
    for (int it1 = 0; it1 < 2; it1++) {
        v[it1] = T(0,0);
    }
    
    v[0] = T(-nhel*SQH,0.0);
    v[1] = T(0.0,nsvahl*signvecc(SQH,p[3]));
    
}

template <typename T>
__device__ void _vx_no_BRST_check_massless (const double* p, double nhel, double nsv, T* out_final) {
        /*

    
        Parameters

    
        ----------

    
            p: tf.Tensor, four-momenta of shape=(None,4)

    
            nhel: tf.Tensor, boson helicity of shape=()

    
            nsv: tf.Tensor, final|initial state of shape=()

    
    

    
        Returns

    
        -------

    
            tf.Tensor, of shape=(None,4) and dtype DTYPECOMPLEX

    
        */
    
    double pp = p[0];
    double pt = sqrt(p[1]*p[1]+p[2]*p[2]);
    T v2 = T(0,0);
    T v5 = T(nhel*pt/pp*SQH,0.0);
    const bool cond = pt!=0;
    T v34[2];
    if (cond) {
        _vx_no_BRST_check_massless_pt_nonzero(p,nhel,nsv,pp,pt,v34);
    }
    else {
        _vx_no_BRST_check_massless_pt_zero(p,nhel,nsv,v34);
    }
    
    out_final[0] = v2;
    for (int it1 = 0; it1 < 2; it1++) {
        out_final[1 + it1] = v34[it1];
    }
    out_final[3] = v5;
    
}

template <typename T>
__device__ void _vx_no_BRST_check_massless_pt_nonzero (const double* p, double nhel, double nsv, double pp, double pt, T* v) {
        /*

    
        Parameters

    
        ----------

    
            p: tf.Tensor, four-momenta of shape=(None,4)

    
            nhel: tf.Tensor, boson helicity of shape=()

    
            nsv: tf.Tensor, final|initial state of shape=()

    
            SQH: tf.Tensor, sqrt(1/2) of shape=()

    
            pp: tf.Tensor, minimum of energy|three-momentum modulus of shape=(None)

    
            pt: tf.Tensor, of shape=(None)

    
    

    
        Returns

    
        -------

    
            tf.Tensor, of shape=(None,2) and dtype DTYPECOMPLEX

    
        */
    
    double pzpt = p[3]/(pp*pt)*SQH*nhel;
    for (int it1 = 0; it1 < 2; it1++) {
        v[it1] = T(0,0);
    }
    
    v[0] = T(-p[1]*pzpt,-nsv*p[2]/pt*SQH);
    v[1] = T(-p[2]*pzpt,nsv*p[1]/pt*SQH);
    
}

template <typename T>
__device__ void _vx_no_BRST_check_massless_pt_zero (const double* p, double nhel, double nsv, T* v) {
        /*

    
        Parameters

    
        ----------

    
            p: tf.Tensor, four-momenta of shape=(None,4)

    
            nhel: tf.Tensor, boson helicity of shape=()

    
            nsv: tf.Tensor, final|initial state of shape=()

    
    

    
        Returns

    
        -------

    
            tf.Tensor, of shape=(None,2) and dtype DTYPECOMPLEX

    
        */
    
    for (int it1 = 0; it1 < 2; it1++) {
        v[it1] = T(0,0);
    }
    
    v[0] = T(-nhel*SQH,0.0);
    v[1] = T(0.0,nsv*signvecc(SQH,p[3]));
    
}

template <typename T>
__device__ void FFV1_0 (T* F1, T* F2, T* V3, T COUP, T& out_final) {
    T cI = T(0,1);
    
    T TMP5 = (F1[2]*(F2[4]*(V3[2]+V3[5])+F2[5]*(V3[3]+cI*(V3[4])))+(F1[3]*(F2[4]*(V3[3]-cI*(V3[4]))+F2[5]*(V3[2]-V3[5]))+(F1[4]*(F2[2]*(V3[2]-V3[5])-F2[3]*(V3[3]+cI*(V3[4])))+F1[5]*(F2[2]*(-V3[3]+cI*(V3[4]))+F2[3]*(V3[2]+V3[5])))));
    T vertex = COUP*-cI*TMP5;
    out_final = vertex;
}

template <typename T>
__device__ void FFV1P0_3 (T* F1, T* F2, T COUP, double M3_, double W3_, T* V3) {
    T cI = T(0,1);
    
    T M3 = T(M3_, 0);
    T W3 = T(W3_, 0);
    for (int it1 = 0; it1 < 6; it1++) {
        V3[it1] = T(0,0);
    }
    
    V3[0] = F1[0]+F2[0];
    V3[1] = F1[1]+F2[1];
    T P3[4];
    P3[0] = T(-V3[0].real(), 0.);
    P3[1] = T( -V3[1].real(), 0.);
    P3[2] = T( -V3[1].imag(), 0.);
    P3[3] = T( -V3[0].imag(), 0.);
    
    T denom = COUP/(P3[0]*P3[0]-P3[1]*P3[1]-P3[2]*P3[2]-P3[3]*P3[3]-M3*(M3-cI*W3));
    V3[2] = denom*(-cI)*(F1[2]*F2[4]+F1[3]*F2[5]+F1[4]*F2[2]+F1[5]*F2[3]);
    V3[3] = denom*(-cI)*(-F1[2]*F2[5]-F1[3]*F2[4]+F1[4]*F2[3]+F1[5]*F2[2]);
    V3[4] = denom*(-cI)*(-cI*(F1[2]*F2[5]+F1[5]*F2[2])+cI*(F1[3]*F2[4]+F1[4]*F2[3]));
    V3[5] = denom*(-cI)*(-F1[2]*F2[4]-F1[5]*F2[3]+F1[3]*F2[5]+F1[4]*F2[2]);
    
}

template <typename T>
__global__ void matrix (const double* all_ps, const double* hel, const double* mdl_MT, const double* mdl_WT, const T* GC_11, double* out_final, const int nevents) {
    //  

    //  MadGraph5_aMC@NLO v. 3.1.0, 2021-03-30

    //  By the MadGraph5_aMC@NLO Development Team

    //  Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch

    //

    // Returns amplitude squared summed/avg over colors

    // for the point with external lines W(0:6,NEXTERNAL)

    //

    // Process: u u~ > t t~ WEIGHTED<=2 @1

    // Process: c c~ > t t~ WEIGHTED<=2 @1

    // Process: d d~ > t t~ WEIGHTED<=2 @1

    // Process: s s~ > t t~ WEIGHTED<=2 @1

    //  

    //  

    // Process parameters

    //  

    int ngraphs = 1;
    int nwavefuncs = 5;
    int ncolor = 2;
    double ZERO = 0.;
    //  

    // Color matrix

    //  

    double denom[2];
    denom[0] = 1;
    denom[1] = 1;
    
    double cf[4];
    cf[0] = 9;
    cf[1] = 3;
    cf[2] = 3;
    cf[3] = 9;
    
    //

    // Model parameters

    //

    // ----------

    // Begin code

    // ----------

    for (int it = blockIdx.x * blockDim.x + threadIdx.x; it < nevents; it += blockDim.x * gridDim.x) {
    T w0[6];
        ixxxxx(all_ps+(16*it + 0),ZERO,hel[0],+1, w0);
        T w1[6];
        oxxxxx(all_ps+(16*it + 4),ZERO,hel[1],-1, w1);
        T w2[6];
        oxxxxx(all_ps+(16*it + 8),mdl_MT[0],hel[2],+1, w2);
        T w3[6];
        ixxxxx(all_ps+(16*it + 12),mdl_MT[0],hel[3],-1, w3);
        T w4[6];
        FFV1P0_3(w0,w1,GC_11[it],ZERO,ZERO, w4);
        // Amplitude(s) for diagram number 1

        T amp0;
        FFV1_0(w3,w2,w4,GC_11[it], amp0);
        
        T jamp[] = {1./2.*(1./3.*amp0),+1./2.*(-amp0)};//array of size 2
        
        double ret = 0;
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                ret += (jamp[i] * cf[i*2+j] * COMPLEX_CONJUGATE(jamp[j])/denom[j]).real();
            }
        }
        
        out_final[it] = ret;
    }
}
template <typename T>
void MatrixFunctor<GPUDevice, T>::operator()(
    const GPUDevice& d,const double* all_ps, const double* hel, const double* mdl_MT, const double* mdl_WT, const T* GC_11, double* out_final, const int nevents, const OpKernelContext* context) {
    // Launch the cuda kernel.
    //
    // See core/util/gpu_kernel_helper.h for example of computing
    // block count and thread_per_block count.
    
    int eventsPerBlock = 1;
    
    int blockSize = DEFAULT_BLOCK_SIZE;
    int numBlocks = (nevents + blockSize - 1) / (eventsPerBlock * blockSize);
    
    //std::cout << blockSize <<  << numBlocks << std::endl;
    if (nevents < blockSize) {
      numBlocks = 1;
      blockSize = nevents;
    }
    
    //int ngraphs = 3;
    //int nwavefuncs = 5;
    //int ncolor = 2;
    
    
    matrix<T><<<numBlocks, blockSize, 0, d.stream()>>>(all_ps, hel, mdl_MT, mdl_WT, GC_11, out_final, nevents);
    
}

// Explicitly instantiate functors for the types of OpKernels registered.
template struct MatrixFunctor<GPUDevice, COMPLEX_TYPE>;
#endif
