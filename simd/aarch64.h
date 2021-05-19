#include <arm_neon.h>
#include "sleef/sleefinline_advsimd.h"

typedef float32x2_t BMAS_svech; // half
typedef float32x4_t BMAS_svec;
typedef float64x2_t BMAS_dvec;
typedef uint32x4_t BMAS_sbool;
typedef uint64x2_t BMAS_dbool;

#define SIMD_SINGLE_STRIDE 4
#define SIMD_DOUBLE_STRIDE 2

// store-load

BMAS_svec static inline BMAS_svec_load(float* ptr){ return vld1q_f32(ptr); }
void static inline BMAS_svec_store(float* ptr, BMAS_svec v){ return vst1q_f32(ptr, v); }

BMAS_dvec static inline BMAS_dvec_load(double* ptr){ return vld1q_f64(ptr); }
void static inline BMAS_dvec_store(double* ptr, BMAS_dvec v){ return vst1q_f64(ptr, v); }

void static inline BMAS_svec_store_bool(BMAS_sbool v, _Bool* ptr, const long stride){
  for(int i=0; i<SIMD_SINGLE_STRIDE; i++) (ptr+i*stride)[0] = v[i];
}
void static inline BMAS_dvec_store_bool(BMAS_dbool v, _Bool* ptr, const long stride){
  for(int i=0; i<SIMD_DOUBLE_STRIDE; i++) (ptr+i*stride)[0] = v[i];
}

BMAS_svech static inline BMAS_svech_load(float* ptr){ return vld1_f32(ptr);}
void static inline BMAS_svech_store(float* ptr, BMAS_svech v){ return vst1_f32(ptr, v);}

// conversion

BMAS_dvec static inline BMAS_svech_to_dvec(BMAS_svech a){return vcvt_f64_f32(a);}
BMAS_svech static inline BMAS_dvec_to_svech(BMAS_dvec a){return vcvt_f32_f64(a);}

// basic arithmetic

BMAS_svec static inline BMAS_vector_sadd(BMAS_svec a, BMAS_svec b){return vaddq_f32(a, b);}
BMAS_svec static inline BMAS_vector_ssub(BMAS_svec a, BMAS_svec b){return vsubq_f32(a, b);}
BMAS_svec static inline BMAS_vector_smul(BMAS_svec a, BMAS_svec b){return vmulq_f32(a, b);}
BMAS_svec static inline BMAS_vector_sdiv(BMAS_svec a, BMAS_svec b){return vdivq_f32(a, b);}

BMAS_dvec static inline BMAS_vector_dadd(BMAS_dvec a, BMAS_dvec b){return vaddq_f64(a, b);}
BMAS_dvec static inline BMAS_vector_dsub(BMAS_dvec a, BMAS_dvec b){return vsubq_f64(a, b);}
BMAS_dvec static inline BMAS_vector_dmul(BMAS_dvec a, BMAS_dvec b){return vmulq_f64(a, b);}
BMAS_dvec static inline BMAS_vector_ddiv(BMAS_dvec a, BMAS_dvec b){return vdivq_f64(a, b);}


// comparison

BMAS_sbool static inline BMAS_vector_slt(BMAS_svec a, BMAS_svec b){return vcltq_f32(a, b);}
BMAS_sbool static inline BMAS_vector_sle(BMAS_svec a, BMAS_svec b){return vcleq_f32(a, b);}
BMAS_sbool static inline BMAS_vector_seq(BMAS_svec a, BMAS_svec b){return vceqq_f32(a, b);}
BMAS_sbool static inline BMAS_vector_sneq(BMAS_svec a, BMAS_svec b){return vceqzq_u32(vceqq_f32(a, b));}
BMAS_sbool static inline BMAS_vector_sgt(BMAS_svec a, BMAS_svec b){return vcgtq_f32(a, b);}
BMAS_sbool static inline BMAS_vector_sge(BMAS_svec a, BMAS_svec b){return vcgeq_f32(a, b);}

BMAS_dbool static inline BMAS_vector_dlt(BMAS_dvec a, BMAS_dvec b){return vcltq_f64(a, b);}
BMAS_dbool static inline BMAS_vector_dle(BMAS_dvec a, BMAS_dvec b){return vcleq_f64(a, b);}
BMAS_dbool static inline BMAS_vector_deq(BMAS_dvec a, BMAS_dvec b){return vceqq_f64(a, b);}
BMAS_dbool static inline BMAS_vector_dneq(BMAS_dvec a, BMAS_dvec b){return vceqzq_u64(vceqq_f64(a, b));}
BMAS_dbool static inline BMAS_vector_dgt(BMAS_dvec a, BMAS_dvec b){return vcgtq_f64(a, b);}
BMAS_dbool static inline BMAS_vector_dge(BMAS_dvec a, BMAS_dvec b){return vcgeq_f64(a, b);}


// trigonometric

BMAS_svec static inline BMAS_vector_ssin(BMAS_svec x){ return Sleef_sinf4_u10advsimd(x);}
BMAS_svec static inline BMAS_vector_scos(BMAS_svec x){ return Sleef_cosf4_u10advsimd(x);}
BMAS_svec static inline BMAS_vector_stan(BMAS_svec x){ return Sleef_tanf4_u10advsimd(x);}

BMAS_svec static inline BMAS_vector_sasin(BMAS_svec x){ return Sleef_asinf4_u10advsimd(x);}
BMAS_svec static inline BMAS_vector_sacos(BMAS_svec x){ return Sleef_acosf4_u10advsimd(x);}
BMAS_svec static inline BMAS_vector_satan(BMAS_svec x){ return Sleef_atanf4_u10advsimd(x);}

BMAS_svec static inline BMAS_vector_ssinh(BMAS_svec x){ return Sleef_sinhf4_u10advsimd(x);}
BMAS_svec static inline BMAS_vector_scosh(BMAS_svec x){ return Sleef_coshf4_u10advsimd(x);}
BMAS_svec static inline BMAS_vector_stanh(BMAS_svec x){ return Sleef_tanhf4_u10advsimd(x);}

BMAS_svec static inline BMAS_vector_sasinh(BMAS_svec x){ return Sleef_asinhf4_u10advsimd(x);}
BMAS_svec static inline BMAS_vector_sacosh(BMAS_svec x){ return Sleef_acoshf4_u10advsimd(x);}
BMAS_svec static inline BMAS_vector_satanh(BMAS_svec x){ return Sleef_atanhf4_u10advsimd(x);}


BMAS_dvec static inline BMAS_vector_dsin(BMAS_dvec x){ return Sleef_sind2_u10advsimd(x);}
BMAS_dvec static inline BMAS_vector_dcos(BMAS_dvec x){ return Sleef_cosd2_u10advsimd(x);}
BMAS_dvec static inline BMAS_vector_dtan(BMAS_dvec x){ return Sleef_tand2_u10advsimd(x);}

BMAS_dvec static inline BMAS_vector_dasin(BMAS_dvec x){ return Sleef_asind2_u10advsimd(x);}
BMAS_dvec static inline BMAS_vector_dacos(BMAS_dvec x){ return Sleef_acosd2_u10advsimd(x);}
BMAS_dvec static inline BMAS_vector_datan(BMAS_dvec x){ return Sleef_atand2_u10advsimd(x);}

BMAS_dvec static inline BMAS_vector_dsinh(BMAS_dvec x){ return Sleef_sinhd2_u10advsimd(x);}
BMAS_dvec static inline BMAS_vector_dcosh(BMAS_dvec x){ return Sleef_coshd2_u10advsimd(x);}
BMAS_dvec static inline BMAS_vector_dtanh(BMAS_dvec x){ return Sleef_tanhd2_u10advsimd(x);}

BMAS_dvec static inline BMAS_vector_dasinh(BMAS_dvec x){ return Sleef_asinhd2_u10advsimd(x);}
BMAS_dvec static inline BMAS_vector_dacosh(BMAS_dvec x){ return Sleef_acoshd2_u10advsimd(x);}
BMAS_dvec static inline BMAS_vector_datanh(BMAS_dvec x){ return Sleef_atanhd2_u10advsimd(x);}


// log

BMAS_svec static inline BMAS_vector_slog(BMAS_svec x)  { return Sleef_logf4_u10advsimd(x);}
BMAS_svec static inline BMAS_vector_slog10(BMAS_svec x){ return Sleef_log10f4_u10advsimd(x);}
BMAS_svec static inline BMAS_vector_slog2(BMAS_svec x) { return Sleef_log2f4_u10advsimd(x);}
BMAS_svec static inline BMAS_vector_slog1p(BMAS_svec x){ return Sleef_log1pf4_u10advsimd(x);}

BMAS_dvec static inline BMAS_vector_dlog(BMAS_dvec x)  { return Sleef_logd2_u10advsimd(x);}
BMAS_dvec static inline BMAS_vector_dlog10(BMAS_dvec x){ return Sleef_log10d2_u10advsimd(x);}
BMAS_dvec static inline BMAS_vector_dlog2(BMAS_dvec x) { return Sleef_log2d2_u10advsimd(x);}
BMAS_dvec static inline BMAS_vector_dlog1p(BMAS_dvec x){ return Sleef_log1pd2_u10advsimd(x);}

// exp

BMAS_svec static inline BMAS_vector_sexp(BMAS_svec x)  { return Sleef_expf4_u10advsimd(x);}
BMAS_svec static inline BMAS_vector_sexp10(BMAS_svec x){ return Sleef_exp10f4_u10advsimd(x);}
BMAS_svec static inline BMAS_vector_sexp2(BMAS_svec x) { return Sleef_exp2f4_u10advsimd(x);}
BMAS_svec static inline BMAS_vector_sexpm1(BMAS_svec x){ return Sleef_expm1f4_u10advsimd(x);}

BMAS_dvec static inline BMAS_vector_dexp(BMAS_dvec x)  { return Sleef_expd2_u10advsimd(x);}
BMAS_dvec static inline BMAS_vector_dexp10(BMAS_dvec x){ return Sleef_exp10d2_u10advsimd(x);}
BMAS_dvec static inline BMAS_vector_dexp2(BMAS_dvec x) { return Sleef_exp2d2_u10advsimd(x);}
BMAS_dvec static inline BMAS_vector_dexpm1(BMAS_dvec x){ return Sleef_expm1d2_u10advsimd(x);}


// pow and atan2
BMAS_svec static inline BMAS_vector_spow(BMAS_svec a, BMAS_svec b){
  return Sleef_powf4_u10advsimd(a, b);
}
BMAS_svec static inline BMAS_vector_satan2(BMAS_svec a, BMAS_svec b){
  return Sleef_atan2f4_u10advsimd(a, b);
}

BMAS_dvec static inline BMAS_vector_dpow(BMAS_dvec a, BMAS_dvec b){
  return Sleef_powd2_u10advsimd(a, b);
}
BMAS_dvec static inline BMAS_vector_datan2(BMAS_dvec a, BMAS_dvec b){
  return Sleef_atan2d2_u10advsimd(a, b);
}


// misc

BMAS_svec static inline BMAS_vector_sfabs(BMAS_svec x)  { return Sleef_fabsf4_advsimd(x);}
BMAS_svec static inline BMAS_vector_sceil(BMAS_svec x)  { return Sleef_ceilf4_advsimd(x);}
BMAS_svec static inline BMAS_vector_strunc(BMAS_svec x) { return Sleef_truncf4_advsimd(x);}
BMAS_svec static inline BMAS_vector_sfloor(BMAS_svec x) { return Sleef_floorf4_advsimd(x);}
BMAS_svec static inline BMAS_vector_sround(BMAS_svec x) { return Sleef_roundf4_advsimd(x);}

BMAS_dvec static inline BMAS_vector_dfabs(BMAS_dvec x)  { return Sleef_fabsd2_advsimd(x);}
BMAS_dvec static inline BMAS_vector_dceil(BMAS_dvec x)  { return Sleef_ceild2_advsimd(x);}
BMAS_dvec static inline BMAS_vector_dtrunc(BMAS_dvec x) { return Sleef_truncd2_advsimd(x);}
BMAS_dvec static inline BMAS_vector_dfloor(BMAS_dvec x) { return Sleef_floord2_advsimd(x);}
BMAS_dvec static inline BMAS_vector_dround(BMAS_dvec x) { return Sleef_roundd2_advsimd(x);}
