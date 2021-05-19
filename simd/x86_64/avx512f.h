 
typedef __m256  BMAS_svech; // half
typedef __m512  BMAS_svec;
typedef __m512d BMAS_dvec;
typedef __mmask8  BMAS_sbool;
typedef __mmask16 BMAS_dbool;

#define SIMD_SINGLE_STRIDE 16
#define SIMD_DOUBLE_STRIDE 8

// store-load

BMAS_svec static inline BMAS_svec_load(float* ptr){ return _mm512_loadu_ps(ptr); }
void static inline BMAS_svec_store(float* ptr, BMAS_svec v){ return _mm512_storeu_ps(ptr, v); }

BMAS_dvec static inline BMAS_dvec_load(double* ptr){ return _mm512_loadu_pd(ptr); }
void static inline BMAS_dvec_store(double* ptr, BMAS_dvec v){ return _mm512_storeu_pd(ptr, v); }

void static inline BMAS_svec_store_bool(BMAS_sbool v, _Bool* ptr, const long stride){
  for(int i=0; i<SIMD_SINGLE_STRIDE; i++) (ptr+i*stride)[0] = (v>>i)&1;
}
void static inline BMAS_dvec_store_bool(BMAS_dbool v, _Bool* ptr, const long stride){
  for(int i=0; i<SIMD_DOUBLE_STRIDE; i++) (ptr+i*stride)[0] = (v>>i)&1;
}

BMAS_svech static inline BMAS_svech_load(float* ptr){ return _mm256_loadu_ps(ptr);}
void static inline BMAS_svech_store(float* ptr, BMAS_svech v){ return _mm256_storeu_ps(ptr, v);}

// conversion

BMAS_dvec static inline BMAS_svech_to_dvec(BMAS_svech a){return _mm512_cvtps_pd(a);}
BMAS_svech static inline BMAS_dvec_to_svech(BMAS_dvec a){return _mm512_cvtpd_ps(a);}

// basic arithmetic

BMAS_svec static inline BMAS_vector_sadd(BMAS_svec a, BMAS_svec b){return _mm512_add_ps(a, b);}
BMAS_svec static inline BMAS_vector_ssub(BMAS_svec a, BMAS_svec b){return _mm512_sub_ps(a, b);}
BMAS_svec static inline BMAS_vector_smul(BMAS_svec a, BMAS_svec b){return _mm512_mul_ps(a, b);}
BMAS_svec static inline BMAS_vector_sdiv(BMAS_svec a, BMAS_svec b){return _mm512_div_ps(a, b);}

BMAS_dvec static inline BMAS_vector_dadd(BMAS_dvec a, BMAS_dvec b){return _mm512_add_pd(a, b);}
BMAS_dvec static inline BMAS_vector_dsub(BMAS_dvec a, BMAS_dvec b){return _mm512_sub_pd(a, b);}
BMAS_dvec static inline BMAS_vector_dmul(BMAS_dvec a, BMAS_dvec b){return _mm512_mul_pd(a, b);}
BMAS_dvec static inline BMAS_vector_ddiv(BMAS_dvec a, BMAS_dvec b){return _mm512_div_pd(a, b);}


// comparison

BMAS_sbool static inline BMAS_vector_slt(BMAS_svec a, BMAS_svec b){
  return _mm512_cmp_ps_mask(a, b, _CMP_LT_OQ);
}
BMAS_sbool static inline BMAS_vector_sle(BMAS_svec a, BMAS_svec b){
  return _mm512_cmp_ps_mask(a, b, _CMP_LE_OQ);
}
BMAS_sbool static inline BMAS_vector_seq(BMAS_svec a, BMAS_svec b){
  return _mm512_cmp_ps_mask(a, b, _CMP_EQ_OQ);
}
BMAS_sbool static inline BMAS_vector_sneq(BMAS_svec a, BMAS_svec b){
  return _mm512_cmp_ps_mask(a, b, _CMP_NEQ_OQ);
}
BMAS_sbool static inline BMAS_vector_sgt(BMAS_svec a, BMAS_svec b){
  return _mm512_cmp_ps_mask(a, b, _CMP_GT_OQ);
}
BMAS_sbool static inline BMAS_vector_sge(BMAS_svec a, BMAS_svec b){
  return _mm512_cmp_ps_mask(a, b, _CMP_GE_OQ);
}

BMAS_dbool static inline BMAS_vector_dlt(BMAS_dvec a, BMAS_dvec b){
  return _mm512_cmp_pd_mask(a, b, _CMP_LT_OQ);
}
BMAS_dbool static inline BMAS_vector_dle(BMAS_dvec a, BMAS_dvec b){
  return _mm512_cmp_pd_mask(a, b, _CMP_LE_OQ);
}
BMAS_dbool static inline BMAS_vector_deq(BMAS_dvec a, BMAS_dvec b){
  return _mm512_cmp_pd_mask(a, b, _CMP_EQ_OQ);
}
BMAS_dbool static inline BMAS_vector_dneq(BMAS_dvec a, BMAS_dvec b){
  return _mm512_cmp_pd_mask(a, b, _CMP_NEQ_OQ);
}
BMAS_dbool static inline BMAS_vector_dgt(BMAS_dvec a, BMAS_dvec b){
  return _mm512_cmp_pd_mask(a, b, _CMP_GT_OQ);
}
BMAS_dbool static inline BMAS_vector_dge(BMAS_dvec a, BMAS_dvec b){
  return _mm512_cmp_pd_mask(a, b, _CMP_GE_OQ);
}


// trigonometric

BMAS_svec static inline BMAS_vector_ssin(BMAS_svec x){ return Sleef_sinf16_u10avx512f(x);}
BMAS_svec static inline BMAS_vector_scos(BMAS_svec x){ return Sleef_cosf16_u10avx512f(x);}
BMAS_svec static inline BMAS_vector_stan(BMAS_svec x){ return Sleef_tanf16_u10avx512f(x);}

BMAS_svec static inline BMAS_vector_sasin(BMAS_svec x){ return Sleef_asinf16_u10avx512f(x);}
BMAS_svec static inline BMAS_vector_sacos(BMAS_svec x){ return Sleef_acosf16_u10avx512f(x);}
BMAS_svec static inline BMAS_vector_satan(BMAS_svec x){ return Sleef_atanf16_u10avx512f(x);}

BMAS_svec static inline BMAS_vector_ssinh(BMAS_svec x){ return Sleef_sinhf16_u10avx512f(x);}
BMAS_svec static inline BMAS_vector_scosh(BMAS_svec x){ return Sleef_coshf16_u10avx512f(x);}
BMAS_svec static inline BMAS_vector_stanh(BMAS_svec x){ return Sleef_tanhf16_u10avx512f(x);}

BMAS_svec static inline BMAS_vector_sasinh(BMAS_svec x){ return Sleef_asinhf16_u10avx512f(x);}
BMAS_svec static inline BMAS_vector_sacosh(BMAS_svec x){ return Sleef_acoshf16_u10avx512f(x);}
BMAS_svec static inline BMAS_vector_satanh(BMAS_svec x){ return Sleef_atanhf16_u10avx512f(x);}


BMAS_dvec static inline BMAS_vector_dsin(BMAS_dvec x){ return Sleef_sind8_u10avx512f(x);}
BMAS_dvec static inline BMAS_vector_dcos(BMAS_dvec x){ return Sleef_cosd8_u10avx512f(x);}
BMAS_dvec static inline BMAS_vector_dtan(BMAS_dvec x){ return Sleef_tand8_u10avx512f(x);}

BMAS_dvec static inline BMAS_vector_dasin(BMAS_dvec x){ return Sleef_asind8_u10avx512f(x);}
BMAS_dvec static inline BMAS_vector_dacos(BMAS_dvec x){ return Sleef_acosd8_u10avx512f(x);}
BMAS_dvec static inline BMAS_vector_datan(BMAS_dvec x){ return Sleef_atand8_u10avx512f(x);}

BMAS_dvec static inline BMAS_vector_dsinh(BMAS_dvec x){ return Sleef_sinhd8_u10avx512f(x);}
BMAS_dvec static inline BMAS_vector_dcosh(BMAS_dvec x){ return Sleef_coshd8_u10avx512f(x);}
BMAS_dvec static inline BMAS_vector_dtanh(BMAS_dvec x){ return Sleef_tanhd8_u10avx512f(x);}

BMAS_dvec static inline BMAS_vector_dasinh(BMAS_dvec x){ return Sleef_asinhd8_u10avx512f(x);}
BMAS_dvec static inline BMAS_vector_dacosh(BMAS_dvec x){ return Sleef_acoshd8_u10avx512f(x);}
BMAS_dvec static inline BMAS_vector_datanh(BMAS_dvec x){ return Sleef_atanhd8_u10avx512f(x);}


// log

BMAS_svec static inline BMAS_vector_slog(BMAS_svec x)  { return Sleef_logf16_u10avx512f(x);}
BMAS_svec static inline BMAS_vector_slog10(BMAS_svec x){ return Sleef_log10f16_u10avx512f(x);}
BMAS_svec static inline BMAS_vector_slog2(BMAS_svec x) { return Sleef_log2f16_u10avx512f(x);}
BMAS_svec static inline BMAS_vector_slog1p(BMAS_svec x){ return Sleef_log1pf16_u10avx512f(x);}

BMAS_dvec static inline BMAS_vector_dlog(BMAS_dvec x)  { return Sleef_logd8_u10avx512f(x);}
BMAS_dvec static inline BMAS_vector_dlog10(BMAS_dvec x){ return Sleef_log10d8_u10avx512f(x);}
BMAS_dvec static inline BMAS_vector_dlog2(BMAS_dvec x) { return Sleef_log2d8_u10avx512f(x);}
BMAS_dvec static inline BMAS_vector_dlog1p(BMAS_dvec x){ return Sleef_log1pd8_u10avx512f(x);}

// exp

BMAS_svec static inline BMAS_vector_sexp(BMAS_svec x)  { return Sleef_expf16_u10avx512f(x);}
BMAS_svec static inline BMAS_vector_sexp10(BMAS_svec x){ return Sleef_exp10f16_u10avx512f(x);}
BMAS_svec static inline BMAS_vector_sexp2(BMAS_svec x) { return Sleef_exp2f16_u10avx512f(x);}
BMAS_svec static inline BMAS_vector_sexpm1(BMAS_svec x){ return Sleef_expm1f16_u10avx512f(x);}

BMAS_dvec static inline BMAS_vector_dexp(BMAS_dvec x)  { return Sleef_expd8_u10avx512f(x);}
BMAS_dvec static inline BMAS_vector_dexp10(BMAS_dvec x){ return Sleef_exp10d8_u10avx512f(x);}
BMAS_dvec static inline BMAS_vector_dexp2(BMAS_dvec x) { return Sleef_exp2d8_u10avx512f(x);}
BMAS_dvec static inline BMAS_vector_dexpm1(BMAS_dvec x){ return Sleef_expm1d8_u10avx512f(x);}


// pow and atan2
BMAS_svec static inline BMAS_vector_spow(BMAS_svec a, BMAS_svec b){
  return Sleef_powf16_u10avx512f(a, b);
}
BMAS_svec static inline BMAS_vector_satan2(BMAS_svec a, BMAS_svec b){
  return Sleef_atan2f16_u10avx512f(a, b);
}

BMAS_dvec static inline BMAS_vector_dpow(BMAS_dvec a, BMAS_dvec b){
  return Sleef_powd8_u10avx512f(a, b);
}
BMAS_dvec static inline BMAS_vector_datan2(BMAS_dvec a, BMAS_dvec b){
  return Sleef_atan2d8_u10avx512f(a, b);
}


// misc

BMAS_svec static inline BMAS_vector_sfabs(BMAS_svec x)  { return Sleef_fabsf16_avx512f(x);}
BMAS_svec static inline BMAS_vector_sceil(BMAS_svec x)  { return Sleef_ceilf16_avx512f(x);}
BMAS_svec static inline BMAS_vector_strunc(BMAS_svec x) { return Sleef_truncf16_avx512f(x);}
BMAS_svec static inline BMAS_vector_sfloor(BMAS_svec x) { return Sleef_floorf16_avx512f(x);}
BMAS_svec static inline BMAS_vector_sround(BMAS_svec x) { return Sleef_roundf16_avx512f(x);}

BMAS_dvec static inline BMAS_vector_dfabs(BMAS_dvec x)  { return Sleef_fabsd8_avx512f(x);}
BMAS_dvec static inline BMAS_vector_dceil(BMAS_dvec x)  { return Sleef_ceild8_avx512f(x);}
BMAS_dvec static inline BMAS_vector_dtrunc(BMAS_dvec x) { return Sleef_truncd8_avx512f(x);}
BMAS_dvec static inline BMAS_vector_dfloor(BMAS_dvec x) { return Sleef_floord8_avx512f(x);}
BMAS_dvec static inline BMAS_vector_dround(BMAS_dvec x) { return Sleef_roundd8_avx512f(x);}

