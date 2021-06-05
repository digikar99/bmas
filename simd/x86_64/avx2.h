 
typedef __m128 BMAS_svech; // half
typedef __m256 BMAS_svec;
typedef __m256d BMAS_dvec;
typedef __m256 BMAS_sbool;
typedef __m256d BMAS_dbool;
typedef __m256i BMAS_ivec;
typedef __m128i BMAS_ivech;

#define SIMD_SINGLE_STRIDE 8
#define SIMD_DOUBLE_STRIDE 4

// store-load

BMAS_svec static inline BMAS_svec_load(float* ptr){ return _mm256_loadu_ps(ptr); }
void static inline BMAS_svec_store(float* ptr, BMAS_svec v){ return _mm256_storeu_ps(ptr, v); }

BMAS_dvec static inline BMAS_dvec_load(double* ptr){ return _mm256_loadu_pd(ptr); }
void static inline BMAS_dvec_store(double* ptr, BMAS_dvec v){ return _mm256_storeu_pd(ptr, v); }

void static inline BMAS_svec_store_bool(BMAS_sbool v, _Bool* ptr, const long stride){
  for(int i=0; i<SIMD_SINGLE_STRIDE; i++) (ptr+i*stride)[0] = v[i];
}
void static inline BMAS_dvec_store_bool(BMAS_dbool v, _Bool* ptr, const long stride){
  for(int i=0; i<SIMD_DOUBLE_STRIDE; i++) (ptr+i*stride)[0] = v[i];
}

BMAS_svech static inline BMAS_svech_load(float* ptr){ return _mm_loadu_ps(ptr);}
void static inline BMAS_svech_store(float* ptr, BMAS_svech v){ return _mm_storeu_ps(ptr, v);}

BMAS_ivec static inline BMAS_ivec_load(void* ptr){ return _mm256_loadu_si256((__m256i *)ptr);}
void static inline BMAS_ivec_store(void* ptr, BMAS_ivec v){_mm256_storeu_si256((__m256i *)ptr, v);}

// conversion

BMAS_dvec static inline BMAS_svech_to_dvec(BMAS_svech a){return _mm256_cvtps_pd(a);}
BMAS_svech static inline BMAS_dvec_to_svech(BMAS_dvec a){return _mm256_cvtpd_ps(a);}

// basic float arithmetic

BMAS_svec static inline BMAS_vector_sadd(BMAS_svec a, BMAS_svec b){return _mm256_add_ps(a, b);}
BMAS_svec static inline BMAS_vector_ssub(BMAS_svec a, BMAS_svec b){return _mm256_sub_ps(a, b);}
BMAS_svec static inline BMAS_vector_smul(BMAS_svec a, BMAS_svec b){return _mm256_mul_ps(a, b);}
BMAS_svec static inline BMAS_vector_sdiv(BMAS_svec a, BMAS_svec b){return _mm256_div_ps(a, b);}

BMAS_dvec static inline BMAS_vector_dadd(BMAS_dvec a, BMAS_dvec b){return _mm256_add_pd(a, b);}
BMAS_dvec static inline BMAS_vector_dsub(BMAS_dvec a, BMAS_dvec b){return _mm256_sub_pd(a, b);}
BMAS_dvec static inline BMAS_vector_dmul(BMAS_dvec a, BMAS_dvec b){return _mm256_mul_pd(a, b);}
BMAS_dvec static inline BMAS_vector_ddiv(BMAS_dvec a, BMAS_dvec b){return _mm256_div_pd(a, b);}

// integer arithmetic
BMAS_ivec static inline BMAS_vector_i32add(BMAS_ivec a, BMAS_ivec b){return _mm256_add_epi32(a, b);}


// float comparison

// A quick test on numpy will suggest that numpy uses the "ordered" "non-signalling"
// comparisons
// - https://www.felixcloutier.com/x86/cmppd#tbl-3-1
BMAS_sbool static inline BMAS_vector_slt(BMAS_svec a, BMAS_svec b){
  return _mm256_cmp_ps(a, b, _CMP_LT_OQ);
}
BMAS_sbool static inline BMAS_vector_sle(BMAS_svec a, BMAS_svec b){
  return _mm256_cmp_ps(a, b, _CMP_LE_OQ);
}
BMAS_sbool static inline BMAS_vector_seq(BMAS_svec a, BMAS_svec b){
  return _mm256_cmp_ps(a, b, _CMP_EQ_OQ);
}
BMAS_sbool static inline BMAS_vector_sneq(BMAS_svec a, BMAS_svec b){
  return _mm256_cmp_ps(a, b, _CMP_NEQ_OQ);
}
BMAS_sbool static inline BMAS_vector_sgt(BMAS_svec a, BMAS_svec b){
  return _mm256_cmp_ps(a, b, _CMP_GT_OQ);
}
BMAS_sbool static inline BMAS_vector_sge(BMAS_svec a, BMAS_svec b){
  return _mm256_cmp_ps(a, b, _CMP_GE_OQ);
}

BMAS_dbool static inline BMAS_vector_dlt(BMAS_dvec a, BMAS_dvec b){
  return _mm256_cmp_pd(a, b, _CMP_LT_OQ);
}
BMAS_dbool static inline BMAS_vector_dle(BMAS_dvec a, BMAS_dvec b){
  return _mm256_cmp_pd(a, b, _CMP_LE_OQ);
}
BMAS_dbool static inline BMAS_vector_deq(BMAS_dvec a, BMAS_dvec b){
  return _mm256_cmp_pd(a, b, _CMP_EQ_OQ);
}
BMAS_dbool static inline BMAS_vector_dneq(BMAS_dvec a, BMAS_dvec b){
  return _mm256_cmp_pd(a, b, _CMP_NEQ_OQ);
}
BMAS_dbool static inline BMAS_vector_dgt(BMAS_dvec a, BMAS_dvec b){
  return _mm256_cmp_pd(a, b, _CMP_GT_OQ);
}
BMAS_dbool static inline BMAS_vector_dge(BMAS_dvec a, BMAS_dvec b){
  return _mm256_cmp_pd(a, b, _CMP_GE_OQ);
}


// trigonometric

/* BMAS_svec static inline BMAS_vector_ssin(BMAS_svec x){ return Sleef_sinf8_u10avx2(x);} */
__m256 _ZGVdN8v_sinf(__m256 x);
BMAS_svec static inline BMAS_vector_ssin(BMAS_svec x){ return _ZGVdN8v_sinf(x);}
/* BMAS_svec static inline BMAS_vector_scos(BMAS_svec x){ return Sleef_cosf8_u10avx2(x);} */
__m256 _ZGVdN8v_cosf(__m256 x);
BMAS_svec static inline BMAS_vector_scos(BMAS_svec x){ return _ZGVdN8v_cosf(x);}
BMAS_svec static inline BMAS_vector_stan(BMAS_svec x){ return Sleef_tanf8_u10avx2(x);}

BMAS_svec static inline BMAS_vector_sasin(BMAS_svec x){ return Sleef_asinf8_u10avx2(x);}
BMAS_svec static inline BMAS_vector_sacos(BMAS_svec x){ return Sleef_acosf8_u10avx2(x);}
BMAS_svec static inline BMAS_vector_satan(BMAS_svec x){ return Sleef_atanf8_u10avx2(x);}

BMAS_svec static inline BMAS_vector_ssinh(BMAS_svec x){ return Sleef_sinhf8_u10avx2(x);}
BMAS_svec static inline BMAS_vector_scosh(BMAS_svec x){ return Sleef_coshf8_u10avx2(x);}
BMAS_svec static inline BMAS_vector_stanh(BMAS_svec x){ return Sleef_tanhf8_u10avx2(x);}

BMAS_svec static inline BMAS_vector_sasinh(BMAS_svec x){ return Sleef_asinhf8_u10avx2(x);}
BMAS_svec static inline BMAS_vector_sacosh(BMAS_svec x){ return Sleef_acoshf8_u10avx2(x);}
BMAS_svec static inline BMAS_vector_satanh(BMAS_svec x){ return Sleef_atanhf8_u10avx2(x);}


BMAS_dvec static inline BMAS_vector_dsin(BMAS_dvec x){ return Sleef_sind4_u10avx2(x);}
BMAS_dvec static inline BMAS_vector_dcos(BMAS_dvec x){ return Sleef_cosd4_u10avx2(x);}
BMAS_dvec static inline BMAS_vector_dtan(BMAS_dvec x){ return Sleef_tand4_u10avx2(x);}

BMAS_dvec static inline BMAS_vector_dasin(BMAS_dvec x){ return Sleef_asind4_u10avx2(x);}
BMAS_dvec static inline BMAS_vector_dacos(BMAS_dvec x){ return Sleef_acosd4_u10avx2(x);}
BMAS_dvec static inline BMAS_vector_datan(BMAS_dvec x){ return Sleef_atand4_u10avx2(x);}

BMAS_dvec static inline BMAS_vector_dsinh(BMAS_dvec x){ return Sleef_sinhd4_u10avx2(x);}
BMAS_dvec static inline BMAS_vector_dcosh(BMAS_dvec x){ return Sleef_coshd4_u10avx2(x);}
BMAS_dvec static inline BMAS_vector_dtanh(BMAS_dvec x){ return Sleef_tanhd4_u10avx2(x);}

BMAS_dvec static inline BMAS_vector_dasinh(BMAS_dvec x){ return Sleef_asinhd4_u10avx2(x);}
BMAS_dvec static inline BMAS_vector_dacosh(BMAS_dvec x){ return Sleef_acoshd4_u10avx2(x);}
BMAS_dvec static inline BMAS_vector_datanh(BMAS_dvec x){ return Sleef_atanhd4_u10avx2(x);}


// log

BMAS_svec static inline BMAS_vector_slog(BMAS_svec x)  { return Sleef_logf8_u10avx2(x);}
BMAS_svec static inline BMAS_vector_slog10(BMAS_svec x){ return Sleef_log10f8_u10avx2(x);}
BMAS_svec static inline BMAS_vector_slog2(BMAS_svec x) { return Sleef_log2f8_u10avx2(x);}
BMAS_svec static inline BMAS_vector_slog1p(BMAS_svec x){ return Sleef_log1pf8_u10avx2(x);}

BMAS_dvec static inline BMAS_vector_dlog(BMAS_dvec x)  { return Sleef_logd4_u10avx2(x);}
BMAS_dvec static inline BMAS_vector_dlog10(BMAS_dvec x){ return Sleef_log10d4_u10avx2(x);}
BMAS_dvec static inline BMAS_vector_dlog2(BMAS_dvec x) { return Sleef_log2d4_u10avx2(x);}
BMAS_dvec static inline BMAS_vector_dlog1p(BMAS_dvec x){ return Sleef_log1pd4_u10avx2(x);}

// exp

BMAS_svec static inline BMAS_vector_sexp(BMAS_svec x)  { return Sleef_expf8_u10avx2(x);}
BMAS_svec static inline BMAS_vector_sexp10(BMAS_svec x){ return Sleef_exp10f8_u10avx2(x);}
BMAS_svec static inline BMAS_vector_sexp2(BMAS_svec x) { return Sleef_exp2f8_u10avx2(x);}
BMAS_svec static inline BMAS_vector_sexpm1(BMAS_svec x){ return Sleef_expm1f8_u10avx2(x);}

BMAS_dvec static inline BMAS_vector_dexp(BMAS_dvec x)  { return Sleef_expd4_u10avx2(x);}
BMAS_dvec static inline BMAS_vector_dexp10(BMAS_dvec x){ return Sleef_exp10d4_u10avx2(x);}
BMAS_dvec static inline BMAS_vector_dexp2(BMAS_dvec x) { return Sleef_exp2d4_u10avx2(x);}
BMAS_dvec static inline BMAS_vector_dexpm1(BMAS_dvec x){ return Sleef_expm1d4_u10avx2(x);}


// pow and atan2
BMAS_svec static inline BMAS_vector_spow(BMAS_svec a, BMAS_svec b){
  return Sleef_powf8_u10avx2(a, b);
}
BMAS_svec static inline BMAS_vector_satan2(BMAS_svec a, BMAS_svec b){
  return Sleef_atan2f8_u10avx2(a, b);
}

BMAS_dvec static inline BMAS_vector_dpow(BMAS_dvec a, BMAS_dvec b){
  return Sleef_powd4_u10avx2(a, b);
}
BMAS_dvec static inline BMAS_vector_datan2(BMAS_dvec a, BMAS_dvec b){
  return Sleef_atan2d4_u10avx2(a, b);
}


// misc

BMAS_svec static inline BMAS_vector_sfabs(BMAS_svec x)  { return Sleef_fabsf8_avx2(x);}
BMAS_svec static inline BMAS_vector_sceil(BMAS_svec x)  { return Sleef_ceilf8_avx2(x);}
BMAS_svec static inline BMAS_vector_strunc(BMAS_svec x) { return Sleef_truncf8_avx2(x);}
BMAS_svec static inline BMAS_vector_sfloor(BMAS_svec x) { return Sleef_floorf8_avx2(x);}
BMAS_svec static inline BMAS_vector_sround(BMAS_svec x) { return Sleef_roundf8_avx2(x);}

BMAS_dvec static inline BMAS_vector_dfabs(BMAS_dvec x)  { return Sleef_fabsd4_avx2(x);}
BMAS_dvec static inline BMAS_vector_dceil(BMAS_dvec x)  { return Sleef_ceild4_avx2(x);}
BMAS_dvec static inline BMAS_vector_dtrunc(BMAS_dvec x) { return Sleef_truncd4_avx2(x);}
BMAS_dvec static inline BMAS_vector_dfloor(BMAS_dvec x) { return Sleef_floord4_avx2(x);}
BMAS_dvec static inline BMAS_vector_dround(BMAS_dvec x) { return Sleef_roundd4_avx2(x);}

