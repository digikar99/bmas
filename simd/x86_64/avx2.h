
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
BMAS_ivech static inline BMAS_ivech_load(void* ptr){ return _mm_loadu_si128((__m128i *)ptr);}
void static inline BMAS_ivec_store(void* ptr, BMAS_ivec v){_mm256_storeu_si256((__m256i *)ptr, v);}

// conversion to floats

BMAS_dvec static inline BMAS_svech_to_dvec(BMAS_svech v){return _mm256_cvtps_pd(v);}
BMAS_svech static inline BMAS_dvec_to_svech(BMAS_dvec v){return _mm256_cvtpd_ps(v);}

BMAS_dvec static inline BMAS_ivech_to_dvec_i8(BMAS_ivech v){
  return _mm256_cvtepi32_pd(_mm_cvtepi8_epi32(v));
}
BMAS_dvec static inline BMAS_ivech_to_dvec_i16(BMAS_ivech v){
  return _mm256_cvtepi32_pd(_mm_cvtepi16_epi32(v));
}
BMAS_dvec static inline BMAS_ivech_to_dvec_i32(BMAS_ivech v){
  return _mm256_cvtepi32_pd(v);
}
// Credits for 64-bit conversions: https://stackoverflow.com/questions/41144668/how-to-efficiently-perform-double-int64-conversions-with-sse-avx
BMAS_dvec static inline BMAS_ivec_to_dvec_i64(BMAS_ivec v){
  __m256i magic_i_lo   = _mm256_set1_epi64x(0x4330000000000000);
  __m256i magic_i_hi32 = _mm256_set1_epi64x(0x4530000080000000);
  __m256i magic_i_all  = _mm256_set1_epi64x(0x4530000080100000);
  __m256d magic_d_all  = _mm256_castsi256_pd(magic_i_all);
  __m256i v_lo         = _mm256_blend_epi32(magic_i_lo, v, 0b01010101);
  __m256i v_hi         = _mm256_srli_epi64(v, 32);
  v_hi         = _mm256_xor_si256(v_hi, magic_i_hi32);
  __m256d v_hi_dbl     = _mm256_sub_pd(_mm256_castsi256_pd(v_hi), magic_d_all);
  __m256d result       = _mm256_add_pd(v_hi_dbl, _mm256_castsi256_pd(v_lo));
  return result;
}

BMAS_dvec static inline BMAS_ivech_to_dvec_u8(BMAS_ivech v){
  return _mm256_cvtepi32_pd(_mm_cvtepu8_epi32(v));
}
BMAS_dvec static inline BMAS_ivech_to_dvec_u16(BMAS_ivech v){
  return _mm256_cvtepi32_pd(_mm_cvtepu16_epi32(v));
}
BMAS_dvec static inline BMAS_ivech_to_dvec_u32(BMAS_ivech v){
  return BMAS_ivec_to_dvec_i64(_mm256_cvtepu32_epi64(v));
}

BMAS_dvec static inline BMAS_ivec_to_dvec_u64(BMAS_ivec v){
  __m256i magic_i_lo   = _mm256_set1_epi64x(0x4330000000000000);
  __m256i magic_i_hi32 = _mm256_set1_epi64x(0x4530000000000000);
  __m256i magic_i_all  = _mm256_set1_epi64x(0x4530000000100000);
  __m256d magic_d_all  = _mm256_castsi256_pd(magic_i_all);
  __m256i v_lo         = _mm256_blend_epi32(magic_i_lo, v, 0b01010101);
  __m256i v_hi         = _mm256_srli_epi64(v, 32);
  v_hi         = _mm256_xor_si256(v_hi, magic_i_hi32);
  __m256d v_hi_dbl     = _mm256_sub_pd(_mm256_castsi256_pd(v_hi), magic_d_all);
  __m256d result       = _mm256_add_pd(v_hi_dbl, _mm256_castsi256_pd(v_lo));
  return result;
}


BMAS_svec static inline BMAS_ivech_to_svec_i8(BMAS_ivech v){
  return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v));
}
BMAS_svec static inline BMAS_ivech_to_svec_i16(BMAS_ivech v){
  return _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(v));
}
BMAS_svec static inline BMAS_ivec_to_svec_i32(BMAS_ivec v){
  return _mm256_cvtepi32_ps(v);
}
BMAS_svech static inline BMAS_ivec_to_svech_i64(BMAS_ivec v){
  return _mm256_cvtpd_ps(BMAS_ivec_to_dvec_i64(v));
}

BMAS_svec static inline BMAS_ivech_to_svec_u8(BMAS_ivech v){
  return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(v));
}
BMAS_svec static inline BMAS_ivech_to_svec_u16(BMAS_ivech v){
  return _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(v));
}
BMAS_svech static inline BMAS_ivech_to_svec_u32(BMAS_ivech v){
  return _mm256_cvtpd_ps(BMAS_ivech_to_dvec_u32(v));
}
BMAS_svech static inline BMAS_ivec_to_svech_u64(BMAS_ivec v){
  return _mm256_cvtpd_ps(BMAS_ivec_to_dvec_u64(v));
}



// basic float arithmetic

BMAS_svec static inline BMAS_vector_sadd(BMAS_svec a, BMAS_svec b){return _mm256_add_ps(a, b);}
BMAS_svec static inline BMAS_vector_ssub(BMAS_svec a, BMAS_svec b){return _mm256_sub_ps(a, b);}
BMAS_svec static inline BMAS_vector_smul(BMAS_svec a, BMAS_svec b){return _mm256_mul_ps(a, b);}
BMAS_svec static inline BMAS_vector_sdiv(BMAS_svec a, BMAS_svec b){return _mm256_div_ps(a, b);}

BMAS_dvec static inline BMAS_vector_dadd(BMAS_dvec a, BMAS_dvec b){return _mm256_add_pd(a, b);}
BMAS_dvec static inline BMAS_vector_dsub(BMAS_dvec a, BMAS_dvec b){return _mm256_sub_pd(a, b);}
BMAS_dvec static inline BMAS_vector_dmul(BMAS_dvec a, BMAS_dvec b){return _mm256_mul_pd(a, b);}
BMAS_dvec static inline BMAS_vector_ddiv(BMAS_dvec a, BMAS_dvec b){return _mm256_div_pd(a, b);}

// integer insert and extract - we need to do this because compiler needs "immediates"
// TODO: Rewrite using something like BOOST Preprocessor Library?

BMAS_ivec static inline BMAS_ivec_make_i64(int64_t* ptr, const int stride){
  BMAS_ivec v;
  v = _mm256_insert_epi64(v, ptr[0*stride], 0);
  v = _mm256_insert_epi64(v, ptr[1*stride], 1);
  v = _mm256_insert_epi64(v, ptr[2*stride], 2);
  v = _mm256_insert_epi64(v, ptr[3*stride], 3);
  return v;
}

BMAS_ivec static inline BMAS_ivec_make_i32(int32_t* ptr, const int stride){
  BMAS_ivec v;
  const long cstride = stride*4;
  v = _mm256_insert_epi32(v, ptr[0*stride], 0);
  v = _mm256_insert_epi32(v, ptr[1*stride], 1);
  v = _mm256_insert_epi32(v, ptr[2*stride], 2);
  v = _mm256_insert_epi32(v, ptr[3*stride], 3);
  v = _mm256_insert_epi32(v, ptr[4*stride], 4);
  v = _mm256_insert_epi32(v, ptr[5*stride], 5);
  v = _mm256_insert_epi32(v, ptr[6*stride], 6);
  v = _mm256_insert_epi32(v, ptr[7*stride], 7);
  return v;
}

BMAS_ivec static inline BMAS_ivec_make_i16(int16_t* ptr, const int stride){
  BMAS_ivec v;
  v = _mm256_insert_epi16(v, ptr[0*stride], 0);
  v = _mm256_insert_epi16(v, ptr[1*stride], 1);
  v = _mm256_insert_epi16(v, ptr[2*stride], 2);
  v = _mm256_insert_epi16(v, ptr[3*stride], 3);
  v = _mm256_insert_epi16(v, ptr[4*stride], 4);
  v = _mm256_insert_epi16(v, ptr[5*stride], 5);
  v = _mm256_insert_epi16(v, ptr[6*stride], 6);
  v = _mm256_insert_epi16(v, ptr[7*stride], 7);
  v = _mm256_insert_epi16(v, ptr[8*stride], 8);
  v = _mm256_insert_epi16(v, ptr[9*stride], 9);
  v = _mm256_insert_epi16(v, ptr[10*stride], 10);
  v = _mm256_insert_epi16(v, ptr[11*stride], 11);
  v = _mm256_insert_epi16(v, ptr[12*stride], 12);
  v = _mm256_insert_epi16(v, ptr[13*stride], 13);
  v = _mm256_insert_epi16(v, ptr[14*stride], 14);
  v = _mm256_insert_epi16(v, ptr[15*stride], 15);
  return v;
}

BMAS_ivec static inline BMAS_ivec_make_i8(int8_t* ptr, const int stride){
  BMAS_ivec v;
  v = _mm256_insert_epi8(v, ptr[0*stride],  0);
  v = _mm256_insert_epi8(v, ptr[1*stride],  1);
  v = _mm256_insert_epi8(v, ptr[2*stride],  2);
  v = _mm256_insert_epi8(v, ptr[3*stride],  3);
  v = _mm256_insert_epi8(v, ptr[4*stride],  4);
  v = _mm256_insert_epi8(v, ptr[5*stride],  5);
  v = _mm256_insert_epi8(v, ptr[6*stride],  6);
  v = _mm256_insert_epi8(v, ptr[7*stride],  7);
  v = _mm256_insert_epi8(v, ptr[8*stride],  8);
  v = _mm256_insert_epi8(v, ptr[9*stride],  9);
  v = _mm256_insert_epi8(v, ptr[10*stride], 10);
  v = _mm256_insert_epi8(v, ptr[11*stride], 11);
  v = _mm256_insert_epi8(v, ptr[12*stride], 12);
  v = _mm256_insert_epi8(v, ptr[13*stride], 13);
  v = _mm256_insert_epi8(v, ptr[14*stride], 14);
  v = _mm256_insert_epi8(v, ptr[15*stride], 15);
  v = _mm256_insert_epi8(v, ptr[16*stride], 16);
  v = _mm256_insert_epi8(v, ptr[17*stride], 17);
  v = _mm256_insert_epi8(v, ptr[18*stride], 18);
  v = _mm256_insert_epi8(v, ptr[19*stride], 19);
  v = _mm256_insert_epi8(v, ptr[20*stride], 20);
  v = _mm256_insert_epi8(v, ptr[21*stride], 21);
  v = _mm256_insert_epi8(v, ptr[22*stride], 22);
  v = _mm256_insert_epi8(v, ptr[23*stride], 23);
  v = _mm256_insert_epi8(v, ptr[24*stride], 24);
  v = _mm256_insert_epi8(v, ptr[25*stride], 25);
  v = _mm256_insert_epi8(v, ptr[26*stride], 26);
  v = _mm256_insert_epi8(v, ptr[27*stride], 27);
  v = _mm256_insert_epi8(v, ptr[28*stride], 28);
  v = _mm256_insert_epi8(v, ptr[29*stride], 29);
  v = _mm256_insert_epi8(v, ptr[30*stride], 30);
  v = _mm256_insert_epi8(v, ptr[31*stride], 31);
  return v;
}

BMAS_ivech static inline BMAS_ivech_make_i64(int64_t* ptr, const int stride){
  BMAS_ivech v;
  v = _mm_insert_epi64(v, ptr[0*stride], 0);
  v = _mm_insert_epi64(v, ptr[1*stride], 1);
  return v;
}

BMAS_ivech static inline BMAS_ivech_make_i32(int32_t* ptr, const int stride){
  BMAS_ivech v;
  const long cstride = stride*4;
  v = _mm_insert_epi32(v, ptr[0*stride], 0);
  v = _mm_insert_epi32(v, ptr[1*stride], 1);
  v = _mm_insert_epi32(v, ptr[2*stride], 2);
  v = _mm_insert_epi32(v, ptr[3*stride], 3);
  return v;
}

BMAS_ivech static inline BMAS_ivech_make_i16(int16_t* ptr, const int stride){
  BMAS_ivech v;
  v = _mm_insert_epi16(v, ptr[0*stride], 0);
  v = _mm_insert_epi16(v, ptr[1*stride], 1);
  v = _mm_insert_epi16(v, ptr[2*stride], 2);
  v = _mm_insert_epi16(v, ptr[3*stride], 3);
  v = _mm_insert_epi16(v, ptr[4*stride], 4);
  v = _mm_insert_epi16(v, ptr[5*stride], 5);
  v = _mm_insert_epi16(v, ptr[6*stride], 6);
  v = _mm_insert_epi16(v, ptr[7*stride], 7);
  return v;
}

BMAS_ivech static inline BMAS_ivech_make_i8(int8_t* ptr, const int stride){
  BMAS_ivech v;
  v = _mm_insert_epi8(v, ptr[0*stride],  0);
  v = _mm_insert_epi8(v, ptr[1*stride],  1);
  v = _mm_insert_epi8(v, ptr[2*stride],  2);
  v = _mm_insert_epi8(v, ptr[3*stride],  3);
  v = _mm_insert_epi8(v, ptr[4*stride],  4);
  v = _mm_insert_epi8(v, ptr[5*stride],  5);
  v = _mm_insert_epi8(v, ptr[6*stride],  6);
  v = _mm_insert_epi8(v, ptr[7*stride],  7);
  v = _mm_insert_epi8(v, ptr[8*stride],  8);
  v = _mm_insert_epi8(v, ptr[9*stride],  9);
  v = _mm_insert_epi8(v, ptr[10*stride], 10);
  v = _mm_insert_epi8(v, ptr[11*stride], 11);
  v = _mm_insert_epi8(v, ptr[12*stride], 12);
  v = _mm_insert_epi8(v, ptr[13*stride], 13);
  v = _mm_insert_epi8(v, ptr[14*stride], 14);
  v = _mm_insert_epi8(v, ptr[15*stride], 15);
  return v;
}


void static inline BMAS_ivec_store_multi_i64(BMAS_ivec v, int64_t* ptr, const int stride){
  ptr[0*stride] = _mm256_extract_epi64(v, 0);
  ptr[1*stride] = _mm256_extract_epi64(v, 1);
  ptr[2*stride] = _mm256_extract_epi64(v, 2);
  ptr[3*stride] = _mm256_extract_epi64(v, 3);
}

void static inline BMAS_ivec_store_multi_i32(BMAS_ivec v, int32_t* ptr, const int stride){
  ptr[0*stride] = _mm256_extract_epi32(v, 0);
  ptr[1*stride] = _mm256_extract_epi32(v, 1);
  ptr[2*stride] = _mm256_extract_epi32(v, 2);
  ptr[3*stride] = _mm256_extract_epi32(v, 3);
  ptr[4*stride] = _mm256_extract_epi32(v, 4);
  ptr[5*stride] = _mm256_extract_epi32(v, 5);
  ptr[6*stride] = _mm256_extract_epi32(v, 6);
  ptr[7*stride] = _mm256_extract_epi32(v, 7);
}

void static inline BMAS_ivec_store_multi_i16(BMAS_ivec v, int16_t* ptr, const int stride){
  ptr[0*stride]  = _mm256_extract_epi16(v, 0);
  ptr[1*stride]  = _mm256_extract_epi16(v, 1);
  ptr[2*stride]  = _mm256_extract_epi16(v, 2);
  ptr[3*stride]  = _mm256_extract_epi16(v, 3);
  ptr[4*stride]  = _mm256_extract_epi16(v, 4);
  ptr[5*stride]  = _mm256_extract_epi16(v, 5);
  ptr[6*stride]  = _mm256_extract_epi16(v, 6);
  ptr[7*stride]  = _mm256_extract_epi16(v, 7);
  ptr[8*stride]  = _mm256_extract_epi16(v, 8);
  ptr[9*stride]  = _mm256_extract_epi16(v, 9);
  ptr[10*stride] = _mm256_extract_epi16(v, 10);
  ptr[11*stride] = _mm256_extract_epi16(v, 11);
  ptr[12*stride] = _mm256_extract_epi16(v, 12);
  ptr[13*stride] = _mm256_extract_epi16(v, 13);
  ptr[14*stride] = _mm256_extract_epi16(v, 14);
  ptr[15*stride] = _mm256_extract_epi16(v, 15);
}

void static inline BMAS_ivec_store_multi_i8(BMAS_ivec v, int8_t* ptr, const int stride){
  ptr[0*stride]  = _mm256_extract_epi8(v, 0);
  ptr[1*stride]  = _mm256_extract_epi8(v, 1);
  ptr[2*stride]  = _mm256_extract_epi8(v, 2);
  ptr[3*stride]  = _mm256_extract_epi8(v, 3);
  ptr[4*stride]  = _mm256_extract_epi8(v, 4);
  ptr[5*stride]  = _mm256_extract_epi8(v, 5);
  ptr[6*stride]  = _mm256_extract_epi8(v, 6);
  ptr[7*stride]  = _mm256_extract_epi8(v, 7);
  ptr[8*stride]  = _mm256_extract_epi8(v, 8);
  ptr[9*stride]  = _mm256_extract_epi8(v, 9);
  ptr[10*stride] = _mm256_extract_epi8(v, 10);
  ptr[11*stride] = _mm256_extract_epi8(v, 11);
  ptr[12*stride] = _mm256_extract_epi8(v, 12);
  ptr[13*stride] = _mm256_extract_epi8(v, 13);
  ptr[14*stride] = _mm256_extract_epi8(v, 14);
  ptr[15*stride] = _mm256_extract_epi8(v, 15);
  ptr[16*stride] = _mm256_extract_epi8(v, 16);
  ptr[17*stride] = _mm256_extract_epi8(v, 17);
  ptr[18*stride] = _mm256_extract_epi8(v, 18);
  ptr[19*stride] = _mm256_extract_epi8(v, 19);
  ptr[20*stride] = _mm256_extract_epi8(v, 20);
  ptr[21*stride] = _mm256_extract_epi8(v, 21);
  ptr[22*stride] = _mm256_extract_epi8(v, 22);
  ptr[23*stride] = _mm256_extract_epi8(v, 23);
  ptr[24*stride] = _mm256_extract_epi8(v, 24);
  ptr[25*stride] = _mm256_extract_epi8(v, 25);
  ptr[26*stride] = _mm256_extract_epi8(v, 26);
  ptr[27*stride] = _mm256_extract_epi8(v, 27);
  ptr[28*stride] = _mm256_extract_epi8(v, 28);
  ptr[29*stride] = _mm256_extract_epi8(v, 29);
  ptr[30*stride] = _mm256_extract_epi8(v, 30);
  ptr[31*stride] = _mm256_extract_epi8(v, 31);
}


// integer arithmetic
BMAS_ivec static inline BMAS_vector_i64add(BMAS_ivec a, BMAS_ivec b){return _mm256_add_epi64(a, b);}
BMAS_ivec static inline BMAS_vector_i32add(BMAS_ivec a, BMAS_ivec b){return _mm256_add_epi32(a, b);}
BMAS_ivec static inline BMAS_vector_i16add(BMAS_ivec a, BMAS_ivec b){return _mm256_add_epi16(a, b);}
BMAS_ivec static inline BMAS_vector_i8add (BMAS_ivec a, BMAS_ivec b){return _mm256_add_epi8(a, b);}

BMAS_ivec static inline BMAS_vector_i64sub(BMAS_ivec a, BMAS_ivec b){return _mm256_sub_epi64(a, b);}
BMAS_ivec static inline BMAS_vector_i32sub(BMAS_ivec a, BMAS_ivec b){return _mm256_sub_epi32(a, b);}
BMAS_ivec static inline BMAS_vector_i16sub(BMAS_ivec a, BMAS_ivec b){return _mm256_sub_epi16(a, b);}
BMAS_ivec static inline BMAS_vector_i8sub (BMAS_ivec a, BMAS_ivec b){return _mm256_sub_epi8(a, b);}

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

