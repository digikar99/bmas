
typedef __m256 BMAS_svec;
typedef __m128 BMAS_svech; // half
typedef __m256d BMAS_dvec;
typedef __m128d BMAS_dvech; // half
typedef __m256 BMAS_sbool;
typedef __m256d BMAS_dbool;
typedef __m256i BMAS_ivec;
typedef __m128i BMAS_ivech;

#define SIMD_SINGLE_STRIDE 8
#define SIMD_DOUBLE_STRIDE 4

BMAS_svec static inline BMAS_vector_szero(){return _mm256_setzero_ps();}
BMAS_dvec static inline BMAS_vector_dzero(){return _mm256_setzero_pd();}
BMAS_ivec static inline BMAS_vector_izero(){return _mm256_setzero_si256();}

BMAS_svec static inline BMAS_vector_sMIN(){
  return _mm256_set_ps(-FLT_MAX,-FLT_MAX,-FLT_MAX,-FLT_MAX,-FLT_MAX,-FLT_MAX,-FLT_MAX,-FLT_MAX);
}
BMAS_dvec static inline BMAS_vector_dMIN(){return _mm256_set_pd(-DBL_MAX,-DBL_MAX,-DBL_MAX,-DBL_MAX);}
BMAS_ivec static inline BMAS_vector_i64MIN(){
  return _mm256_set_epi64x(0x8000000000000000, 0x8000000000000000,
                           0x8000000000000000, 0x8000000000000000);
}
BMAS_ivec static inline BMAS_vector_i32MIN(){
  return _mm256_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000,
                          0x80000000, 0x80000000, 0x80000000, 0x80000000);
}
BMAS_ivec static inline BMAS_vector_i16MIN(){
  return _mm256_set_epi32(0x80008000, 0x80008000, 0x80008000, 0x80008000,
                          0x80008000, 0x80008000, 0x80008000, 0x80008000);
}
BMAS_ivec static inline BMAS_vector_i8MIN(){
  return _mm256_set_epi32(0x80808080, 0x80808080, 0x80808080, 0x80808080,
                          0x80808080, 0x80808080, 0x80808080, 0x80808080);
}
BMAS_ivec static inline BMAS_vector_uMIN(){return _mm256_set_epi64x(0,0,0,0);}

BMAS_svec static inline BMAS_vector_sMAX(){
  return _mm256_set_ps(FLT_MAX,FLT_MAX,FLT_MAX,FLT_MAX,FLT_MAX,FLT_MAX,FLT_MAX,FLT_MAX);
}
BMAS_dvec static inline BMAS_vector_dMAX(){return _mm256_set_pd(DBL_MAX,DBL_MAX,DBL_MAX,DBL_MAX);}
BMAS_ivec static inline BMAS_vector_i64MAX(){
  return _mm256_set_epi64x(0x7fffffffffffffff, 0x7fffffffffffffff,
                           0x7fffffffffffffff, 0x7fffffffffffffff);
}
BMAS_ivec static inline BMAS_vector_i32MAX(){
  return _mm256_set_epi32(0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff,
                          0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff);
}
BMAS_ivec static inline BMAS_vector_i16MAX(){
  return _mm256_set_epi32(0x7fff7fff, 0x7fff7fff, 0x7fff7fff, 0x7fff7fff,
                          0x7fff7fff, 0x7fff7fff, 0x7fff7fff, 0x7fff7fff);
}
BMAS_ivec static inline BMAS_vector_i8MAX(){
  return _mm256_set_epi32(0x7f7f7f7f, 0x7f7f7f7f, 0x7f7f7f7f, 0x7f7f7f7f,
                          0x7f7f7f7f, 0x7f7f7f7f, 0x7f7f7f7f, 0x7f7f7f7f);
}
BMAS_ivec static inline BMAS_vector_uMAX(){
  return _mm256_set_epi64x(0xffffffffffffffff,0xffffffffffffffff,
                           0xffffffffffffffff,0xffffffffffffffff);
}

// store-load

BMAS_svec static inline BMAS_svec_load(float* ptr){ return _mm256_loadu_ps(ptr); }
void static inline BMAS_svec_store(float* ptr, BMAS_svec v){ return _mm256_storeu_ps(ptr, v); }

BMAS_dvec static inline BMAS_dvec_load(double* ptr){ return _mm256_loadu_pd(ptr); }
void static inline BMAS_dvec_store(double* ptr, BMAS_dvec v){ return _mm256_storeu_pd(ptr, v); }

void static inline BMAS_svec_store_bool(_Bool* ptr, const long stride, BMAS_sbool v, const int elt_size){
  const int simd_len = 256 / elt_size;
  for(int i=0; i<simd_len; i++) (ptr+i*stride)[0] = v[i];
}

void static inline BMAS_svec_store_boolx4(_Bool* ptr,
                                          BMAS_sbool v1, BMAS_sbool v2,
                                          BMAS_sbool v3, BMAS_sbool v4,
                                          const int elt_size){
  BMAS_ivec vi1 = _mm256_castps_si256(v1);
  BMAS_ivec vi2 = _mm256_castps_si256(v2);
  BMAS_ivec vi3 = _mm256_castps_si256(v3);
  BMAS_ivec vi4 = _mm256_castps_si256(v4);

  // Packing on AVX2 interleaves the registers
  BMAS_ivec vi5 = _mm256_packs_epi32(vi1, vi2);
  vi5 = _mm256_permute4x64_epi64(vi5, 0b11011000);
  BMAS_ivec vi6 = _mm256_packs_epi32(vi3, vi4);
  vi6 = _mm256_permute4x64_epi64(vi6, 0b11011000);

  BMAS_ivec vi = _mm256_packs_epi16(vi5, vi6);
  vi = _mm256_permute4x64_epi64(vi, 0b11011000);

  vi = _mm256_and_si256(vi,
                        _mm256_set_epi32(0x01010101, 0x01010101, 0x01010101, 0x01010101,
                                         0x01010101, 0x01010101, 0x01010101, 0x01010101));
  _mm256_storeu_si256((__m256i*)(ptr), vi);
}

void static inline BMAS_dvec_store_bool(_Bool* ptr, const long stride, BMAS_dbool v, const int elt_size){
  const int simd_len = 256 / elt_size;
  for(int i=0; i<simd_len; i++) (ptr+i*stride)[0] = v[i];
}

void static inline BMAS_dvec_store_boolx4(_Bool* ptr,
                                          BMAS_dbool v1, BMAS_dbool v2,
                                          BMAS_dbool v3, BMAS_dbool v4,
                                          const int elt_size){
  // Permute to shift to lower 128 bits and then extract
  BMAS_ivec idx = _mm256_set_epi32(7, 7, 7, 7, 6, 4, 2, 0);
  BMAS_ivec vi1 = _mm256_castpd_si256(v1);
  BMAS_ivec vi2 = _mm256_castpd_si256(v2);
  BMAS_ivec vi3 = _mm256_castpd_si256(v3);
  BMAS_ivec vi4 = _mm256_castpd_si256(v4);
  vi1 = _mm256_permutevar8x32_epi32(vi1, idx);
  vi2 = _mm256_permutevar8x32_epi32(vi2, idx);
  vi3 = _mm256_permutevar8x32_epi32(vi3, idx);
  vi4 = _mm256_permutevar8x32_epi32(vi4, idx);

  BMAS_ivech vih1 = _mm256_castsi256_si128(vi1);
  BMAS_ivech vih2 = _mm256_castsi256_si128(vi2);
  BMAS_ivech vih3 = _mm256_castsi256_si128(vi3);
  BMAS_ivech vih4 = _mm256_castsi256_si128(vi4);

  BMAS_ivech vih5 = _mm_packs_epi32(vih1, vih2);
  BMAS_ivech vih6 = _mm_packs_epi32(vih3, vih4);

  BMAS_ivech vih = _mm_packs_epi16(vih5, vih6);
  vih = _mm_and_si128(vih, _mm_set_epi32(0x01010101, 0x01010101, 0x01010101, 0x01010101));
  _mm_storeu_si128((__m128i*)(ptr), vih);
}

void static inline BMAS_ivec_store_boolx1(_Bool* ptr, BMAS_ivec v){
  v = _mm256_and_si256(v,
                       _mm256_set_epi32(0x01010101, 0x01010101, 0x01010101, 0x01010101,
                                        0x01010101, 0x01010101, 0x01010101, 0x01010101));
  _mm256_storeu_si256((__m256i*)ptr, v);
}
void static inline BMAS_ivec_store_boolx2(_Bool* ptr, BMAS_ivec v1, BMAS_ivec v2){
  // It is implicit that v1 and v2 contain 16 bit integers;
  // so that 32 8-bit Bools would make up 2 registers of 16 bit integers
  BMAS_ivec v = _mm256_packs_epi16(v1, v2);
  v = _mm256_permute4x64_epi64(v, 0b11011000);
  v = _mm256_and_si256(v,
                       _mm256_set_epi32(0x01010101, 0x01010101, 0x01010101, 0x01010101,
                                        0x01010101, 0x01010101, 0x01010101, 0x01010101));
  _mm256_storeu_si256((__m256i*)ptr, v);
}
void static inline BMAS_ivec_store_boolx4(_Bool* ptr, BMAS_ivec v1, BMAS_ivec v2, BMAS_ivec v3, BMAS_ivec v4, const int elt_size){
  if (elt_size == 4){
    // Packing on AVX2 interleaves the registers
    BMAS_ivec v5 = _mm256_packs_epi32(v1, v2);
    v5 = _mm256_permute4x64_epi64(v5, 0b11011000);
    BMAS_ivec v6 = _mm256_packs_epi32(v3, v4);
    v6 = _mm256_permute4x64_epi64(v6, 0b11011000);
    BMAS_ivec v = _mm256_packs_epi16(v5, v6);
    v = _mm256_permute4x64_epi64(v, 0b11011000);
    v = _mm256_and_si256(v,
                         _mm256_set_epi32(0x01010101, 0x01010101, 0x01010101, 0x01010101,
                                          0x01010101, 0x01010101, 0x01010101, 0x01010101));
    _mm256_storeu_si256((__m256i*)ptr, v);
  }else{ // elt_size == 8
    // Permute to shift to lower 128 bits and then extract
    BMAS_ivec idx = _mm256_set_epi32(7, 7, 7, 7, 6, 4, 2, 0);
    v1 = _mm256_permutevar8x32_epi32(v1, idx);
    v2 = _mm256_permutevar8x32_epi32(v2, idx);
    v3 = _mm256_permutevar8x32_epi32(v3, idx);
    v4 = _mm256_permutevar8x32_epi32(v4, idx);

    BMAS_ivech vh1 = _mm256_castsi256_si128(v1);
    BMAS_ivech vh2 = _mm256_castsi256_si128(v2);
    BMAS_ivech vh3 = _mm256_castsi256_si128(v3);
    BMAS_ivech vh4 = _mm256_castsi256_si128(v4);

    BMAS_ivech vh5 = _mm_packs_epi32(vh1, vh2);
    BMAS_ivech vh6 = _mm_packs_epi32(vh3, vh4);

    BMAS_ivech vh = _mm_packs_epi16(vh5, vh6);
    vh = _mm_and_si128(vh, _mm_set_epi32(0x01010101, 0x01010101, 0x01010101, 0x01010101));
    _mm_storeu_si128((__m128i*)(ptr), vh);
  }
}

#define V_EXTRACT64_AND_SET_BOOL_PTR(I) ptr[I*stride] = (_Bool)_mm256_extract_epi64(v, I);
#define V_EXTRACT32_AND_SET_BOOL_PTR(I) ptr[I*stride] = (_Bool)_mm256_extract_epi32(v, I);
#define V_EXTRACT16_AND_SET_BOOL_PTR(I) ptr[I*stride] = (_Bool)_mm256_extract_epi16(v, I);
#define V_EXTRACT8_AND_SET_BOOL_PTR(I)  ptr[I*stride] = (_Bool)_mm256_extract_epi8 (v, I);

void static inline BMAS_ivec_store_bool(_Bool* ptr, const long stride, BMAS_ivec v, const int elt_size){
  if (elt_size == 8){
    // 64 bit integers
    MACRO_LOOP(4, V_EXTRACT64_AND_SET_BOOL_PTR);
  }else if(elt_size == 4){
    MACRO_LOOP(8, V_EXTRACT32_AND_SET_BOOL_PTR);
  }else if(elt_size == 2){
    MACRO_LOOP(16, V_EXTRACT16_AND_SET_BOOL_PTR);
  }else if(elt_size == 1){
    MACRO_LOOP(32, V_EXTRACT8_AND_SET_BOOL_PTR);
  }
}

#undef V_EXTRACT64_AND_SET_BOOL_PTR
#undef V_EXTRACT32_AND_SET_BOOL_PTR
#undef V_EXTRACT16_AND_SET_BOOL_PTR
#undef V_EXTRACT8_AND_SET_BOOL_PTR

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
// Credits for 64-bit conversions: user wim from https://stackoverflow.com/a/41223013/8957330
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

// integer scalar bitshift

BMAS_ivec static inline BMAS_vector_i64sra(BMAS_ivec v, BMAS_ivec count){
  BMAS_ivec gt0_shift  = _mm256_srlv_epi64(v, count); // logical shift

  BMAS_ivec identity   = _mm256_cmpeq_epi8(v, v);

  BMAS_ivec lt0_comp   = _mm256_xor_si256(identity, v);
  BMAS_ivec lt0_shiftc = _mm256_srlv_epi64(lt0_comp, count);
  BMAS_ivec lt0_shift  = _mm256_xor_si256(identity, lt0_shiftc);

  BMAS_ivec lt0_mask   = _mm256_cmpgt_epi64(_mm256_setzero_si256(), v);
  BMAS_ivec result = _mm256_or_si256(_mm256_and_si256(lt0_mask, lt0_shift),
                                     _mm256_andnot_si256(lt0_mask, gt0_shift));
  return result;
}
BMAS_ivec static inline BMAS_vector_i32sra(BMAS_ivec v, BMAS_ivec count){return _mm256_srav_epi32(v, count);}
BMAS_ivec static inline BMAS_vector_i16sra(BMAS_ivec v, BMAS_ivec count){
  BMAS_ivech v1 = _mm256_extracti128_si256(v, 1);
  BMAS_ivech v2 = _mm256_castsi256_si128(v);
  BMAS_ivech c1 = _mm256_extracti128_si256(count, 1);
  BMAS_ivech c2 = _mm256_castsi256_si128(count);
  BMAS_ivec v1ex = _mm256_cvtepi16_epi32(v1);
  BMAS_ivec v2ex = _mm256_cvtepi16_epi32(v2);
  BMAS_ivec c1ex = _mm256_cvtepi16_epi32(c1);
  BMAS_ivec c2ex = _mm256_cvtepi16_epi32(c2);
  BMAS_ivec v1ex_shift = BMAS_vector_i32sra(v1ex, c1ex);
  BMAS_ivec v2ex_shift = BMAS_vector_i32sra(v2ex, c2ex);
  BMAS_ivec result_permuted = _mm256_packs_epi32(v1ex_shift, v2ex_shift);
  BMAS_ivec result = _mm256_permute4x64_epi64(result_permuted, 0b11011000);
  return result;
}
BMAS_ivec static inline BMAS_vector_i8sra(BMAS_ivec v, BMAS_ivec count){
  BMAS_ivech v1 = _mm256_extracti128_si256(v, 1);
  BMAS_ivech v2 = _mm256_castsi256_si128(v);
  BMAS_ivech c1 = _mm256_extracti128_si256(count, 1);
  BMAS_ivech c2 = _mm256_castsi256_si128(count);
  BMAS_ivec v1ex = _mm256_cvtepi8_epi16(v1);
  BMAS_ivec v2ex = _mm256_cvtepi8_epi16(v2);
  BMAS_ivec c1ex = _mm256_cvtepi8_epi16(c1);
  BMAS_ivec c2ex = _mm256_cvtepi8_epi16(c2);
  BMAS_ivec v1ex_shift = BMAS_vector_i16sra(v1ex, c1ex);
  BMAS_ivec v2ex_shift = BMAS_vector_i16sra(v2ex, c2ex);
  BMAS_ivec result_permuted = _mm256_packs_epi16(v1ex_shift, v2ex_shift);
  BMAS_ivec result = _mm256_permute4x64_epi64(result_permuted, 0b11011000);
  return result;
}

BMAS_ivec static inline BMAS_vector_u64srl(BMAS_ivec v, BMAS_ivec count){return _mm256_srlv_epi64(v, count);}
BMAS_ivec static inline BMAS_vector_u32srl(BMAS_ivec v, BMAS_ivec count){return _mm256_srlv_epi32(v, count);}
BMAS_ivec static inline BMAS_vector_u16srl(BMAS_ivec v, BMAS_ivec count){
  BMAS_ivech v1 = _mm256_extracti128_si256(v, 1);
  BMAS_ivech v2 = _mm256_castsi256_si128(v);
  BMAS_ivech c1 = _mm256_extracti128_si256(count, 1);
  BMAS_ivech c2 = _mm256_castsi256_si128(count);
  BMAS_ivec v1ex = _mm256_cvtepi16_epi32(v1);
  BMAS_ivec v2ex = _mm256_cvtepi16_epi32(v2);
  BMAS_ivec c1ex = _mm256_cvtepi16_epi32(c1);
  BMAS_ivec c2ex = _mm256_cvtepi16_epi32(c2);
  BMAS_ivec v1ex_shift = BMAS_vector_u32srl(v1ex, c1ex);
  BMAS_ivec v2ex_shift = BMAS_vector_u32srl(v2ex, c2ex);
  BMAS_ivec result_permuted = _mm256_packs_epi32(v1ex_shift, v2ex_shift);
  BMAS_ivec result = _mm256_permute4x64_epi64(result_permuted, 0b11011000);
  return result;
}
BMAS_ivec static inline BMAS_vector_u8srl(BMAS_ivec v, BMAS_ivec count){
  BMAS_ivech v1 = _mm256_extracti128_si256(v, 1);
  BMAS_ivech v2 = _mm256_castsi256_si128(v);
  BMAS_ivech c1 = _mm256_extracti128_si256(count, 1);
  BMAS_ivech c2 = _mm256_castsi256_si128(count);
  BMAS_ivec v1ex = _mm256_cvtepi8_epi16(v1);
  BMAS_ivec v2ex = _mm256_cvtepi8_epi16(v2);
  BMAS_ivec c1ex = _mm256_cvtepi8_epi16(c1);
  BMAS_ivec c2ex = _mm256_cvtepi8_epi16(c2);
  BMAS_ivec v1ex_shift = BMAS_vector_u16srl(v1ex, c1ex);
  BMAS_ivec v2ex_shift = BMAS_vector_u16srl(v2ex, c2ex);
  BMAS_ivec result_permuted = _mm256_packs_epi16(v1ex_shift, v2ex_shift);
  BMAS_ivec result = _mm256_permute4x64_epi64(result_permuted, 0b11011000);
  return result;
}

BMAS_ivec static inline BMAS_vector_u64sll(BMAS_ivec v, BMAS_ivec count){return _mm256_sllv_epi64(v, count);}
BMAS_ivec static inline BMAS_vector_u32sll(BMAS_ivec v, BMAS_ivec count){return _mm256_sllv_epi32(v, count);}
BMAS_ivec static inline BMAS_vector_u16sll(BMAS_ivec v, BMAS_ivec count){
  BMAS_ivech v1 = _mm256_extracti128_si256(v, 1);
  BMAS_ivech v2 = _mm256_castsi256_si128(v);
  BMAS_ivech c1 = _mm256_extracti128_si256(count, 1);
  BMAS_ivech c2 = _mm256_castsi256_si128(count);
  BMAS_ivec v1ex = _mm256_cvtepi16_epi32(v1);
  BMAS_ivec v2ex = _mm256_cvtepi16_epi32(v2);
  BMAS_ivec c1ex = _mm256_cvtepi16_epi32(c1);
  BMAS_ivec c2ex = _mm256_cvtepi16_epi32(c2);
  BMAS_ivec v1ex_shift = BMAS_vector_u32sll(v1ex, c1ex);
  BMAS_ivec v2ex_shift = BMAS_vector_u32sll(v2ex, c2ex);
  BMAS_ivec result_permuted = _mm256_packs_epi32(v1ex_shift, v2ex_shift);
  BMAS_ivec result = _mm256_permute4x64_epi64(result_permuted, 0b11011000);
  return result;
}
BMAS_ivec static inline BMAS_vector_u8sll(BMAS_ivec v, BMAS_ivec count){
  BMAS_ivech v1 = _mm256_extracti128_si256(v, 1);
  BMAS_ivech v2 = _mm256_castsi256_si128(v);
  BMAS_ivech c1 = _mm256_extracti128_si256(count, 1);
  BMAS_ivech c2 = _mm256_castsi256_si128(count);
  BMAS_ivec v1ex = _mm256_cvtepi8_epi16(v1);
  BMAS_ivec v2ex = _mm256_cvtepi8_epi16(v2);
  BMAS_ivec c1ex = _mm256_cvtepi8_epi16(c1);
  BMAS_ivec c2ex = _mm256_cvtepi8_epi16(c2);
  BMAS_ivec v1ex_shift = BMAS_vector_u16sll(v1ex, c1ex);
  BMAS_ivec v2ex_shift = BMAS_vector_u16sll(v2ex, c2ex);
  BMAS_ivec result_permuted = _mm256_packs_epi16(v1ex_shift, v2ex_shift);
  BMAS_ivec result = _mm256_permute4x64_epi64(result_permuted, 0b11011000);
  return result;
}


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

// boolean logical
BMAS_ivec static inline BMAS_vector_i8and(BMAS_ivec a, BMAS_ivec b){return _mm256_and_si256(a, b);}
BMAS_ivec static inline BMAS_vector_i8or (BMAS_ivec a, BMAS_ivec b){return _mm256_or_si256(a, b);}
BMAS_ivec static inline BMAS_vector_i8xor(BMAS_ivec a, BMAS_ivec b){return _mm256_xor_si256(a, b);}
BMAS_ivec static inline BMAS_vector_i8not(BMAS_ivec a){
  // intel's page lists epi8 as having a minimal latency-throughtput
  return _mm256_xor_si256(a, _mm256_cmpeq_epi8(a, a));
}
BMAS_ivec static inline BMAS_vector_i8andnot(BMAS_ivec a, BMAS_ivec b){return _mm256_andnot_si256(a, b);}


// integer comparison


BMAS_ivec static inline BMAS_vector_i8lt (BMAS_ivec a, BMAS_ivec b){return _mm256_cmpgt_epi8(b, a);}
BMAS_ivec static inline BMAS_vector_i8le (BMAS_ivec a, BMAS_ivec b){return BMAS_vector_i8not(_mm256_cmpgt_epi8(a, b));}
BMAS_ivec static inline BMAS_vector_i8eq (BMAS_ivec a, BMAS_ivec b){return _mm256_cmpeq_epi8(a, b);}
BMAS_ivec static inline BMAS_vector_i8neq(BMAS_ivec a, BMAS_ivec b){return BMAS_vector_i8not(_mm256_cmpeq_epi8(a, b));}
BMAS_ivec static inline BMAS_vector_i8gt (BMAS_ivec a, BMAS_ivec b){return _mm256_cmpgt_epi8(a, b);}
BMAS_ivec static inline BMAS_vector_i8ge (BMAS_ivec a, BMAS_ivec b){return BMAS_vector_i8not(_mm256_cmpgt_epi8(b, a));}

BMAS_ivec static inline BMAS_vector_i16lt (BMAS_ivec a, BMAS_ivec b){return _mm256_cmpgt_epi16(b, a);}
BMAS_ivec static inline BMAS_vector_i16le (BMAS_ivec a, BMAS_ivec b){return BMAS_vector_i8not(_mm256_cmpgt_epi16(a, b));}
BMAS_ivec static inline BMAS_vector_i16eq (BMAS_ivec a, BMAS_ivec b){return _mm256_cmpeq_epi16(a, b);}
BMAS_ivec static inline BMAS_vector_i16neq(BMAS_ivec a, BMAS_ivec b){return BMAS_vector_i8not(_mm256_cmpeq_epi16(a, b));}
BMAS_ivec static inline BMAS_vector_i16gt (BMAS_ivec a, BMAS_ivec b){return _mm256_cmpgt_epi16(a, b);}
BMAS_ivec static inline BMAS_vector_i16ge (BMAS_ivec a, BMAS_ivec b){return BMAS_vector_i8not(_mm256_cmpgt_epi16(b, a));}

BMAS_ivec static inline BMAS_vector_i32lt (BMAS_ivec a, BMAS_ivec b){return _mm256_cmpgt_epi32(b, a);}
BMAS_ivec static inline BMAS_vector_i32le (BMAS_ivec a, BMAS_ivec b){return BMAS_vector_i8not(_mm256_cmpgt_epi32(a, b));}
BMAS_ivec static inline BMAS_vector_i32eq (BMAS_ivec a, BMAS_ivec b){return _mm256_cmpeq_epi32(a, b);}
BMAS_ivec static inline BMAS_vector_i32neq(BMAS_ivec a, BMAS_ivec b){return BMAS_vector_i8not(_mm256_cmpeq_epi32(a, b));}
BMAS_ivec static inline BMAS_vector_i32gt (BMAS_ivec a, BMAS_ivec b){return _mm256_cmpgt_epi32(a, b);}
BMAS_ivec static inline BMAS_vector_i32ge (BMAS_ivec a, BMAS_ivec b){return BMAS_vector_i8not(_mm256_cmpgt_epi32(b, a));}

BMAS_ivec static inline BMAS_vector_i64lt (BMAS_ivec a, BMAS_ivec b){return _mm256_cmpgt_epi64(b, a);}
BMAS_ivec static inline BMAS_vector_i64le (BMAS_ivec a, BMAS_ivec b){return BMAS_vector_i8not(_mm256_cmpgt_epi64(a, b));}
BMAS_ivec static inline BMAS_vector_i64eq (BMAS_ivec a, BMAS_ivec b){return _mm256_cmpeq_epi64(a, b);}
BMAS_ivec static inline BMAS_vector_i64neq(BMAS_ivec a, BMAS_ivec b){return BMAS_vector_i8not(_mm256_cmpeq_epi64(a, b));}
BMAS_ivec static inline BMAS_vector_i64gt (BMAS_ivec a, BMAS_ivec b){return _mm256_cmpgt_epi64(a, b);}
BMAS_ivec static inline BMAS_vector_i64ge (BMAS_ivec a, BMAS_ivec b){return BMAS_vector_i8not(_mm256_cmpgt_epi64(b, a));}



BMAS_ivec static inline BMAS_vector_u8gt (BMAS_ivec a, BMAS_ivec b){
  BMAS_ivec fill = _mm256_set_epi32(0x80808080, 0x80808080, 0x80808080, 0x80808080,
                                    0x80808080, 0x80808080, 0x80808080, 0x80808080);
  a = _mm256_sub_epi8(a, fill);
  b = _mm256_sub_epi8(b, fill);
  return _mm256_cmpgt_epi8(a, b);
}
BMAS_ivec static inline BMAS_vector_u8lt (BMAS_ivec a, BMAS_ivec b){return BMAS_vector_u8gt(b, a);}
BMAS_ivec static inline BMAS_vector_u8le (BMAS_ivec a, BMAS_ivec b){return BMAS_vector_i8not(BMAS_vector_u8gt(a, b));}
BMAS_ivec static inline BMAS_vector_u8eq (BMAS_ivec a, BMAS_ivec b){return _mm256_cmpeq_epi8(a, b);}
BMAS_ivec static inline BMAS_vector_u8neq(BMAS_ivec a, BMAS_ivec b){return BMAS_vector_i8not(_mm256_cmpeq_epi8(a, b));}
BMAS_ivec static inline BMAS_vector_u8ge (BMAS_ivec a, BMAS_ivec b){return BMAS_vector_i8not(BMAS_vector_u8gt(b, a));}

BMAS_ivec static inline BMAS_vector_u16gt (BMAS_ivec a, BMAS_ivec b){
  BMAS_ivec fill = _mm256_set_epi32(0x80008000, 0x80008000, 0x80008000, 0x80008000,
                                    0x80008000, 0x80008000, 0x80008000, 0x80008000);
  a = _mm256_sub_epi16(a, fill);
  b = _mm256_sub_epi16(b, fill);
  return _mm256_cmpgt_epi16(a, b);
}
BMAS_ivec static inline BMAS_vector_u16lt (BMAS_ivec a, BMAS_ivec b){return BMAS_vector_u16gt(b, a);}
BMAS_ivec static inline BMAS_vector_u16le (BMAS_ivec a, BMAS_ivec b){return BMAS_vector_i8not(BMAS_vector_u16gt(a, b));}
BMAS_ivec static inline BMAS_vector_u16eq (BMAS_ivec a, BMAS_ivec b){return _mm256_cmpeq_epi16(a, b);}
BMAS_ivec static inline BMAS_vector_u16neq(BMAS_ivec a, BMAS_ivec b){return BMAS_vector_i8not(_mm256_cmpeq_epi16(a, b));}
BMAS_ivec static inline BMAS_vector_u16ge (BMAS_ivec a, BMAS_ivec b){return BMAS_vector_i8not(BMAS_vector_u16gt(b, a));}

BMAS_ivec static inline BMAS_vector_u32gt (BMAS_ivec a, BMAS_ivec b){
  BMAS_ivec fill = _mm256_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000,
                                    0x80000000, 0x80000000, 0x80000000, 0x80000000);
  a = _mm256_sub_epi32(a, fill);
  b = _mm256_sub_epi32(b, fill);
  return _mm256_cmpgt_epi32(a, b);
}
BMAS_ivec static inline BMAS_vector_u32lt (BMAS_ivec a, BMAS_ivec b){return BMAS_vector_u32gt(b, a);}
BMAS_ivec static inline BMAS_vector_u32le (BMAS_ivec a, BMAS_ivec b){return BMAS_vector_i8not(BMAS_vector_u32gt(a, b));}
BMAS_ivec static inline BMAS_vector_u32eq (BMAS_ivec a, BMAS_ivec b){return _mm256_cmpeq_epi32(a, b);}
BMAS_ivec static inline BMAS_vector_u32neq(BMAS_ivec a, BMAS_ivec b){return BMAS_vector_i8not(_mm256_cmpeq_epi32(a, b));}
BMAS_ivec static inline BMAS_vector_u32ge (BMAS_ivec a, BMAS_ivec b){return BMAS_vector_i8not(BMAS_vector_u32gt(b, a));}

BMAS_ivec static inline BMAS_vector_u64gt (BMAS_ivec a, BMAS_ivec b){
  BMAS_ivec fill = _mm256_set_epi64x(0x8000000000000000, 0x8000000000000000,
                                     0x8000000000000000, 0x8000000000000000);
  a = _mm256_sub_epi64(a, fill);
  b = _mm256_sub_epi64(b, fill);
  return _mm256_cmpgt_epi64(a, b);
}
BMAS_ivec static inline BMAS_vector_u64lt (BMAS_ivec a, BMAS_ivec b){return _mm256_cmpgt_epi64(b, a);}
BMAS_ivec static inline BMAS_vector_u64le (BMAS_ivec a, BMAS_ivec b){return BMAS_vector_i8not(_mm256_cmpgt_epi64(a, b));}
BMAS_ivec static inline BMAS_vector_u64eq (BMAS_ivec a, BMAS_ivec b){return _mm256_cmpeq_epi64(a, b);}
BMAS_ivec static inline BMAS_vector_u64neq(BMAS_ivec a, BMAS_ivec b){return BMAS_vector_i8not(_mm256_cmpeq_epi64(a, b));}
BMAS_ivec static inline BMAS_vector_u64ge (BMAS_ivec a, BMAS_ivec b){return BMAS_vector_i8not(_mm256_cmpgt_epi64(b, a));}

// basic arithmetic and other vertical functions

BMAS_svec static inline BMAS_vector_sadd(BMAS_svec a, BMAS_svec b){return _mm256_add_ps(a, b);}
BMAS_svec static inline BMAS_vector_ssub(BMAS_svec a, BMAS_svec b){return _mm256_sub_ps(a, b);}
BMAS_svec static inline BMAS_vector_smul(BMAS_svec a, BMAS_svec b){return _mm256_mul_ps(a, b);}
BMAS_svec static inline BMAS_vector_sdiv(BMAS_svec a, BMAS_svec b){return _mm256_div_ps(a, b);}

BMAS_dvec static inline BMAS_vector_dadd(BMAS_dvec a, BMAS_dvec b){return _mm256_add_pd(a, b);}
BMAS_dvec static inline BMAS_vector_dsub(BMAS_dvec a, BMAS_dvec b){return _mm256_sub_pd(a, b);}
BMAS_dvec static inline BMAS_vector_dmul(BMAS_dvec a, BMAS_dvec b){return _mm256_mul_pd(a, b);}
BMAS_dvec static inline BMAS_vector_ddiv(BMAS_dvec a, BMAS_dvec b){return _mm256_div_pd(a, b);}

BMAS_svec static inline BMAS_vector_smax(BMAS_svec a, BMAS_svec b){return _mm256_max_ps(a, b);}
BMAS_dvec static inline BMAS_vector_dmax(BMAS_dvec a, BMAS_dvec b){return _mm256_max_pd(a, b);}
BMAS_ivec static inline BMAS_vector_i64max(BMAS_ivec a, BMAS_ivec b){
  BMAS_ivec asep = _mm256_permutevar8x32_epi32(a, _mm256_set_epi32(6,4,2,0,7,5,3,1));
  BMAS_ivec bsep = _mm256_permutevar8x32_epi32(b, _mm256_set_epi32(6,4,2,0,7,5,3,1));

  // The upper half needs signed comparison, lower half needs unsigned comparison
  BMAS_ivech alow = _mm256_extracti128_si256(asep, 1);
  BMAS_ivech blow = _mm256_extracti128_si256(bsep, 1);
  BMAS_ivech ahigh = _mm256_castsi256_si128(asep);
  BMAS_ivech bhigh = _mm256_castsi256_si128(bsep);

  BMAS_ivech gthigh = _mm_cmpgt_epi32(ahigh, bhigh);

  BMAS_ivech fill = _mm_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000);
  BMAS_ivech alow_epu = _mm_sub_epi32(alow, fill);
  BMAS_ivech blow_epu = _mm_sub_epi32(blow, fill);
  BMAS_ivech gtlow  = _mm_cmpgt_epi32(alow_epu, blow_epu);
  // gthigh == 0 && gtlow == 1 => mask == 1
  // gthigh == 0 && gtlow == 0 => mask == 0
  // gthigh == 1 && gtlow == 0 => mask == 1
  // gthigh == 1 && gtlow == 1 => mask == 1
  BMAS_ivech mask = _mm_or_si128(gthigh, gtlow);
  BMAS_ivec fullmask = _mm256_cvtepi32_epi64(mask);
  BMAS_ivec resulta = _mm256_and_si256(fullmask, a);
  BMAS_ivec resultb = _mm256_andnot_si256(fullmask, b);
  BMAS_ivec result = _mm256_or_si256(resulta, resultb);

  return result;
}
BMAS_ivec static inline BMAS_vector_i32max(BMAS_ivec a, BMAS_ivec b){return _mm256_max_epi32(a, b);}
BMAS_ivec static inline BMAS_vector_i16max(BMAS_ivec a, BMAS_ivec b){return _mm256_max_epi16(a, b);}
BMAS_ivec static inline BMAS_vector_i8max (BMAS_ivec a, BMAS_ivec b){return _mm256_max_epi8(a, b);}
BMAS_ivec static inline BMAS_vector_u64max(BMAS_ivec a, BMAS_ivec b){
  BMAS_ivec asep = _mm256_permutevar8x32_epi32(a, _mm256_set_epi32(6,4,2,0,7,5,3,1));
  BMAS_ivec bsep = _mm256_permutevar8x32_epi32(b, _mm256_set_epi32(6,4,2,0,7,5,3,1));

  BMAS_ivec gt = BMAS_vector_u32gt(asep, bsep);

  // The upper half needs unsigned comparison, lower half needs unsigned comparison
  /* BMAS_ivech alow = _mm256_extracti128_si256(asep, 1); */
  /* BMAS_ivech blow = _mm256_extracti128_si256(bsep, 1); */
  /* BMAS_ivech ahigh = _mm256_castsi256_si128(asep); */
  /* BMAS_ivech bhigh = _mm256_castsi256_si128(bsep); */

  BMAS_ivech gthigh = _mm256_extracti128_si256(gt, 1);
  BMAS_ivech gtlow  = _mm256_castsi256_si128(gt);

  /* BMAS_ivech gthigh = _mm_cmpgt_epu32(ahigh, bhigh); */
  /* BMAS_ivech gtlow  = _mm_cmpgt_epu32(alow, blow); */
  // gthigh == 0 && gtlow == 1 => mask == 1
  // gthigh == 0 && gtlow == 0 => mask == 0
  // gthigh == 1 && gtlow == 0 => mask == 1
  // gthigh == 1 && gtlow == 1 => mask == 1
  BMAS_ivech mask = _mm_or_si128(gthigh, gtlow);
  BMAS_ivec fullmask = _mm256_cvtepi32_epi64(mask);
  BMAS_ivec resulta = _mm256_and_si256(fullmask, a);
  BMAS_ivec resultb = _mm256_andnot_si256(fullmask, b);
  BMAS_ivec result = _mm256_or_si256(resulta, resultb);

  return result;
}
BMAS_ivec static inline BMAS_vector_u32max(BMAS_ivec a, BMAS_ivec b){return _mm256_max_epu32(a, b);}
BMAS_ivec static inline BMAS_vector_u16max(BMAS_ivec a, BMAS_ivec b){return _mm256_max_epu16(a, b);}
BMAS_ivec static inline BMAS_vector_u8max (BMAS_ivec a, BMAS_ivec b){return _mm256_max_epu8(a, b);}

BMAS_svec static inline BMAS_vector_smin(BMAS_svec a, BMAS_svec b){return _mm256_min_ps(a, b);}
BMAS_dvec static inline BMAS_vector_dmin(BMAS_dvec a, BMAS_dvec b){return _mm256_min_pd(a, b);}
BMAS_ivec static inline BMAS_vector_i64min(BMAS_ivec a, BMAS_ivec b){
  // high bits in the lower 128 bits; low bits in the upper 128 bits
  BMAS_ivec asep = _mm256_permutevar8x32_epi32(a, _mm256_set_epi32(6,4,2,0,7,5,3,1));
  BMAS_ivec bsep = _mm256_permutevar8x32_epi32(b, _mm256_set_epi32(6,4,2,0,7,5,3,1));

  // The upper half needs signed comparison, lower half needs unsigned comparison
  BMAS_ivech alow = _mm256_extracti128_si256(asep, 1);
  BMAS_ivech blow = _mm256_extracti128_si256(bsep, 1);
  BMAS_ivech ahigh = _mm256_castsi256_si128(asep);
  BMAS_ivech bhigh = _mm256_castsi256_si128(bsep);

  BMAS_ivech gthigh = _mm_cmpgt_epi32(ahigh, bhigh);

  /* BMAS_ivech fill = _mm_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000); */
  /* BMAS_ivech alow_epu = _mm_sub_epi32(alow, fill); */
  /* BMAS_ivech blow_epu = _mm_sub_epi32(blow, fill); */
  /* BMAS_ivech gtlow  = _mm_cmpgt_epi32(alow_epu, blow_epu); */
  BMAS_ivech gtlow = _mm_cmpgt_epi32(alow, blow);
  // gthigh == 0 && gtlow == 1 => mask == 1
  // gthigh == 0 && gtlow == 0 => mask == 0
  // gthigh == 1 && gtlow == 0 => mask == 1
  // gthigh == 1 && gtlow == 1 => mask == 1
  BMAS_ivech mask = _mm_or_si128(gthigh, gtlow);
  BMAS_ivec fullmask = _mm256_cvtepi32_epi64(mask);
  BMAS_ivec resulta = _mm256_andnot_si256(fullmask, a);
  BMAS_ivec resultb = _mm256_and_si256(fullmask, b);
  BMAS_ivec result = _mm256_or_si256(resulta, resultb);

  return result;

  /* BMAS_ivec asep = _mm256_permutevar8x32_epi32(a, _mm256_set_epi32(8,4,2,0,7,5,3,1)); */
  /* BMAS_ivec bsep = _mm256_permutevar8x32_epi32(b, _mm256_set_epi32(8,4,2,0,7,5,3,1)); */

  /* // The upper half needs signed comparison, lower half needs unsigned comparison */
  /* BMAS_ivech alow = _mm256_extracti128_si256(asep, 1); */
  /* BMAS_ivech blow = _mm256_extracti128_si256(bsep, 1); */
  /* BMAS_ivech ahigh = _mm256_castsi256_si128(asep); */
  /* BMAS_ivech bhigh = _mm256_castsi256_si128(bsep); */

  /* BMAS_ivech lthigh = _mm_cmpgt_epi32(bhigh, ahigh); */

  /* BMAS_ivech fill = _mm_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000); */
  /* BMAS_ivech alow_epu = _mm_sub_epi32(alow, fill); */
  /* BMAS_ivech blow_epu = _mm_sub_epi32(blow, fill); */
  /* BMAS_ivech ltlow  = _mm_cmpgt_epi32(blow_epu, alow_epu); */
  /* // lthigh == 0 && ltlow == 1 => mask == 1 */
  /* // lthigh == 0 && ltlow == 0 => mask == 0 */
  /* // lthigh == 1 && ltlow == 0 => mask == 1 */
  /* // lthigh == 1 && ltlow == 1 => mask == 1 */
  /* BMAS_ivech mask = _mm_or_si128(lthigh, ltlow); */
  /* BMAS_ivec fullmask = _mm256_cvtepi32_epi64(mask); */
  /* BMAS_ivec resulta = _mm256_and_si256(a, fullmask); */
  /* BMAS_ivec resultb = _mm256_andnot_si256(b, fullmask); */
  /* BMAS_ivec result = _mm256_or_si256(resulta, resultb); */

  /* return result; */
}
BMAS_ivec static inline BMAS_vector_i32min(BMAS_ivec a, BMAS_ivec b){return _mm256_min_epi32(a, b);}
BMAS_ivec static inline BMAS_vector_i16min(BMAS_ivec a, BMAS_ivec b){return _mm256_min_epi16(a, b);}
BMAS_ivec static inline BMAS_vector_i8min (BMAS_ivec a, BMAS_ivec b){return _mm256_min_epi8(a, b);}
BMAS_ivec static inline BMAS_vector_u64min(BMAS_ivec a, BMAS_ivec b){
  BMAS_ivec asep = _mm256_permutevar8x32_epi32(a, _mm256_set_epi32(6,4,2,0,7,5,3,1));
  BMAS_ivec bsep = _mm256_permutevar8x32_epi32(b, _mm256_set_epi32(6,4,2,0,7,5,3,1));

  // The upper half needs unsigned comparison, lower half needs unsigned comparison
  BMAS_ivec lt = BMAS_vector_u32gt(bsep, asep);

  BMAS_ivech lthigh = _mm256_extracti128_si256(lt, 1);
  BMAS_ivech ltlow  = _mm256_castsi256_si128(lt);

  // lthigh == 0 && ltlow == 1 => mask == 1
  // lthigh == 0 && ltlow == 0 => mask == 0
  // lthigh == 1 && ltlow == 0 => mask == 1
  // lthigh == 1 && ltlow == 1 => mask == 1
  BMAS_ivech mask = _mm_or_si128(lthigh, ltlow);
  BMAS_ivec fullmask = _mm256_cvtepi32_epi64(mask);
  BMAS_ivec resulta = _mm256_and_si256(fullmask, a);
  BMAS_ivec resultb = _mm256_andnot_si256(fullmask, b);
  BMAS_ivec result = _mm256_or_si256(resulta, resultb);

  return result;
}
BMAS_ivec static inline BMAS_vector_u32min(BMAS_ivec a, BMAS_ivec b){return _mm256_min_epu32(a, b);}
BMAS_ivec static inline BMAS_vector_u16min(BMAS_ivec a, BMAS_ivec b){return _mm256_min_epu16(a, b);}
BMAS_ivec static inline BMAS_vector_u8min (BMAS_ivec a, BMAS_ivec b){return _mm256_min_epu8(a, b);}


// horizontal functions


// First do, 4 2-elt, then 2 4-elt, finally 1 8-elt
// 1st and 2nd elements of v4 should be from the 3rd and 4th of v3 (1-indexed)
// 1st element of v6 should be from the second element of v5
#define HORIZONTAL_FUNCTION_BODY_F32(name)          \
  BMAS_svech v1 = _mm256_castps256_ps128(a);        \
  BMAS_svech v2 = _mm256_extractf128_ps(a, 1);      \
  BMAS_svech v3 = _mm_##name##_ps(v1, v2);          \
  BMAS_svech v4 = _mm_permute_ps(v3, 0b11101110);   \
  BMAS_svech v5 = _mm_##name##_ps(v3, v4);          \
  BMAS_svech v6 = _mm_permute_ps(v5, 0b01010101);   \
  BMAS_svech v = _mm_##name##_ps(v5, v6);           \
  return _mm_cvtss_f32(v);

// First do, 2 2-elt, finally 1 4-elt
// move the 2nd element to 1st position
#define HORIZONTAL_FUNCTION_BODY_F64(name)      \
  BMAS_dvech v1 = _mm256_castpd256_pd128(a);    \
  BMAS_dvech v2 = _mm256_extractf128_pd(a, 1);  \
  BMAS_dvech v3 = _mm_##name##_pd(v1, v2);      \
  BMAS_dvech v4 = _mm_permute_pd(v3, 0b01);     \
  BMAS_dvech v = _mm_##name##_pd(v3, v4);       \
  return _mm_cvtsd_f64(v);

#define HORIZONTAL_FUNCTION_BODY_IU64(name, type)   \
  BMAS_ivech v1 = _mm256_castsi256_si128(a);        \
  BMAS_ivech v2 = _mm256_extracti128_si256(a, 1);   \
  BMAS_ivech v3 = _mm_##name##_##type(v1, v2);      \
  BMAS_ivech v4 = _mm_bsrli_si128(v3, 8);           \
  BMAS_ivech v = _mm_##name##_##type(v3, v4);       \
  return _mm_extract_epi64(v, 0);

#define HORIZONTAL_FUNCTION_BODY_IU32(name, type)   \
  BMAS_ivech v1 = _mm256_castsi256_si128(a);        \
  BMAS_ivech v2 = _mm256_extracti128_si256(a, 1);   \
  BMAS_ivech v3 = _mm_##name##_##type(v1, v2);      \
  BMAS_ivech v4 = _mm_bsrli_si128(v3, 8);           \
  BMAS_ivech v5 = _mm_##name##_##type(v3, v4);      \
  BMAS_ivech v6 = _mm_bsrli_si128(v5, 4);           \
  BMAS_ivech v = _mm_##name##_##type(v5, v6);       \
  return _mm_extract_epi32(v, 0);

#define HORIZONTAL_FUNCTION_BODY_INDIRECT(from, to, final, scalar_fn)   \
  BMAS_ivech v1 = _mm256_castsi256_si128(a);                            \
  BMAS_ivech v2 = _mm256_extracti128_si256(a, 1);                       \
  BMAS_ivec v1e = _mm256_cvt##from##_##to(v1);                          \
  BMAS_ivec v2e = _mm256_cvt##from##_##to(v2);                          \
  return BMAS_scalar_##scalar_fn(BMAS_vector_##final(v1e), BMAS_vector_##final(v2e));

float static inline BMAS_vector_shadd(BMAS_svec a){HORIZONTAL_FUNCTION_BODY_F32(add);}
double static inline BMAS_vector_dhadd(BMAS_dvec a){HORIZONTAL_FUNCTION_BODY_F64(add);}
int64_t static inline BMAS_vector_i64hadd(BMAS_ivec a){HORIZONTAL_FUNCTION_BODY_IU64(add, epi64);}
int32_t static inline BMAS_vector_i32hadd(BMAS_ivec a){HORIZONTAL_FUNCTION_BODY_IU32(add, epi32);}
int16_t static inline BMAS_vector_i16hadd(BMAS_ivec a){HORIZONTAL_FUNCTION_BODY_INDIRECT(epi16, epi32, i32hadd, i16add);}
int8_t static inline BMAS_vector_i8hadd(BMAS_ivec a){HORIZONTAL_FUNCTION_BODY_INDIRECT(epi8, epi16, i16hadd, i8add);}

float static inline BMAS_vector_shmax(BMAS_svec a){HORIZONTAL_FUNCTION_BODY_F32(max);}
double static inline BMAS_vector_dhmax(BMAS_dvec a){HORIZONTAL_FUNCTION_BODY_F64(max);}
int64_t static inline BMAS_vector_i64hmax(BMAS_ivec a){
  BMAS_ivec v1 = a;
  BMAS_ivec v2 = _mm256_castsi128_si256(_mm256_extracti128_si256(a, 1));
  BMAS_ivec v3 = BMAS_vector_i64max(v1, v2);
  BMAS_ivec v4 = _mm256_srli_si256(v3, 8);
  BMAS_ivec v = BMAS_vector_i64max(v3, v4);
  return _mm256_extract_epi64(v, 0);
}
int32_t static inline BMAS_vector_i32hmax(BMAS_ivec a){HORIZONTAL_FUNCTION_BODY_IU32(max, epi32);}
int16_t static inline BMAS_vector_i16hmax(BMAS_ivec a){HORIZONTAL_FUNCTION_BODY_INDIRECT(epi16, epi32, i32hmax, i16max);}
int8_t static inline BMAS_vector_i8hmax(BMAS_ivec a){HORIZONTAL_FUNCTION_BODY_INDIRECT(epi8, epi16, i16hmax, i8max);}
uint64_t static inline BMAS_vector_u64hmax(BMAS_ivec a){
  BMAS_ivec v1 = a;
  BMAS_ivec v2 = _mm256_castsi128_si256(_mm256_extracti128_si256(a, 1));
  BMAS_ivec v3 = BMAS_vector_u64max(v1, v2);
  BMAS_ivec v4 = _mm256_srli_si256(v3, 8);
  BMAS_ivec v = BMAS_vector_u64max(v3, v4);
  return _mm256_extract_epi64(v, 0);
}
uint32_t static inline BMAS_vector_u32hmax(BMAS_ivec a){HORIZONTAL_FUNCTION_BODY_IU32(max, epu32);}
uint16_t static inline BMAS_vector_u16hmax(BMAS_ivec a){HORIZONTAL_FUNCTION_BODY_INDIRECT(epu16, epi32, u32hmax, u16max);}
uint8_t static inline BMAS_vector_u8hmax(BMAS_ivec a){HORIZONTAL_FUNCTION_BODY_INDIRECT(epu8, epi16, u16hmax, u8max);}

float static inline BMAS_vector_shmin(BMAS_svec a){HORIZONTAL_FUNCTION_BODY_F32(min);}
double static inline BMAS_vector_dhmin(BMAS_dvec a){HORIZONTAL_FUNCTION_BODY_F64(min);}
int64_t static inline BMAS_vector_i64hmin(BMAS_ivec a){
  BMAS_ivec v1 = a;
  BMAS_ivec v2 = _mm256_castsi128_si256(_mm256_extracti128_si256(a, 1));
  BMAS_ivec v3 = BMAS_vector_i64min(v1, v2);
  BMAS_ivec v4 = _mm256_srli_si256(v3, 8);
  BMAS_ivec v = BMAS_vector_i64min(v3, v4);
  return _mm256_extract_epi64(v, 0);
}
int32_t static inline BMAS_vector_i32hmin(BMAS_ivec a){HORIZONTAL_FUNCTION_BODY_IU32(min, epi32);}
int16_t static inline BMAS_vector_i16hmin(BMAS_ivec a){HORIZONTAL_FUNCTION_BODY_INDIRECT(epi16, epi32, i32hmin, i16min);}
int8_t static inline BMAS_vector_i8hmin(BMAS_ivec a){HORIZONTAL_FUNCTION_BODY_INDIRECT(epi8, epi16, i16hmin, i8min);}
uint64_t static inline BMAS_vector_u64hmin(BMAS_ivec a){
  BMAS_ivec v1 = a;
  BMAS_ivec v2 = _mm256_castsi128_si256(_mm256_extracti128_si256(a, 1));
  BMAS_ivec v3 = BMAS_vector_u64min(v1, v2);
  BMAS_ivec v4 = _mm256_srli_si256(v3, 8);
  BMAS_ivec v = BMAS_vector_u64min(v3, v4);
  return _mm256_extract_epi64(v, 0);
}
uint32_t static inline BMAS_vector_u32hmin(BMAS_ivec a){HORIZONTAL_FUNCTION_BODY_IU32(min, epu32);}
uint16_t static inline BMAS_vector_u16hmin(BMAS_ivec a){HORIZONTAL_FUNCTION_BODY_INDIRECT(epu16, epi32, u32hmin, u16min);}
uint8_t static inline BMAS_vector_u8hmin(BMAS_ivec a){HORIZONTAL_FUNCTION_BODY_INDIRECT(epu8, epi16, u16hmin, u8min);}


#undef HORIZONTAL_FUNCTION_BODY_F32
#undef HORIZONTAL_FUNCTION_BODY_F64
#undef HORIZONTAL_FUNCTION_BODY_IU64
#undef HORIZONTAL_FUNCTION_BODY_IU32
#undef HORIZONTAL_FUNCTION_BODY_INDIRECT


// integer insert and extract - we need to do this because compiler needs "immediates"

BMAS_ivec static inline BMAS_ivec_make_i64(int64_t* ptr, const int stride){
  BMAS_ivec v;
  v = _mm256_insert_epi64(v, ptr[0*stride], 0);
  v = _mm256_insert_epi64(v, ptr[1*stride], 1);
  v = _mm256_insert_epi64(v, ptr[2*stride], 2);
  v = _mm256_insert_epi64(v, ptr[3*stride], 3);
  return v;
}

#define V256_INSERT32_FROM_PTR(I) v = _mm256_insert_epi32(v, ptr[I*stride], I);
#define V256_INSERT16_FROM_PTR(I) v = _mm256_insert_epi16(v, ptr[I*stride], I);
#define V256_INSERT8_FROM_PTR(I) v = _mm256_insert_epi8(v, ptr[I*stride], I);

BMAS_ivec static inline BMAS_ivec_make_i32(int32_t* ptr, const int stride){
  BMAS_ivec v;
  MACRO_LOOP(8, V256_INSERT32_FROM_PTR);
  return v;
}

BMAS_ivec static inline BMAS_ivec_make_i16(int16_t* ptr, const int stride){
  BMAS_ivec v;
  MACRO_LOOP(16, V256_INSERT16_FROM_PTR);
  return v;
}

BMAS_ivec static inline BMAS_ivec_make_i8(int8_t* ptr, const int stride){
  BMAS_ivec v;
  MACRO_LOOP(32, V256_INSERT8_FROM_PTR);
  return v;
}

#undef V256_INSERT32_FROM_PTR
#undef V256_INSERT16_FROM_PTR
#undef V256_INSERT8_FROM_PTR

BMAS_ivech static inline BMAS_ivech_make_i64(int64_t* ptr, const int stride){
  BMAS_ivech v;
  v = _mm_insert_epi64(v, ptr[0*stride], 0);
  v = _mm_insert_epi64(v, ptr[1*stride], 1);
  return v;
}

BMAS_ivech static inline BMAS_ivech_make_i32(int32_t* ptr, const int stride){
  BMAS_ivech v;
  v = _mm_insert_epi32(v, ptr[0*stride], 0);
  v = _mm_insert_epi32(v, ptr[1*stride], 1);
  v = _mm_insert_epi32(v, ptr[2*stride], 2);
  v = _mm_insert_epi32(v, ptr[3*stride], 3);
  return v;
}

#define V128_INSERT16_FROM_PTR(I) v = _mm_insert_epi16(v, ptr[I*stride], I);
#define V128_INSERT8_FROM_PTR(I) v = _mm_insert_epi8(v, ptr[I*stride], I);

BMAS_ivech static inline BMAS_ivech_make_i16(int16_t* ptr, const int stride){
  BMAS_ivech v;
  MACRO_LOOP(8, V128_INSERT16_FROM_PTR);
  return v;
}

BMAS_ivech static inline BMAS_ivech_make_i8(int8_t* ptr, const int stride){
  BMAS_ivech v;
  MACRO_LOOP(16, V128_INSERT8_FROM_PTR);
  return v;
}

#undef V128_INSERT16_FROM_PTR
#undef V128_INSERT8_FROM_PTR

void static inline BMAS_ivec_store_multi_i64(BMAS_ivec v, int64_t* ptr, const int stride){
  ptr[0*stride] = _mm256_extract_epi64(v, 0);
  ptr[1*stride] = _mm256_extract_epi64(v, 1);
  ptr[2*stride] = _mm256_extract_epi64(v, 2);
  ptr[3*stride] = _mm256_extract_epi64(v, 3);
}

#define V256_EXTRACT32_FROM_PTR(I) ptr[I*stride] = _mm256_extract_epi32(v, I);
#define V256_EXTRACT16_FROM_PTR(I) ptr[I*stride] = _mm256_extract_epi16(v, I);
#define V256_EXTRACT8_FROM_PTR(I)  ptr[I*stride] = _mm256_extract_epi8(v, I);

void static inline BMAS_ivec_store_multi_i32(BMAS_ivec v, int32_t* ptr, const int stride){
  MACRO_LOOP(8, V256_EXTRACT32_FROM_PTR);
}
void static inline BMAS_ivec_store_multi_i16(BMAS_ivec v, int16_t* ptr, const int stride){
  MACRO_LOOP(16, V256_EXTRACT16_FROM_PTR);
}
void static inline BMAS_ivec_store_multi_i8(BMAS_ivec v, int8_t* ptr, const int stride){
  MACRO_LOOP(32, V256_EXTRACT8_FROM_PTR);
}

#undef V256_EXTRACT32_FROM_PTR
#undef V256_EXTRACT16_FROM_PTR
#undef V256_EXTRACT8_FROM_PTR


// integer arithmetic
BMAS_ivec static inline BMAS_vector_i64add(BMAS_ivec a, BMAS_ivec b){return _mm256_add_epi64(a, b);}
BMAS_ivec static inline BMAS_vector_i32add(BMAS_ivec a, BMAS_ivec b){return _mm256_add_epi32(a, b);}
BMAS_ivec static inline BMAS_vector_i16add(BMAS_ivec a, BMAS_ivec b){return _mm256_add_epi16(a, b);}
BMAS_ivec static inline BMAS_vector_i8add (BMAS_ivec a, BMAS_ivec b){return _mm256_add_epi8(a, b);}

BMAS_ivec static inline BMAS_vector_i64sub(BMAS_ivec a, BMAS_ivec b){return _mm256_sub_epi64(a, b);}
BMAS_ivec static inline BMAS_vector_i32sub(BMAS_ivec a, BMAS_ivec b){return _mm256_sub_epi32(a, b);}
BMAS_ivec static inline BMAS_vector_i16sub(BMAS_ivec a, BMAS_ivec b){return _mm256_sub_epi16(a, b);}
BMAS_ivec static inline BMAS_vector_i8sub (BMAS_ivec a, BMAS_ivec b){return _mm256_sub_epi8(a, b);}

BMAS_ivec static inline BMAS_vector_i64mul(BMAS_ivec a, BMAS_ivec b){
  // Credits: user Z boson from https://stackoverflow.com/a/37322570/8957330
  __m256i bswap   = _mm256_shuffle_epi32(b,0xB1);           // swap H<->L
  __m256i prodlh  = _mm256_mullo_epi32(a,bswap);            // 32 bit L*H products
  __m256i zero    = _mm256_setzero_si256();                 // 0
  __m256i prodlh2 = _mm256_hadd_epi32(prodlh,zero);         // a0Lb0H+a0Hb0L,a1Lb1H+a1Hb1L,0,0
  __m256i prodlh3 = _mm256_shuffle_epi32(prodlh2,0x73);     // 0, a0Lb0H+a0Hb0L, 0, a1Lb1H+a1Hb1L
  __m256i prodll  = _mm256_mul_epu32(a,b);                  // a0Lb0L,a1Lb1L, 64 bit unsigned products
  __m256i prod    = _mm256_add_epi64(prodll,prodlh3);       // a0Lb0L+(a0Lb0H+a0Hb0L)<<32, a1Lb1L+(a1Lb1H+a1Hb1L)<<32
  return  prod;
}
BMAS_ivec static inline BMAS_vector_i32mul(BMAS_ivec a, BMAS_ivec b){return _mm256_mullo_epi32(a, b);}
BMAS_ivec static inline BMAS_vector_i16mul(BMAS_ivec a, BMAS_ivec b){return _mm256_mullo_epi16(a, b);}
BMAS_ivec static inline BMAS_vector_i8mul (BMAS_ivec a, BMAS_ivec b){

  BMAS_ivech ah1 = _mm256_extractf128_si256(a, 0);
  BMAS_ivech bh1 = _mm256_extractf128_si256(b, 0);
  BMAS_ivec ah1_16bit = _mm256_cvtepi8_epi16(ah1);
  BMAS_ivec bh1_16bit = _mm256_cvtepi8_epi16(bh1);
  BMAS_ivec v1 = _mm256_mullo_epi16(ah1_16bit, bh1_16bit);

  BMAS_ivech ah2 = _mm256_extractf128_si256(a, 1);
  BMAS_ivech bh2 = _mm256_extractf128_si256(b, 1);
  BMAS_ivec ah2_16bit = _mm256_cvtepi8_epi16(ah2);
  BMAS_ivec bh2_16bit = _mm256_cvtepi8_epi16(bh2);
  BMAS_ivec v2 = _mm256_mullo_epi16(ah2_16bit, bh2_16bit);

  // Collect low 8 bits into the lower 64 bits of each 128 bit half-register
  v1 = _mm256_shuffle_epi8(v1, _mm256_setr_epi32(0x06040200, 0x0E0C0A08, 0x0, 0x0,
                                                 0x06040200, 0x0E0C0A08, 0x0, 0x0));
  v2 = _mm256_shuffle_epi8(v2, _mm256_setr_epi32(0x0, 0x0, 0x06040200, 0x0E0C0A08,
                                                 0x0, 0x0, 0x06040200, 0x0E0C0A08));

  // Collect low 8 bits into the lower 64 bits of each 128 bit half-register
  v1 = _mm256_and_si256(v1, _mm256_setr_epi64x(0xFFFFFFFFFFFFFFFF, 0x0000000000000000,
                                               0xFFFFFFFFFFFFFFFF, 0x0000000000000000));
  v2 = _mm256_and_si256(v2, _mm256_setr_epi64x(0x0000000000000000, 0xFFFFFFFFFFFFFFFF,
                                               0x0000000000000000, 0xFFFFFFFFFFFFFFFF));
  BMAS_ivec v = _mm256_or_si256(v1, v2);
  // Reorder to get in the correct order
  v = _mm256_permute4x64_epi64(v, 0b11011000);
  return v;
}

BMAS_ivec static inline BMAS_vector_u64mul(BMAS_ivec a, BMAS_ivec b){
  // Identical to i64mul except for the use of mul_epu32 instead of mul_epi32
  __m256i bswap   = _mm256_shuffle_epi32(b,0xB1);           // swap H<->L
  __m256i prodlh  = _mm256_mullo_epi32(a,bswap);            // 32 bit L*H products
  __m256i zero    = _mm256_setzero_si256();                 // 0
  __m256i prodlh2 = _mm256_hadd_epi32(prodlh,zero);         // a0Lb0H+a0Hb0L,a1Lb1H+a1Hb1L,0,0
  __m256i prodlh3 = _mm256_shuffle_epi32(prodlh2,0x73);     // 0, a0Lb0H+a0Hb0L, 0, a1Lb1H+a1Hb1L
  __m256i prodll  = _mm256_mul_epu32(a,b);                  // a0Lb0L,a1Lb1L, 64 bit unsigned products
  __m256i prod    = _mm256_add_epi64(prodll,prodlh3);       // a0Lb0L+(a0Lb0H+a0Hb0L)<<32, a1Lb1L+(a1Lb1H+a1Hb1L)<<32
  return  prod;
}
BMAS_ivec static inline BMAS_vector_u32mul(BMAS_ivec a, BMAS_ivec b){return _mm256_mullo_epi32(a, b);}
BMAS_ivec static inline BMAS_vector_u16mul(BMAS_ivec a, BMAS_ivec b){return _mm256_mullo_epi16(a, b);}
BMAS_ivec static inline BMAS_vector_u8mul(BMAS_ivec a, BMAS_ivec b){
  // Identical to i8mul except for the use of cvtepu8_epi16 instead of cvtepi8_epi16
  BMAS_ivech ah1 = _mm256_extractf128_si256(a, 0);
  BMAS_ivech bh1 = _mm256_extractf128_si256(b, 0);
  BMAS_ivec ah1_16bit = _mm256_cvtepu8_epi16(ah1);
  BMAS_ivec bh1_16bit = _mm256_cvtepu8_epi16(bh1);
  BMAS_ivec v1 = _mm256_mullo_epi16(ah1_16bit, bh1_16bit);

  BMAS_ivech ah2 = _mm256_extractf128_si256(a, 1);
  BMAS_ivech bh2 = _mm256_extractf128_si256(b, 1);
  BMAS_ivec ah2_16bit = _mm256_cvtepu8_epi16(ah2);
  BMAS_ivec bh2_16bit = _mm256_cvtepu8_epi16(bh2);
  BMAS_ivec v2 = _mm256_mullo_epi16(ah2_16bit, bh2_16bit);

  // Collect low 8 bits into the lower 64 bits of each 128 bit half-register
  v1 = _mm256_shuffle_epi8(v1, _mm256_setr_epi32(0x06040200, 0x0E0C0A08, 0x0, 0x0,
                                                 0x06040200, 0x0E0C0A08, 0x0, 0x0));
  v2 = _mm256_shuffle_epi8(v2, _mm256_setr_epi32(0x0, 0x0, 0x06040200, 0x0E0C0A08,
                                                 0x0, 0x0, 0x06040200, 0x0E0C0A08));

  // Collect low 8 bits into the lower 64 bits of each 128 bit half-register
  v1 = _mm256_and_si256(v1, _mm256_setr_epi64x(0xFFFFFFFFFFFFFFFF, 0x0000000000000000,
                                               0xFFFFFFFFFFFFFFFF, 0x0000000000000000));
  v2 = _mm256_and_si256(v2, _mm256_setr_epi64x(0x0000000000000000, 0xFFFFFFFFFFFFFFFF,
                                               0x0000000000000000, 0xFFFFFFFFFFFFFFFF));
  BMAS_ivec v = _mm256_or_si256(v1, v2);
  // Reorder to get in the correct order
  v = _mm256_permute4x64_epi64(v, 0b11011000);
  return v;
}


BMAS_ivec static inline BMAS_vector_i64abs(BMAS_ivec a){
  __m256i lt0_mask = _mm256_cmpgt_epi64(_mm256_setzero_si256(), a);
  // Take 2's complement of the negative
  __m256i a_mask   = _mm256_and_si256(lt0_mask, a);
  __m256i a_unmask = _mm256_andnot_si256(lt0_mask, a);
  __m256i max64bit = _mm256_and_si256(lt0_mask,
                                      _mm256_set_epi64x(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF,
                                                        0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF));
  __m256i a_lt0_inv = _mm256_sub_epi64(max64bit, a_mask);
  __m256i singlebit = _mm256_and_si256(lt0_mask, _mm256_set_epi64x(1,1,1,1));
  __m256i a_lt0_abs = _mm256_add_epi64(singlebit, a_lt0_inv);
  // Merge back after finding the abs
  return _mm256_or_si256(a_lt0_abs, a_unmask);
}
BMAS_ivec static inline BMAS_vector_i32abs(BMAS_ivec a){return _mm256_abs_epi32(a);}
BMAS_ivec static inline BMAS_vector_i16abs(BMAS_ivec a){return _mm256_abs_epi16(a);}
BMAS_ivec static inline BMAS_vector_i8abs (BMAS_ivec a){return _mm256_abs_epi8(a);}



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
