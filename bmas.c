#include <math.h>
#include <string.h>
#include <stdint.h>
#include "scalar.h"

#if defined(__x86_64)
  #include "simd/x86_64.h"
#elif defined(__aarch64__)
  #include "simd/aarch64.h"
#endif

BMAS_svech static inline BMAS_svech_make(
  float* ptr, const long stride, const int elt_size){
  BMAS_svech v;
  for(int i=0; i<SIMD_DOUBLE_STRIDE; i++) v[i] = ptr[i*stride];
  return v;
}
BMAS_svec static inline BMAS_svec_make(
  float* ptr, const long stride, const int elt_size){
  BMAS_svec v;
  for(int i=0; i<SIMD_SINGLE_STRIDE; i++) v[i] = ptr[i*stride];
  return v;
}
void static inline BMAS_svec_store_multi(
  BMAS_svec v, float* ptr, const long stride, const int elt_size){
  // TODO: Optimize this
  for(int i=0; i<SIMD_SINGLE_STRIDE; i++) (ptr+i*stride)[0] = v[i];
}
void static inline BMAS_svech_store_multi(
  BMAS_svech v, float* ptr, const long stride, const int elt_size){
  // TODO: Optimize this
  for(int i=0; i<SIMD_DOUBLE_STRIDE; i++) (ptr+i*stride)[0] = v[i];
}

BMAS_dvec static inline BMAS_dvec_make(
  double* ptr, const long stride, const int elt_size){
  BMAS_dvec v;
  for(int i=0; i<SIMD_DOUBLE_STRIDE; i++) v[i] = ptr[i*stride];
  return v;
}
void static inline BMAS_dvec_store_multi(
  BMAS_dvec v, double* ptr, const long stride, const int elt_size){
  for(int i=0; i<SIMD_DOUBLE_STRIDE; i++) (ptr+i*stride)[0] = v[i];
}


BMAS_ivec static inline BMAS_ivec_make(
  void* ptr, const long stride, const int elt_size){

  if (elt_size == 1){
    return BMAS_ivec_make_i8 (ptr, stride);
  }else if (elt_size == 2){
    return BMAS_ivec_make_i16(ptr, stride);
  }else if (elt_size == 4){
    return BMAS_ivec_make_i32(ptr, stride);
  }else if (elt_size == 8){
    return BMAS_ivec_make_i64(ptr, stride);
  }
}

BMAS_ivech static inline BMAS_ivech_make(
  void* ptr, const long stride, const int elt_size){

  if (elt_size == 1){
    return BMAS_ivech_make_i8 (ptr, stride);
  }else if (elt_size == 2){
    return BMAS_ivech_make_i16(ptr, stride);
  }else if (elt_size == 4){
    return BMAS_ivech_make_i32(ptr, stride);
  }else if (elt_size == 8){
    return BMAS_ivech_make_i64(ptr, stride);
  }
}

#include <stdio.h>
void static inline BMAS_ivec_store_multi(
  BMAS_ivec v, void* ptr, const long stride, const int elt_size){
  printf("elt_size: %d\n", elt_size);
  if (elt_size == 1){
    BMAS_ivec_store_multi_i8 (v, ptr, stride);
  }else if (elt_size == 2){
    BMAS_ivec_store_multi_i16(v, ptr, stride);
  }else if (elt_size == 4){
    BMAS_ivec_store_multi_i32(v, ptr, stride);
  }else if (elt_size == 8){
    BMAS_ivec_store_multi_i64(v, ptr, stride);
  }
}


#include "one_arg_fn_body.h"
#include "two_arg_fn_body.h"
#include "comparison.h"
#include "cast.h"
#include "copy.h"

one_arg_fn_body(ssin, SIMD_SINGLE_STRIDE, float, BMAS_svec, float, BMAS_svec);
one_arg_fn_body(scos, SIMD_SINGLE_STRIDE, float, BMAS_svec, float, BMAS_svec);
one_arg_fn_body(stan, SIMD_SINGLE_STRIDE, float, BMAS_svec, float, BMAS_svec);

one_arg_fn_body(sasin, SIMD_SINGLE_STRIDE, float, BMAS_svec, float, BMAS_svec);
one_arg_fn_body(sacos, SIMD_SINGLE_STRIDE, float, BMAS_svec, float, BMAS_svec);
one_arg_fn_body(satan, SIMD_SINGLE_STRIDE, float, BMAS_svec, float, BMAS_svec);

one_arg_fn_body(ssinh, SIMD_SINGLE_STRIDE, float, BMAS_svec, float, BMAS_svec);
one_arg_fn_body(scosh, SIMD_SINGLE_STRIDE, float, BMAS_svec, float, BMAS_svec);
one_arg_fn_body(stanh, SIMD_SINGLE_STRIDE, float, BMAS_svec, float, BMAS_svec);

one_arg_fn_body(sasinh, SIMD_SINGLE_STRIDE, float, BMAS_svec, float, BMAS_svec);
one_arg_fn_body(sacosh, SIMD_SINGLE_STRIDE, float, BMAS_svec, float, BMAS_svec);
one_arg_fn_body(satanh, SIMD_SINGLE_STRIDE, float, BMAS_svec, float, BMAS_svec);

one_arg_fn_body(dsin, SIMD_DOUBLE_STRIDE, double, BMAS_dvec, double, BMAS_dvec);
one_arg_fn_body(dcos, SIMD_DOUBLE_STRIDE, double, BMAS_dvec, double, BMAS_dvec);
one_arg_fn_body(dtan, SIMD_DOUBLE_STRIDE, double, BMAS_dvec, double, BMAS_dvec);

one_arg_fn_body(dasin, SIMD_DOUBLE_STRIDE, double, BMAS_dvec, double, BMAS_dvec);
one_arg_fn_body(dacos, SIMD_DOUBLE_STRIDE, double, BMAS_dvec, double, BMAS_dvec);
one_arg_fn_body(datan, SIMD_DOUBLE_STRIDE, double, BMAS_dvec, double, BMAS_dvec);

one_arg_fn_body(dsinh, SIMD_DOUBLE_STRIDE, double, BMAS_dvec, double, BMAS_dvec);
one_arg_fn_body(dcosh, SIMD_DOUBLE_STRIDE, double, BMAS_dvec, double, BMAS_dvec);
one_arg_fn_body(dtanh, SIMD_DOUBLE_STRIDE, double, BMAS_dvec, double, BMAS_dvec);

one_arg_fn_body(dasinh, SIMD_DOUBLE_STRIDE, double, BMAS_dvec, double, BMAS_dvec);
one_arg_fn_body(dacosh, SIMD_DOUBLE_STRIDE, double, BMAS_dvec, double, BMAS_dvec);
one_arg_fn_body(datanh, SIMD_DOUBLE_STRIDE, double, BMAS_dvec, double, BMAS_dvec);



one_arg_fn_body(slog,   SIMD_SINGLE_STRIDE, float, BMAS_svec, float, BMAS_svec);
one_arg_fn_body(slog10, SIMD_SINGLE_STRIDE, float, BMAS_svec, float, BMAS_svec);
one_arg_fn_body(slog2,  SIMD_SINGLE_STRIDE, float, BMAS_svec, float, BMAS_svec);
one_arg_fn_body(slog1p, SIMD_SINGLE_STRIDE, float, BMAS_svec, float, BMAS_svec);

one_arg_fn_body(dlog,   SIMD_DOUBLE_STRIDE, double, BMAS_dvec, double, BMAS_dvec);
one_arg_fn_body(dlog10, SIMD_DOUBLE_STRIDE, double, BMAS_dvec, double, BMAS_dvec);
one_arg_fn_body(dlog2,  SIMD_DOUBLE_STRIDE, double, BMAS_dvec, double, BMAS_dvec);
one_arg_fn_body(dlog1p, SIMD_DOUBLE_STRIDE, double, BMAS_dvec, double, BMAS_dvec);



one_arg_fn_body(sexp,   SIMD_SINGLE_STRIDE, float, BMAS_svec, float, BMAS_svec);
one_arg_fn_body(sexp10, SIMD_SINGLE_STRIDE, float, BMAS_svec, float, BMAS_svec);
one_arg_fn_body(sexp2,  SIMD_SINGLE_STRIDE, float, BMAS_svec, float, BMAS_svec);
one_arg_fn_body(sexpm1, SIMD_SINGLE_STRIDE, float, BMAS_svec, float, BMAS_svec);

one_arg_fn_body(dexp,   SIMD_DOUBLE_STRIDE, double, BMAS_dvec, double, BMAS_dvec);
one_arg_fn_body(dexp10, SIMD_DOUBLE_STRIDE, double, BMAS_dvec, double, BMAS_dvec);
one_arg_fn_body(dexp2,  SIMD_DOUBLE_STRIDE, double, BMAS_dvec, double, BMAS_dvec);
one_arg_fn_body(dexpm1, SIMD_DOUBLE_STRIDE, double, BMAS_dvec, double, BMAS_dvec);


// We aren't adding rint because it has no common-lisp equivalent
// - none I know of

one_arg_fn_body(sfabs,  SIMD_SINGLE_STRIDE, float, BMAS_svec, float, BMAS_svec);
one_arg_fn_body(sceil,  SIMD_SINGLE_STRIDE, float, BMAS_svec, float, BMAS_svec);
one_arg_fn_body(strunc, SIMD_SINGLE_STRIDE, float, BMAS_svec, float, BMAS_svec);
one_arg_fn_body(sfloor, SIMD_SINGLE_STRIDE, float, BMAS_svec, float, BMAS_svec);
one_arg_fn_body(sround, SIMD_SINGLE_STRIDE, float, BMAS_svec, float, BMAS_svec);

one_arg_fn_body(dfabs,  SIMD_DOUBLE_STRIDE, double, BMAS_dvec, double, BMAS_dvec);
one_arg_fn_body(dceil,  SIMD_DOUBLE_STRIDE, double, BMAS_dvec, double, BMAS_dvec);
one_arg_fn_body(dtrunc, SIMD_DOUBLE_STRIDE, double, BMAS_dvec, double, BMAS_dvec);
one_arg_fn_body(dfloor, SIMD_DOUBLE_STRIDE, double, BMAS_dvec, double, BMAS_dvec);
one_arg_fn_body(dround, SIMD_DOUBLE_STRIDE, double, BMAS_dvec, double, BMAS_dvec);



two_arg_fn_body(spow,   SIMD_SINGLE_STRIDE, float, BMAS_svec, float, BMAS_svec);
two_arg_fn_body(satan2, SIMD_SINGLE_STRIDE, float, BMAS_svec, float, BMAS_svec);
two_arg_fn_body(sadd,   SIMD_SINGLE_STRIDE, float, BMAS_svec, float, BMAS_svec);
two_arg_fn_body(ssub,   SIMD_SINGLE_STRIDE, float, BMAS_svec, float, BMAS_svec);
two_arg_fn_body(smul,   SIMD_SINGLE_STRIDE, float, BMAS_svec, float, BMAS_svec);
two_arg_fn_body(sdiv,   SIMD_SINGLE_STRIDE, float, BMAS_svec, float, BMAS_svec);

two_arg_fn_body(dpow,   SIMD_DOUBLE_STRIDE, double, BMAS_dvec, double, BMAS_dvec);
two_arg_fn_body(datan2, SIMD_DOUBLE_STRIDE, double, BMAS_dvec, double, BMAS_dvec);
two_arg_fn_body(dadd,   SIMD_DOUBLE_STRIDE, double, BMAS_dvec, double, BMAS_dvec);
two_arg_fn_body(dsub,   SIMD_DOUBLE_STRIDE, double, BMAS_dvec, double, BMAS_dvec);
two_arg_fn_body(dmul,   SIMD_DOUBLE_STRIDE, double, BMAS_dvec, double, BMAS_dvec);
two_arg_fn_body(ddiv,   SIMD_DOUBLE_STRIDE, double, BMAS_dvec, double, BMAS_dvec);


two_arg_fn_body_comparison(slt,  SIMD_SINGLE_STRIDE, float, BMAS_svec, BMAS_sbool);
two_arg_fn_body_comparison(sle,  SIMD_SINGLE_STRIDE, float, BMAS_svec, BMAS_sbool);
two_arg_fn_body_comparison(seq,  SIMD_SINGLE_STRIDE, float, BMAS_svec, BMAS_sbool);
two_arg_fn_body_comparison(sneq, SIMD_SINGLE_STRIDE, float, BMAS_svec, BMAS_sbool);
two_arg_fn_body_comparison(sge,  SIMD_SINGLE_STRIDE, float, BMAS_svec, BMAS_sbool);
two_arg_fn_body_comparison(sgt,  SIMD_SINGLE_STRIDE, float, BMAS_svec, BMAS_sbool);

two_arg_fn_body_comparison(dlt,  SIMD_DOUBLE_STRIDE, double, BMAS_dvec, BMAS_dbool);
two_arg_fn_body_comparison(dle,  SIMD_DOUBLE_STRIDE, double, BMAS_dvec, BMAS_dbool);
two_arg_fn_body_comparison(deq,  SIMD_DOUBLE_STRIDE, double, BMAS_dvec, BMAS_dbool);
two_arg_fn_body_comparison(dneq, SIMD_DOUBLE_STRIDE, double, BMAS_dvec, BMAS_dbool);
two_arg_fn_body_comparison(dge,  SIMD_DOUBLE_STRIDE, double, BMAS_dvec, BMAS_dbool);
two_arg_fn_body_comparison(dgt,  SIMD_DOUBLE_STRIDE, double, BMAS_dvec, BMAS_dbool);


// Integer Arithmetic

two_arg_fn_body(i64add, SIMD_SINGLE_STRIDE/2, int64_t, BMAS_ivec, int64_t, BMAS_ivec);
two_arg_fn_body(i32add, SIMD_SINGLE_STRIDE,   int32_t, BMAS_ivec, int32_t, BMAS_ivec);
two_arg_fn_body(i16add, SIMD_SINGLE_STRIDE*2, int16_t, BMAS_ivec, int16_t, BMAS_ivec);
two_arg_fn_body(i8add,  SIMD_SINGLE_STRIDE*4, int8_t,  BMAS_ivec, int8_t,  BMAS_ivec);

two_arg_fn_body(i64sub, SIMD_SINGLE_STRIDE/2, int64_t, BMAS_ivec, int64_t, BMAS_ivec);
two_arg_fn_body(i32sub, SIMD_SINGLE_STRIDE,   int32_t, BMAS_ivec, int32_t, BMAS_ivec);
two_arg_fn_body(i16sub, SIMD_SINGLE_STRIDE*2, int16_t, BMAS_ivec, int16_t, BMAS_ivec);
two_arg_fn_body(i8sub,  SIMD_SINGLE_STRIDE*4, int8_t,  BMAS_ivec, int8_t,  BMAS_ivec);


/* // two_arg_fn_body(copysign); */
/* // two_arg_fn_body(fmax); */
/* // two_arg_fn_body(fmin); */
/* // two_arg_fn_body(fdim); */

/* // two_arg_fn_body(hypot); */
