#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "macro_loop.h"
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
#include "one_arg_reduce_fn_body.h"
#include "one_arg_ireduce_fn_body.h"
#include "dot_fn_body.h"

copy_fn_body(s,     SIMD_SINGLE_STRIDE, float,   BMAS_svec);
copy_fn_body(d,     SIMD_DOUBLE_STRIDE, double,  BMAS_dvec);
copy_fn_body(i8,  4*SIMD_SINGLE_STRIDE, int8_t,  BMAS_ivec);
copy_fn_body(i16, 2*SIMD_SINGLE_STRIDE, int16_t, BMAS_ivec);
copy_fn_body(i32,   SIMD_SINGLE_STRIDE, int32_t, BMAS_ivec);
copy_fn_body(i64,   SIMD_DOUBLE_STRIDE, int64_t, BMAS_ivec);

cast_to_float_body(d,   SIMD_DOUBLE_STRIDE, double,   BMAS_dvec,  BMAS_svech, BMAS_dvec_to_svech);
cast_to_float_body(i8,  SIMD_SINGLE_STRIDE, int8_t,   BMAS_ivech, BMAS_svec,  BMAS_ivech_to_svec_i8);
cast_to_float_body(i16, SIMD_SINGLE_STRIDE, int16_t,  BMAS_ivech, BMAS_svec,  BMAS_ivech_to_svec_i16);
cast_to_float_body(i32, SIMD_SINGLE_STRIDE, int32_t,  BMAS_ivec,  BMAS_svec,  BMAS_ivec_to_svec_i32);
cast_to_float_body(i64, SIMD_DOUBLE_STRIDE, int64_t,  BMAS_ivec,  BMAS_svech, BMAS_ivec_to_svech_i64);
cast_to_float_body(u8,  SIMD_SINGLE_STRIDE, uint8_t,  BMAS_ivech, BMAS_svec,  BMAS_ivech_to_svec_u8);
cast_to_float_body(u16, SIMD_SINGLE_STRIDE, uint16_t, BMAS_ivech, BMAS_svec,  BMAS_ivech_to_svec_u16);
cast_to_float_body(u32, SIMD_DOUBLE_STRIDE, uint32_t, BMAS_ivech, BMAS_svech, BMAS_ivech_to_svec_u32);
cast_to_float_body(u64, SIMD_DOUBLE_STRIDE, uint64_t, BMAS_ivec,  BMAS_svech, BMAS_ivec_to_svech_u64);

// int8_t can be useful for tasks like counting _Bool elements
one_arg_reduce_fn_body(ssum,     SIMD_SINGLE_STRIDE, float,   BMAS_svec, float,   szero, sadd,   shadd);
one_arg_reduce_fn_body(dsum,     SIMD_DOUBLE_STRIDE, double,  BMAS_dvec, double,  dzero, dadd,   dhadd);
one_arg_reduce_fn_body(i64sum,   SIMD_DOUBLE_STRIDE, int64_t, BMAS_ivec, int64_t, izero, i64add, i64hadd);
one_arg_reduce_fn_body(i32sum,   SIMD_SINGLE_STRIDE, int32_t, BMAS_ivec, int32_t, izero, i32add, i32hadd);
one_arg_reduce_fn_body(i16sum, 2*SIMD_SINGLE_STRIDE, int16_t, BMAS_ivec, int16_t, izero, i16add, i16hadd);
one_arg_reduce_fn_body(i8sum,  4*SIMD_SINGLE_STRIDE, int8_t,  BMAS_ivec, int8_t,  izero, i8add,  i8hadd);

one_arg_reduce_fn_body(shmax,     SIMD_SINGLE_STRIDE, float,    BMAS_svec, float,    sMIN, smax,   shmax);
one_arg_reduce_fn_body(dhmax,     SIMD_DOUBLE_STRIDE, double,   BMAS_dvec, double,   dMIN, dmax,   dhmax);
one_arg_reduce_fn_body(i64hmax,   SIMD_DOUBLE_STRIDE, int64_t,  BMAS_ivec, int64_t,  i64MIN, i64max, i64hmax);
one_arg_reduce_fn_body(i32hmax,   SIMD_SINGLE_STRIDE, int32_t,  BMAS_ivec, int32_t,  i32MIN, i32max, i32hmax);
one_arg_reduce_fn_body(i16hmax, 2*SIMD_SINGLE_STRIDE, int16_t,  BMAS_ivec, int16_t,  i16MIN, i16max, i16hmax);
one_arg_reduce_fn_body(i8hmax,  4*SIMD_SINGLE_STRIDE, int8_t,   BMAS_ivec, int8_t,   i8MIN,  i8max,  i8hmax);
one_arg_reduce_fn_body(u64hmax,   SIMD_DOUBLE_STRIDE, uint64_t, BMAS_ivec, uint64_t, izero, u64max, u64hmax);
one_arg_reduce_fn_body(u32hmax,   SIMD_SINGLE_STRIDE, uint32_t, BMAS_ivec, uint32_t, izero, u32max, u32hmax);
one_arg_reduce_fn_body(u16hmax, 2*SIMD_SINGLE_STRIDE, uint16_t, BMAS_ivec, uint16_t, izero, u16max, u16hmax);
one_arg_reduce_fn_body(u8hmax,  4*SIMD_SINGLE_STRIDE, uint8_t,  BMAS_ivec, uint8_t,  izero, u8max,  u8hmax);

one_arg_reduce_fn_body(shmin,     SIMD_SINGLE_STRIDE, float,    BMAS_svec, float,    sMAX, smin,   shmin);
one_arg_reduce_fn_body(dhmin,     SIMD_DOUBLE_STRIDE, double,   BMAS_dvec, double,   dMAX, dmin,   dhmin);
one_arg_reduce_fn_body(i64hmin,   SIMD_DOUBLE_STRIDE, int64_t,  BMAS_ivec, int64_t,  i64MAX, i64min, i64hmin);
one_arg_reduce_fn_body(i32hmin,   SIMD_SINGLE_STRIDE, int32_t,  BMAS_ivec, int32_t,  i32MAX, i32min, i32hmin);
one_arg_reduce_fn_body(i16hmin, 2*SIMD_SINGLE_STRIDE, int16_t,  BMAS_ivec, int16_t,  i16MAX, i16min, i16hmin);
one_arg_reduce_fn_body(i8hmin,  4*SIMD_SINGLE_STRIDE, int8_t,   BMAS_ivec, int8_t,   i8MAX,  i8min,  i8hmin);
one_arg_reduce_fn_body(u64hmin,   SIMD_DOUBLE_STRIDE, uint64_t, BMAS_ivec, uint64_t, uMAX, u64min, u64hmin);
one_arg_reduce_fn_body(u32hmin,   SIMD_SINGLE_STRIDE, uint32_t, BMAS_ivec, uint32_t, uMAX, u32min, u32hmin);
one_arg_reduce_fn_body(u16hmin, 2*SIMD_SINGLE_STRIDE, uint16_t, BMAS_ivec, uint16_t, uMAX, u16min, u16hmin);
one_arg_reduce_fn_body(u8hmin,  4*SIMD_SINGLE_STRIDE, uint8_t,  BMAS_ivec, uint8_t,  uMAX, u8min,  u8hmin);

one_arg_ireduce_fn_body(shimax,     SIMD_SINGLE_STRIDE, float,    svec, sINDEX, sMIN,   sindex,   shindex,   'a', sgt);
one_arg_ireduce_fn_body(dhimax,     SIMD_DOUBLE_STRIDE, double,   dvec, dINDEX, dMIN,   dindex,   dhindex,   'a', dgt);
one_arg_ireduce_fn_body(i64himax,   SIMD_DOUBLE_STRIDE, int64_t,  ivec, iINDEX, i64MIN, i64index, i64hindex, 'a', i64gt);
one_arg_ireduce_fn_body(i32himax,   SIMD_SINGLE_STRIDE, int32_t,  ivec, iINDEX, i32MIN, i32index, i32hindex, 'a', i32gt);
one_arg_ireduce_fn_body(i16himax, 2*SIMD_SINGLE_STRIDE, int16_t,  ivec, iINDEX, i16MIN, i16index, i16hindex, 'a', i16gt);
one_arg_ireduce_fn_body(i8himax,  4*SIMD_SINGLE_STRIDE, int8_t,   ivec, iINDEX, i8MIN,  i8index,  i8hindex,  'a', i8gt);
one_arg_ireduce_fn_body(u64himax,   SIMD_DOUBLE_STRIDE, uint64_t, ivec, iINDEX, uMIN,   u64index, u64hindex, 'a', u64gt);
one_arg_ireduce_fn_body(u32himax,   SIMD_SINGLE_STRIDE, uint32_t, ivec, iINDEX, uMIN,   u32index, u32hindex, 'a', u32gt);
one_arg_ireduce_fn_body(u16himax, 2*SIMD_SINGLE_STRIDE, uint16_t, ivec, iINDEX, uMIN,   u16index, u16hindex, 'a', u16gt);
one_arg_ireduce_fn_body(u8himax,  4*SIMD_SINGLE_STRIDE, uint8_t,  ivec, iINDEX, uMIN,   u8index,  u8hindex,  'a', u8gt);

one_arg_ireduce_fn_body(shimin,     SIMD_SINGLE_STRIDE, float,    svec, sINDEX, sMAX,   sindex,   shindex,   'i', slt);
one_arg_ireduce_fn_body(dhimin,     SIMD_DOUBLE_STRIDE, double,   dvec, dINDEX, dMAX,   dindex,   dhindex,   'i', dlt);
one_arg_ireduce_fn_body(i64himin,   SIMD_DOUBLE_STRIDE, int64_t,  ivec, iINDEX, i64MAX, i64index, i64hindex, 'i', i64lt);
one_arg_ireduce_fn_body(i32himin,   SIMD_SINGLE_STRIDE, int32_t,  ivec, iINDEX, i32MAX, i32index, i32hindex, 'i', i32lt);
one_arg_ireduce_fn_body(i16himin, 2*SIMD_SINGLE_STRIDE, int16_t,  ivec, iINDEX, i16MAX, i16index, i16hindex, 'i', i16lt);
one_arg_ireduce_fn_body(i8himin,  4*SIMD_SINGLE_STRIDE, int8_t,   ivec, iINDEX, i8MAX,  i8index,  i8hindex,  'i', i8lt);
one_arg_ireduce_fn_body(u64himin,   SIMD_DOUBLE_STRIDE, uint64_t, ivec, iINDEX, uMAX,   u64index, u64hindex, 'i', u64lt);
one_arg_ireduce_fn_body(u32himin,   SIMD_SINGLE_STRIDE, uint32_t, ivec, iINDEX, uMAX,   u32index, u32hindex, 'i', u32lt);
one_arg_ireduce_fn_body(u16himin, 2*SIMD_SINGLE_STRIDE, uint16_t, ivec, iINDEX, uMAX,   u16index, u16hindex, 'i', u16lt);
one_arg_ireduce_fn_body(u8himin,  4*SIMD_SINGLE_STRIDE, uint8_t,  ivec, iINDEX, uMAX,   u8index,  u8hindex,  'i', u8lt);


dot_fn_body(sdot,     SIMD_SINGLE_STRIDE, float,   BMAS_svec, float,   szero, sadd, smul, shadd);
dot_fn_body(ddot,     SIMD_DOUBLE_STRIDE, double,  BMAS_dvec, double,  dzero, dadd, dmul, dhadd);
dot_fn_body(i64dot,   SIMD_DOUBLE_STRIDE, int64_t, BMAS_ivec, int64_t, izero, i64add, i64mul, i64hadd);
dot_fn_body(i32dot,   SIMD_SINGLE_STRIDE, int32_t, BMAS_ivec, int32_t, izero, i32add, i32mul, i32hadd);
dot_fn_body(i16dot, 2*SIMD_SINGLE_STRIDE, int16_t, BMAS_ivec, int16_t, izero, i16add, i16mul, i16hadd);
dot_fn_body(i8dot,  4*SIMD_SINGLE_STRIDE, int8_t,  BMAS_ivec, int8_t,  izero, i8add,  i8mul,  i8hadd);
dot_fn_body(u64dot,   SIMD_DOUBLE_STRIDE, uint64_t, BMAS_ivec, uint64_t, izero, i64add, u64mul, i64hadd);
dot_fn_body(u32dot,   SIMD_SINGLE_STRIDE, uint32_t, BMAS_ivec, uint32_t, izero, i32add, u32mul, i32hadd);
dot_fn_body(u16dot, 2*SIMD_SINGLE_STRIDE, uint16_t, BMAS_ivec, uint16_t, izero, i16add, u16mul, i16hadd);
dot_fn_body(u8dot,  4*SIMD_SINGLE_STRIDE, uint8_t,  BMAS_ivec, uint8_t,  izero, i8add,  u8mul,  i8hadd);


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

one_arg_fn_body(ssqrt, SIMD_SINGLE_STRIDE, float,  BMAS_svec, float,  BMAS_svec);
one_arg_fn_body(dsqrt, SIMD_DOUBLE_STRIDE, double, BMAS_dvec, double, BMAS_dvec);


two_arg_fn_body(spow,   SIMD_SINGLE_STRIDE, float, BMAS_svec, float, BMAS_svec);
two_arg_fn_body(satan2, SIMD_SINGLE_STRIDE, float, BMAS_svec, float, BMAS_svec);
two_arg_fn_body(sadd,   SIMD_SINGLE_STRIDE, float, BMAS_svec, float, BMAS_svec);
two_arg_fn_body(ssub,   SIMD_SINGLE_STRIDE, float, BMAS_svec, float, BMAS_svec);
two_arg_fn_body(smul,   SIMD_SINGLE_STRIDE, float, BMAS_svec, float, BMAS_svec);
two_arg_fn_body(sdiv,   SIMD_SINGLE_STRIDE, float, BMAS_svec, float, BMAS_svec);
two_arg_fn_body(smin,   SIMD_SINGLE_STRIDE, float, BMAS_svec, float, BMAS_svec);
two_arg_fn_body(smax,   SIMD_SINGLE_STRIDE, float, BMAS_svec, float, BMAS_svec);

two_arg_fn_body(dpow,   SIMD_DOUBLE_STRIDE, double, BMAS_dvec, double, BMAS_dvec);
two_arg_fn_body(datan2, SIMD_DOUBLE_STRIDE, double, BMAS_dvec, double, BMAS_dvec);
two_arg_fn_body(dadd,   SIMD_DOUBLE_STRIDE, double, BMAS_dvec, double, BMAS_dvec);
two_arg_fn_body(dsub,   SIMD_DOUBLE_STRIDE, double, BMAS_dvec, double, BMAS_dvec);
two_arg_fn_body(dmul,   SIMD_DOUBLE_STRIDE, double, BMAS_dvec, double, BMAS_dvec);
two_arg_fn_body(ddiv,   SIMD_DOUBLE_STRIDE, double, BMAS_dvec, double, BMAS_dvec);
two_arg_fn_body(dmin,   SIMD_DOUBLE_STRIDE, double, BMAS_dvec, double, BMAS_dvec);
two_arg_fn_body(dmax,   SIMD_DOUBLE_STRIDE, double, BMAS_dvec, double, BMAS_dvec);


two_arg_fn_body_comparisonx4(slt,  SIMD_SINGLE_STRIDE, float, BMAS_svec, BMAS_sbool);
two_arg_fn_body_comparisonx4(sle,  SIMD_SINGLE_STRIDE, float, BMAS_svec, BMAS_sbool);
two_arg_fn_body_comparisonx4(seq,  SIMD_SINGLE_STRIDE, float, BMAS_svec, BMAS_sbool);
two_arg_fn_body_comparisonx4(sneq, SIMD_SINGLE_STRIDE, float, BMAS_svec, BMAS_sbool);
two_arg_fn_body_comparisonx4(sge,  SIMD_SINGLE_STRIDE, float, BMAS_svec, BMAS_sbool);
two_arg_fn_body_comparisonx4(sgt,  SIMD_SINGLE_STRIDE, float, BMAS_svec, BMAS_sbool);

two_arg_fn_body_comparisonx4(dlt,  SIMD_DOUBLE_STRIDE, double, BMAS_dvec, BMAS_dbool);
two_arg_fn_body_comparisonx4(dle,  SIMD_DOUBLE_STRIDE, double, BMAS_dvec, BMAS_dbool);
two_arg_fn_body_comparisonx4(deq,  SIMD_DOUBLE_STRIDE, double, BMAS_dvec, BMAS_dbool);
two_arg_fn_body_comparisonx4(dneq, SIMD_DOUBLE_STRIDE, double, BMAS_dvec, BMAS_dbool);
two_arg_fn_body_comparisonx4(dge,  SIMD_DOUBLE_STRIDE, double, BMAS_dvec, BMAS_dbool);
two_arg_fn_body_comparisonx4(dgt,  SIMD_DOUBLE_STRIDE, double, BMAS_dvec, BMAS_dbool);


two_arg_fn_body_comparisonx1(i8lt,  4*SIMD_SINGLE_STRIDE, int8_t, BMAS_ivec, BMAS_ivec);
two_arg_fn_body_comparisonx1(i8le,  4*SIMD_SINGLE_STRIDE, int8_t, BMAS_ivec, BMAS_ivec);
two_arg_fn_body_comparisonx1(i8eq,  4*SIMD_SINGLE_STRIDE, int8_t, BMAS_ivec, BMAS_ivec);
two_arg_fn_body_comparisonx1(i8neq, 4*SIMD_SINGLE_STRIDE, int8_t, BMAS_ivec, BMAS_ivec);
two_arg_fn_body_comparisonx1(i8ge,  4*SIMD_SINGLE_STRIDE, int8_t, BMAS_ivec, BMAS_ivec);
two_arg_fn_body_comparisonx1(i8gt,  4*SIMD_SINGLE_STRIDE, int8_t, BMAS_ivec, BMAS_ivec);

two_arg_fn_body_comparisonx2(i16lt,  2*SIMD_SINGLE_STRIDE, int16_t, BMAS_ivec, BMAS_ivec);
two_arg_fn_body_comparisonx2(i16le,  2*SIMD_SINGLE_STRIDE, int16_t, BMAS_ivec, BMAS_ivec);
two_arg_fn_body_comparisonx2(i16eq,  2*SIMD_SINGLE_STRIDE, int16_t, BMAS_ivec, BMAS_ivec);
two_arg_fn_body_comparisonx2(i16neq, 2*SIMD_SINGLE_STRIDE, int16_t, BMAS_ivec, BMAS_ivec);
two_arg_fn_body_comparisonx2(i16ge,  2*SIMD_SINGLE_STRIDE, int16_t, BMAS_ivec, BMAS_ivec);
two_arg_fn_body_comparisonx2(i16gt,  2*SIMD_SINGLE_STRIDE, int16_t, BMAS_ivec, BMAS_ivec);

two_arg_fn_body_comparisonx4(i32lt,  SIMD_SINGLE_STRIDE, int32_t, BMAS_ivec, BMAS_ivec);
two_arg_fn_body_comparisonx4(i32le,  SIMD_SINGLE_STRIDE, int32_t, BMAS_ivec, BMAS_ivec);
two_arg_fn_body_comparisonx4(i32eq,  SIMD_SINGLE_STRIDE, int32_t, BMAS_ivec, BMAS_ivec);
two_arg_fn_body_comparisonx4(i32neq, SIMD_SINGLE_STRIDE, int32_t, BMAS_ivec, BMAS_ivec);
two_arg_fn_body_comparisonx4(i32ge,  SIMD_SINGLE_STRIDE, int32_t, BMAS_ivec, BMAS_ivec);
two_arg_fn_body_comparisonx4(i32gt,  SIMD_SINGLE_STRIDE, int32_t, BMAS_ivec, BMAS_ivec);

two_arg_fn_body_comparisonx4(i64lt,  SIMD_DOUBLE_STRIDE, int64_t, BMAS_ivec, BMAS_ivec);
two_arg_fn_body_comparisonx4(i64le,  SIMD_DOUBLE_STRIDE, int64_t, BMAS_ivec, BMAS_ivec);
two_arg_fn_body_comparisonx4(i64eq,  SIMD_DOUBLE_STRIDE, int64_t, BMAS_ivec, BMAS_ivec);
two_arg_fn_body_comparisonx4(i64neq, SIMD_DOUBLE_STRIDE, int64_t, BMAS_ivec, BMAS_ivec);
two_arg_fn_body_comparisonx4(i64ge,  SIMD_DOUBLE_STRIDE, int64_t, BMAS_ivec, BMAS_ivec);
two_arg_fn_body_comparisonx4(i64gt,  SIMD_DOUBLE_STRIDE, int64_t, BMAS_ivec, BMAS_ivec);


two_arg_fn_body_comparisonx1(u8lt,  4*SIMD_SINGLE_STRIDE, int8_t, BMAS_ivec, BMAS_ivec);
two_arg_fn_body_comparisonx1(u8le,  4*SIMD_SINGLE_STRIDE, int8_t, BMAS_ivec, BMAS_ivec);
two_arg_fn_body_comparisonx1(u8eq,  4*SIMD_SINGLE_STRIDE, int8_t, BMAS_ivec, BMAS_ivec);
two_arg_fn_body_comparisonx1(u8neq, 4*SIMD_SINGLE_STRIDE, int8_t, BMAS_ivec, BMAS_ivec);
two_arg_fn_body_comparisonx1(u8ge,  4*SIMD_SINGLE_STRIDE, int8_t, BMAS_ivec, BMAS_ivec);
two_arg_fn_body_comparisonx1(u8gt,  4*SIMD_SINGLE_STRIDE, int8_t, BMAS_ivec, BMAS_ivec);

two_arg_fn_body_comparisonx2(u16lt,  2*SIMD_SINGLE_STRIDE, int16_t, BMAS_ivec, BMAS_ivec);
two_arg_fn_body_comparisonx2(u16le,  2*SIMD_SINGLE_STRIDE, int16_t, BMAS_ivec, BMAS_ivec);
two_arg_fn_body_comparisonx2(u16eq,  2*SIMD_SINGLE_STRIDE, int16_t, BMAS_ivec, BMAS_ivec);
two_arg_fn_body_comparisonx2(u16neq, 2*SIMD_SINGLE_STRIDE, int16_t, BMAS_ivec, BMAS_ivec);
two_arg_fn_body_comparisonx2(u16ge,  2*SIMD_SINGLE_STRIDE, int16_t, BMAS_ivec, BMAS_ivec);
two_arg_fn_body_comparisonx2(u16gt,  2*SIMD_SINGLE_STRIDE, int16_t, BMAS_ivec, BMAS_ivec);

two_arg_fn_body_comparisonx4(u32lt,  SIMD_SINGLE_STRIDE, int32_t, BMAS_ivec, BMAS_ivec);
two_arg_fn_body_comparisonx4(u32le,  SIMD_SINGLE_STRIDE, int32_t, BMAS_ivec, BMAS_ivec);
two_arg_fn_body_comparisonx4(u32eq,  SIMD_SINGLE_STRIDE, int32_t, BMAS_ivec, BMAS_ivec);
two_arg_fn_body_comparisonx4(u32neq, SIMD_SINGLE_STRIDE, int32_t, BMAS_ivec, BMAS_ivec);
two_arg_fn_body_comparisonx4(u32ge,  SIMD_SINGLE_STRIDE, int32_t, BMAS_ivec, BMAS_ivec);
two_arg_fn_body_comparisonx4(u32gt,  SIMD_SINGLE_STRIDE, int32_t, BMAS_ivec, BMAS_ivec);

two_arg_fn_body_comparisonx4(u64lt,  SIMD_DOUBLE_STRIDE, int64_t, BMAS_ivec, BMAS_ivec);
two_arg_fn_body_comparisonx4(u64le,  SIMD_DOUBLE_STRIDE, int64_t, BMAS_ivec, BMAS_ivec);
two_arg_fn_body_comparisonx4(u64eq,  SIMD_DOUBLE_STRIDE, int64_t, BMAS_ivec, BMAS_ivec);
two_arg_fn_body_comparisonx4(u64neq, SIMD_DOUBLE_STRIDE, int64_t, BMAS_ivec, BMAS_ivec);
two_arg_fn_body_comparisonx4(u64ge,  SIMD_DOUBLE_STRIDE, int64_t, BMAS_ivec, BMAS_ivec);
two_arg_fn_body_comparisonx4(u64gt,  SIMD_DOUBLE_STRIDE, int64_t, BMAS_ivec, BMAS_ivec);


// Integer Arithmetic, Absolute, Minimum, Maximum

two_arg_fn_body(i64add, SIMD_SINGLE_STRIDE/2, int64_t, BMAS_ivec, int64_t, BMAS_ivec);
two_arg_fn_body(i32add, SIMD_SINGLE_STRIDE,   int32_t, BMAS_ivec, int32_t, BMAS_ivec);
two_arg_fn_body(i16add, SIMD_SINGLE_STRIDE*2, int16_t, BMAS_ivec, int16_t, BMAS_ivec);
two_arg_fn_body(i8add,  SIMD_SINGLE_STRIDE*4, int8_t,  BMAS_ivec, int8_t,  BMAS_ivec);

two_arg_fn_body(i64sub, SIMD_SINGLE_STRIDE/2, int64_t, BMAS_ivec, int64_t, BMAS_ivec);
two_arg_fn_body(i32sub, SIMD_SINGLE_STRIDE,   int32_t, BMAS_ivec, int32_t, BMAS_ivec);
two_arg_fn_body(i16sub, SIMD_SINGLE_STRIDE*2, int16_t, BMAS_ivec, int16_t, BMAS_ivec);
two_arg_fn_body(i8sub,  SIMD_SINGLE_STRIDE*4, int8_t,  BMAS_ivec, int8_t,  BMAS_ivec);

two_arg_fn_body(i64mul, SIMD_SINGLE_STRIDE/2, int64_t, BMAS_ivec, int64_t, BMAS_ivec);
two_arg_fn_body(i32mul, SIMD_SINGLE_STRIDE,   int32_t, BMAS_ivec, int32_t, BMAS_ivec);
two_arg_fn_body(i16mul, SIMD_SINGLE_STRIDE*2, int16_t, BMAS_ivec, int16_t, BMAS_ivec);
two_arg_fn_body(i8mul,  SIMD_SINGLE_STRIDE*4, int8_t,  BMAS_ivec, int8_t,  BMAS_ivec);

two_arg_fn_body(u64mul, SIMD_SINGLE_STRIDE/2, uint64_t, BMAS_ivec, uint64_t, BMAS_ivec);
two_arg_fn_body(u32mul, SIMD_SINGLE_STRIDE,   uint32_t, BMAS_ivec, uint32_t, BMAS_ivec);
two_arg_fn_body(u16mul, SIMD_SINGLE_STRIDE*2, uint16_t, BMAS_ivec, uint16_t, BMAS_ivec);
two_arg_fn_body(u8mul,  SIMD_SINGLE_STRIDE*4, uint8_t,  BMAS_ivec, uint8_t,  BMAS_ivec);


one_arg_fn_body(i64abs, SIMD_SINGLE_STRIDE/2, int64_t, BMAS_ivec, int64_t, BMAS_ivec);
one_arg_fn_body(i32abs, SIMD_SINGLE_STRIDE,   int32_t, BMAS_ivec, int32_t, BMAS_ivec);
one_arg_fn_body(i16abs, SIMD_SINGLE_STRIDE*2, int16_t, BMAS_ivec, int16_t, BMAS_ivec);
one_arg_fn_body(i8abs,  SIMD_SINGLE_STRIDE*4, int8_t,  BMAS_ivec, int8_t,  BMAS_ivec);


two_arg_fn_body(i64min, SIMD_SINGLE_STRIDE/2, int64_t, BMAS_ivec, int64_t, BMAS_ivec);
two_arg_fn_body(i32min, SIMD_SINGLE_STRIDE,   int32_t, BMAS_ivec, int32_t, BMAS_ivec);
two_arg_fn_body(i16min, SIMD_SINGLE_STRIDE*2, int16_t, BMAS_ivec, int16_t, BMAS_ivec);
two_arg_fn_body(i8min,  SIMD_SINGLE_STRIDE*4, int8_t,  BMAS_ivec, int8_t,  BMAS_ivec);

two_arg_fn_body(u64min, SIMD_SINGLE_STRIDE/2, int64_t, BMAS_ivec, int64_t, BMAS_ivec);
two_arg_fn_body(u32min, SIMD_SINGLE_STRIDE,   int32_t, BMAS_ivec, int32_t, BMAS_ivec);
two_arg_fn_body(u16min, SIMD_SINGLE_STRIDE*2, int16_t, BMAS_ivec, int16_t, BMAS_ivec);
two_arg_fn_body(u8min,  SIMD_SINGLE_STRIDE*4, int8_t,  BMAS_ivec, int8_t,  BMAS_ivec);

two_arg_fn_body(i64max, SIMD_SINGLE_STRIDE/2, int64_t, BMAS_ivec, int64_t, BMAS_ivec);
two_arg_fn_body(i32max, SIMD_SINGLE_STRIDE,   int32_t, BMAS_ivec, int32_t, BMAS_ivec);
two_arg_fn_body(i16max, SIMD_SINGLE_STRIDE*2, int16_t, BMAS_ivec, int16_t, BMAS_ivec);
two_arg_fn_body(i8max,  SIMD_SINGLE_STRIDE*4, int8_t,  BMAS_ivec, int8_t,  BMAS_ivec);

two_arg_fn_body(u64max, SIMD_SINGLE_STRIDE/2, int64_t, BMAS_ivec, int64_t, BMAS_ivec);
two_arg_fn_body(u32max, SIMD_SINGLE_STRIDE,   int32_t, BMAS_ivec, int32_t, BMAS_ivec);
two_arg_fn_body(u16max, SIMD_SINGLE_STRIDE*2, int16_t, BMAS_ivec, int16_t, BMAS_ivec);
two_arg_fn_body(u8max,  SIMD_SINGLE_STRIDE*4, int8_t,  BMAS_ivec, int8_t,  BMAS_ivec);


// Logical Operators
one_arg_fn_body(i64not, SIMD_DOUBLE_STRIDE,   int64_t, BMAS_ivec, int64_t, BMAS_ivec);
one_arg_fn_body(i32not, SIMD_SINGLE_STRIDE,   int32_t, BMAS_ivec, int32_t, BMAS_ivec);
one_arg_fn_body(i16not, SIMD_SINGLE_STRIDE*2, int16_t, BMAS_ivec, int16_t, BMAS_ivec);
one_arg_fn_body(i8not,  SIMD_SINGLE_STRIDE*4, int8_t,  BMAS_ivec, int8_t,  BMAS_ivec);

one_arg_fn_body(u64not, SIMD_DOUBLE_STRIDE,   uint64_t, BMAS_ivec, uint64_t, BMAS_ivec);
one_arg_fn_body(u32not, SIMD_SINGLE_STRIDE,   uint32_t, BMAS_ivec, uint32_t, BMAS_ivec);
one_arg_fn_body(u16not, SIMD_SINGLE_STRIDE*2, uint16_t, BMAS_ivec, uint16_t, BMAS_ivec);
one_arg_fn_body(u8not,  SIMD_SINGLE_STRIDE*4, uint8_t,  BMAS_ivec, uint8_t,  BMAS_ivec);


two_arg_fn_body(i64and, SIMD_DOUBLE_STRIDE,   int64_t, BMAS_ivec, int64_t, BMAS_ivec);
two_arg_fn_body(i32and, SIMD_SINGLE_STRIDE,   int32_t, BMAS_ivec, int32_t, BMAS_ivec);
two_arg_fn_body(i16and, SIMD_SINGLE_STRIDE*2, int16_t, BMAS_ivec, int16_t, BMAS_ivec);
two_arg_fn_body(i8and,  SIMD_SINGLE_STRIDE*4, int8_t,  BMAS_ivec, int8_t,  BMAS_ivec);

two_arg_fn_body(u64and, SIMD_DOUBLE_STRIDE,   uint64_t, BMAS_ivec, uint64_t, BMAS_ivec);
two_arg_fn_body(u32and, SIMD_SINGLE_STRIDE,   uint32_t, BMAS_ivec, uint32_t, BMAS_ivec);
two_arg_fn_body(u16and, SIMD_SINGLE_STRIDE*2, uint16_t, BMAS_ivec, uint16_t, BMAS_ivec);
two_arg_fn_body(u8and,  SIMD_SINGLE_STRIDE*4, uint8_t,  BMAS_ivec, uint8_t,  BMAS_ivec);


two_arg_fn_body(i64or, SIMD_DOUBLE_STRIDE,   int64_t, BMAS_ivec, int64_t, BMAS_ivec);
two_arg_fn_body(i32or, SIMD_SINGLE_STRIDE,   int32_t, BMAS_ivec, int32_t, BMAS_ivec);
two_arg_fn_body(i16or, SIMD_SINGLE_STRIDE*2, int16_t, BMAS_ivec, int16_t, BMAS_ivec);
two_arg_fn_body(i8or,  SIMD_SINGLE_STRIDE*4, int8_t,  BMAS_ivec, int8_t,  BMAS_ivec);

two_arg_fn_body(u64or, SIMD_DOUBLE_STRIDE,   uint64_t, BMAS_ivec, uint64_t, BMAS_ivec);
two_arg_fn_body(u32or, SIMD_SINGLE_STRIDE,   uint32_t, BMAS_ivec, uint32_t, BMAS_ivec);
two_arg_fn_body(u16or, SIMD_SINGLE_STRIDE*2, uint16_t, BMAS_ivec, uint16_t, BMAS_ivec);
two_arg_fn_body(u8or,  SIMD_SINGLE_STRIDE*4, uint8_t,  BMAS_ivec, uint8_t,  BMAS_ivec);


two_arg_fn_body(i64xor, SIMD_DOUBLE_STRIDE,   int64_t, BMAS_ivec, int64_t, BMAS_ivec);
two_arg_fn_body(i32xor, SIMD_SINGLE_STRIDE,   int32_t, BMAS_ivec, int32_t, BMAS_ivec);
two_arg_fn_body(i16xor, SIMD_SINGLE_STRIDE*2, int16_t, BMAS_ivec, int16_t, BMAS_ivec);
two_arg_fn_body(i8xor,  SIMD_SINGLE_STRIDE*4, int8_t,  BMAS_ivec, int8_t,  BMAS_ivec);

two_arg_fn_body(u64xor, SIMD_DOUBLE_STRIDE,   uint64_t, BMAS_ivec, uint64_t, BMAS_ivec);
two_arg_fn_body(u32xor, SIMD_SINGLE_STRIDE,   uint32_t, BMAS_ivec, uint32_t, BMAS_ivec);
two_arg_fn_body(u16xor, SIMD_SINGLE_STRIDE*2, uint16_t, BMAS_ivec, uint16_t, BMAS_ivec);
two_arg_fn_body(u8xor,  SIMD_SINGLE_STRIDE*4, uint8_t,  BMAS_ivec, uint8_t,  BMAS_ivec);


two_arg_fn_body(i64andnot, SIMD_DOUBLE_STRIDE,   int64_t, BMAS_ivec, int64_t, BMAS_ivec);
two_arg_fn_body(i32andnot, SIMD_SINGLE_STRIDE,   int32_t, BMAS_ivec, int32_t, BMAS_ivec);
two_arg_fn_body(i16andnot, SIMD_SINGLE_STRIDE*2, int16_t, BMAS_ivec, int16_t, BMAS_ivec);
two_arg_fn_body(i8andnot,  SIMD_SINGLE_STRIDE*4, int8_t,  BMAS_ivec, int8_t,  BMAS_ivec);

two_arg_fn_body(u64andnot, SIMD_DOUBLE_STRIDE,   uint64_t, BMAS_ivec, uint64_t, BMAS_ivec);
two_arg_fn_body(u32andnot, SIMD_SINGLE_STRIDE,   uint32_t, BMAS_ivec, uint32_t, BMAS_ivec);
two_arg_fn_body(u16andnot, SIMD_SINGLE_STRIDE*2, uint16_t, BMAS_ivec, uint16_t, BMAS_ivec);
two_arg_fn_body(u8andnot,  SIMD_SINGLE_STRIDE*4, uint8_t,  BMAS_ivec, uint8_t,  BMAS_ivec);



// Bitshift Operators

two_arg_fn_body(i64sra,   SIMD_DOUBLE_STRIDE,   int64_t,  BMAS_ivec, int64_t,  BMAS_ivec);
two_arg_fn_body(i32sra,   SIMD_SINGLE_STRIDE,   int32_t,  BMAS_ivec, int32_t,  BMAS_ivec);
two_arg_fn_body(i16sra,   SIMD_SINGLE_STRIDE*2, int16_t,  BMAS_ivec, int16_t,  BMAS_ivec);
two_arg_fn_body(i8sra,    SIMD_SINGLE_STRIDE*4, int8_t,   BMAS_ivec, int8_t,   BMAS_ivec);

two_arg_fn_body(u64srl,   SIMD_DOUBLE_STRIDE,   uint64_t, BMAS_ivec, uint64_t, BMAS_ivec);
two_arg_fn_body(u32srl,   SIMD_SINGLE_STRIDE,   uint32_t, BMAS_ivec, uint32_t, BMAS_ivec);
two_arg_fn_body(u16srl,   SIMD_SINGLE_STRIDE*2, uint16_t, BMAS_ivec, uint16_t, BMAS_ivec);
two_arg_fn_body(u8srl,    SIMD_SINGLE_STRIDE*4, uint8_t,  BMAS_ivec, uint8_t,  BMAS_ivec);

two_arg_fn_body(u64sll,   SIMD_DOUBLE_STRIDE,   uint64_t, BMAS_ivec, uint64_t, BMAS_ivec);
two_arg_fn_body(u32sll,   SIMD_SINGLE_STRIDE,   uint32_t, BMAS_ivec, uint32_t, BMAS_ivec);
two_arg_fn_body(u16sll,   SIMD_SINGLE_STRIDE*2, uint16_t, BMAS_ivec, uint16_t, BMAS_ivec);
two_arg_fn_body(u8sll,    SIMD_SINGLE_STRIDE*4, uint8_t,  BMAS_ivec, uint8_t,  BMAS_ivec);

/* // two_arg_fn_body(copysign); */
/* // two_arg_fn_body(fmax); */
/* // two_arg_fn_body(fmin); */
/* // two_arg_fn_body(fdim); */

/* // two_arg_fn_body(hypot); */
