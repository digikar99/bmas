#include <stdint.h>

#define one_arg_fn_int(name)                        \
  void BMAS_s##name(const long n,                   \
                    float* x, const long incx,      \
                    long* out, const long inc_out); \
  void BMAS_d##name(const long n,                   \
                    double* x, const long incx,     \
                    long* out, const long inc_out);


#define two_arg_fn(name, itype, otype)              \
  void BMAS_##name(const long n,                    \
                   itype* x, const long incx,       \
                   itype* y, const long incy,       \
                   otype* out, const long inc_out);

/* Example expansion of two_arg_fn(sadd, float, float):
 * void BMAS_sadd(const long n,
 *                float* x, const long incx,
 *                float* y, const long incy,
 *                float* out, const long inc_out);
 */

#define copy_fn(prefix, type) \
  void BMAS_##prefix##copy(const long n,\
                           type* x, const long incx,\
                           type* y, const long incy);

copy_fn(s, float);
copy_fn(d, double);
// The only difference is perhaps how 'n' affects how much to copy_fn
copy_fn(i8,  int8_t);
copy_fn(i16, int16_t);
copy_fn(i32, int32_t);
copy_fn(i64, int64_t);

#define cast_to_single(prefix, itype)                        \
  void BMAS_cast_##prefix##s(const long n,                   \
                             itype* x, const long incx,      \
                             float* y, const long incy);

#define cast_to_double(prefix, itype)                       \
  void BMAS_cast_##prefix##d(const long n,                  \
                            itype* x, const long incx,      \
                            double* y, const long incy);

cast_to_single(d,   double);
cast_to_single(i8,  int8_t);
cast_to_single(i16, int16_t);
cast_to_single(i32, int32_t);
cast_to_single(i64, int64_t);
cast_to_single(u8,  uint8_t);
cast_to_single(u16, uint16_t);
cast_to_single(u32, uint32_t);
cast_to_single(u64, uint64_t);

cast_to_double(s,   float);
cast_to_double(i8,  int8_t);
cast_to_double(i16, int16_t);
cast_to_double(i32, int32_t);
cast_to_double(i64, int64_t);
cast_to_double(u8,  uint8_t);
cast_to_double(u16, uint16_t);
cast_to_double(u32, uint32_t);
cast_to_double(u64, uint64_t);


#define one_arg_fn(name, itype, otype)                  \
  void BMAS_##name(const long n,                        \
                   itype* x, const int64_t incx,        \
                   otype* out, const int64_t inc_out);

/* Example expansion of one_arg_fn(ssin, float, float):
 * void BMAS_ssin(const long n,
 *               float* x, const long incx,
 *               float* out, const long inc_out);
 */

one_arg_fn(i8abs,  int8_t,  int8_t);
one_arg_fn(i16abs, int16_t, int16_t);
one_arg_fn(i32abs, int32_t, int32_t);
one_arg_fn(i64abs, int64_t, int64_t);

one_arg_fn(ssin, float, float);
one_arg_fn(scos, float, float);
one_arg_fn(stan, float, float);

one_arg_fn(sasin,  float, float);
one_arg_fn(sacos,  float, float);
one_arg_fn(satan,  float, float);

one_arg_fn(ssinh,  float, float);
one_arg_fn(scosh,  float, float);
one_arg_fn(stanh,  float, float);
one_arg_fn(sasinh, float, float);
one_arg_fn(sacosh, float, float);
one_arg_fn(satanh, float, float);

one_arg_fn(dsin,  double, double);
one_arg_fn(dcos,  double, double);
one_arg_fn(dtan,  double, double);

one_arg_fn(dasin,  double, double);
one_arg_fn(dacos,  double, double);
one_arg_fn(datan,  double, double);

one_arg_fn(dsinh,  double, double);
one_arg_fn(dcosh,  double, double);
one_arg_fn(dtanh,  double, double);
one_arg_fn(dasinh, double, double);
one_arg_fn(dacosh, double, double);
one_arg_fn(datanh, double, double);


one_arg_fn(slog,   float, float);
one_arg_fn(slog10, float, float);
one_arg_fn(slog2,  float, float);
one_arg_fn(slog1p, float, float);

one_arg_fn(sexp,   float, float);
one_arg_fn(sexp2,  float, float);
one_arg_fn(sexp10, float, float);
one_arg_fn(sexpm1, float, float);

one_arg_fn(dlog,   double, double);
one_arg_fn(dlog10, double, double);
one_arg_fn(dlog2,  double, double);
one_arg_fn(dlog1p, double, double);

one_arg_fn(dexp,   double, double);
one_arg_fn(dexp2,  double, double);
one_arg_fn(dexp10, double, double);
one_arg_fn(dexpm1, double, double);

one_arg_fn(ssqrt,  float, float);
one_arg_fn(dsqrt,  double, double);


two_arg_fn(spow,   float, float);
two_arg_fn(satan2, float, float);
two_arg_fn(sadd,   float, float);
two_arg_fn(ssub,   float, float);
two_arg_fn(smul,   float, float);
two_arg_fn(sdiv,   float, float);
two_arg_fn(smin,   float, float);
two_arg_fn(smax,   float, float);

two_arg_fn(dpow,   double, double);
two_arg_fn(datan2, double, double);
two_arg_fn(dadd,   double, double);
two_arg_fn(dsub,   double, double);
two_arg_fn(dmul,   double, double);
two_arg_fn(ddiv,   double, double);
two_arg_fn(dmin,   double, double);
two_arg_fn(dmax,   double, double);

two_arg_fn(slt, float, _Bool);
two_arg_fn(sle, float, _Bool);
two_arg_fn(seq, float, _Bool);
two_arg_fn(sneq, float, _Bool);
two_arg_fn(sgt, float, _Bool);
two_arg_fn(sge, float, _Bool);

two_arg_fn(dlt, double, _Bool);
two_arg_fn(dle, double, _Bool);
two_arg_fn(deq, double, _Bool);
two_arg_fn(dneq, double, _Bool);
two_arg_fn(dgt, double, _Bool);
two_arg_fn(dge, double, _Bool);



two_arg_fn(i64add, int64_t, int64_t);
two_arg_fn(i32add, int32_t, int32_t);
two_arg_fn(i16add, int16_t, int16_t);
two_arg_fn(i8add,  int8_t,  int8_t);

two_arg_fn(i64sub, int64_t, int64_t);
two_arg_fn(i32sub, int32_t, int32_t);
two_arg_fn(i16sub, int16_t, int16_t);
two_arg_fn(i8sub,  int8_t,  int8_t);

two_arg_fn(i64mul, int64_t, int64_t);
two_arg_fn(i32mul, int32_t, int32_t);
two_arg_fn(i16mul, int16_t, int16_t);
two_arg_fn(i8mul,  int8_t,  int8_t);

two_arg_fn(u64mul, uint64_t, uint64_t);
two_arg_fn(u32mul, uint32_t, uint32_t);
two_arg_fn(u16mul, uint16_t, uint16_t);
two_arg_fn(u8mul,  uint8_t,  uint8_t);

two_arg_fn(i64min, int64_t, int64_t);
two_arg_fn(i32min, int32_t, int32_t);
two_arg_fn(i16min, int16_t, int16_t);
two_arg_fn(i8min,  int8_t,  int8_t);

two_arg_fn(u64min, uint64_t, uint64_t);
two_arg_fn(u32min, uint32_t, uint32_t);
two_arg_fn(u16min, uint16_t, uint16_t);
two_arg_fn(u8min,  uint8_t,  uint8_t);

two_arg_fn(i64max, int64_t, int64_t);
two_arg_fn(i32max, int32_t, int32_t);
two_arg_fn(i16max, int16_t, int16_t);
two_arg_fn(i8max,  int8_t,  int8_t);

two_arg_fn(u64max, uint64_t, uint64_t);
two_arg_fn(u32max, uint32_t, uint32_t);
two_arg_fn(u16max, uint16_t, uint16_t);
two_arg_fn(u8max,  uint8_t,  uint8_t);


two_arg_fn(i64lt, int64_t, int64_t);
two_arg_fn(i32lt, int32_t, int32_t);
two_arg_fn(i16lt, int16_t, int16_t);
two_arg_fn(i8lt,  int8_t, int8_t);

two_arg_fn(i64le, int64_t, int64_t);
two_arg_fn(i32le, int32_t, int32_t);
two_arg_fn(i16le, int16_t, int16_t);
two_arg_fn(i8le,  int8_t, int8_t);

two_arg_fn(i64eq, int64_t, int64_t);
two_arg_fn(i32eq, int32_t, int32_t);
two_arg_fn(i16eq, int16_t, int16_t);
two_arg_fn(i8eq,  int8_t, int8_t);

two_arg_fn(i64neq, int64_t, int64_t);
two_arg_fn(i32neq, int32_t, int32_t);
two_arg_fn(i16neq, int16_t, int16_t);
two_arg_fn(i8neq,  int8_t, int8_t);

two_arg_fn(i64gt, int64_t, int64_t);
two_arg_fn(i32gt, int32_t, int32_t);
two_arg_fn(i16gt, int16_t, int16_t);
two_arg_fn(i8gt,  int8_t,  int8_t);

two_arg_fn(i64ge, int64_t, int64_t);
two_arg_fn(i32ge, int32_t, int32_t);
two_arg_fn(i16ge, int16_t, int16_t);
two_arg_fn(i8ge,  int8_t,  int8_t);


two_arg_fn(u64lt, int64_t, int64_t);
two_arg_fn(u32lt, int32_t, int32_t);
two_arg_fn(u16lt, int16_t, int16_t);
two_arg_fn(u8lt,  int8_t,  int8_t);

two_arg_fn(u64le, int64_t, int64_t);
two_arg_fn(u32le, int32_t, int32_t);
two_arg_fn(u16le, int16_t, int16_t);
two_arg_fn(u8le,  int8_t,  int8_t);

two_arg_fn(u64eq, int64_t, int64_t);
two_arg_fn(u32eq, int32_t, int32_t);
two_arg_fn(u16eq, int16_t, int16_t);
two_arg_fn(u8eq,  int8_t,  int8_t);

two_arg_fn(u64neq, int64_t, int64_t);
two_arg_fn(u32neq, int32_t, int32_t);
two_arg_fn(u16neq, int16_t, int16_t);
two_arg_fn(u8neq,  int8_t,  int8_t);

two_arg_fn(u64gt, int64_t, int64_t);
two_arg_fn(u32gt, int32_t, int32_t);
two_arg_fn(u16gt, int16_t, int16_t);
two_arg_fn(u8gt,  int8_t,  int8_t);

two_arg_fn(u64ge, int64_t, int64_t);
two_arg_fn(u32ge, int32_t, int32_t);
two_arg_fn(u16ge, int16_t, int16_t);
two_arg_fn(u8ge,  int8_t,  int8_t);

two_arg_fn(i64sra, int64_t, int64_t);
two_arg_fn(i32sra, int32_t, int32_t);
two_arg_fn(i16sra, int16_t, int16_t);
two_arg_fn(i8sra,  int8_t,  int8_t);

two_arg_fn(u64srl, uint64_t, uint64_t);
two_arg_fn(u32srl, uint32_t, uint32_t);
two_arg_fn(u16srl, uint16_t, uint16_t);
two_arg_fn(u8srl,  uint8_t,  uint8_t);

two_arg_fn(u64sll, uint64_t, uint64_t);
two_arg_fn(u32sll, uint32_t, uint32_t);
two_arg_fn(u16sll, uint16_t, uint16_t);
two_arg_fn(u8sll,  uint8_t,  uint8_t);



void BMAS_i8not(const long N, int8_t* x, const long inc_x, int8_t* out, const long inc_out);
two_arg_fn(i8and, int8_t, int8_t);
two_arg_fn(i8or,  int8_t, int8_t);
two_arg_fn(i8xor, int8_t, int8_t);
two_arg_fn(i8andnot, int8_t, int8_t);

one_arg_fn(cbrt);

one_arg_fn(erf);
one_arg_fn(erfc); // u15
one_arg_fn(tgamma);
one_arg_fn(lgamma); // log gamma

one_arg_fn(sinpi);
one_arg_fn(cospi);
one_arg_fn(sincospi);

one_arg_fn(sqrt);

one_arg_fn(strunc, float, float);
one_arg_fn(sfloor, float, float);
one_arg_fn(sceil,  float, float);
one_arg_fn(sround, float, float);
one_arg_fn(sfabs,  float, float);

one_arg_fn(dtrunc, double, double);
one_arg_fn(dfloor, double, double);
one_arg_fn(dceil,  double, double);
one_arg_fn(dround, double, double);
one_arg_fn(dfabs,  double, double);


#define one_arg_reduce_fn(name, itype, otype) otype BMAS_##name(const long n, itype* x, const int64_t incx);
one_arg_reduce_fn(ssum, float, float);
one_arg_reduce_fn(dsum, double, double);
// Perhaps these should be larger than int8_t and int32_t?
one_arg_reduce_fn(i8sum,  int8_t,  int8_t);
one_arg_reduce_fn(i16sum, int16_t, int16_t);
one_arg_reduce_fn(i32sum, int32_t, int32_t);
one_arg_reduce_fn(i64sum, int64_t, int64_t);

one_arg_reduce_fn(shmax, float, float);
one_arg_reduce_fn(dhmax, double, double);
one_arg_reduce_fn(i8hmax,  int8_t,  int8_t);
one_arg_reduce_fn(i16hmax, int16_t, int16_t);
one_arg_reduce_fn(i32hmax, int32_t, int32_t);
one_arg_reduce_fn(i64hmax, int64_t, int64_t);
one_arg_reduce_fn(u8hmax,  uint8_t,  uint8_t);
one_arg_reduce_fn(u16hmax, uint16_t, uint16_t);
one_arg_reduce_fn(u32hmax, uint32_t, uint32_t);
one_arg_reduce_fn(u64hmax, uint64_t, uint64_t);

one_arg_reduce_fn(shmin, float, float);
one_arg_reduce_fn(dhmin, double, double);
one_arg_reduce_fn(i8hmin,  int8_t,  int8_t);
one_arg_reduce_fn(i16hmin, int16_t, int16_t);
one_arg_reduce_fn(i32hmin, int32_t, int32_t);
one_arg_reduce_fn(i64hmin, int64_t, int64_t);
one_arg_reduce_fn(u8hmin,  uint8_t,  uint8_t);
one_arg_reduce_fn(u16hmin, uint16_t, uint16_t);
one_arg_reduce_fn(u32hmin, uint32_t, uint32_t);
one_arg_reduce_fn(u64hmin, uint64_t, uint64_t);


#define dot_fn(name, itype, otype) \
  otype BMAS_##name(const long n,\
                    itype* x, const int64_t incx,\
                    itype* y, const int64_t incy);
dot_fn(sdot, float, float);
dot_fn(ddot, double, double);
// Perhaps these should be larger than int8_t and int32_t?
dot_fn(i8dot,  int8_t,  int8_t);
dot_fn(i16dot, int16_t, int16_t);
dot_fn(i32dot, int32_t, int32_t);
dot_fn(i64dot, int64_t, int64_t);
/* dot_fn(u8dot,  uint8_t,  uint8_t); */
/* dot_fn(u16dot, uint16_t, uint16_t); */
/* dot_fn(u32dot, uint32_t, uint32_t); */
/* dot_fn(u64dot, uint64_t, uint64_t); */
