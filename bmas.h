#include <stdint.h>

#define one_arg_fn(name)                                \
  void BMAS_s##name(const long n,                       \
                    float* x, const long incx,          \
                    float* out, const long inc_out);    \
  void BMAS_d##name(const long n,                       \
                    double* x, const long incx,         \
                    double* out, const long inc_out);

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

/* Example expansion of one_arg_fn(sin):
 * void BMAS_ssin(const long n,
 *               float* x, const long incx,
 *               float* out, const long inc_out);
 * void BMAS_dsin(const long n,
 *               double* x, const long incx,
 *               double* out, const long inc_out);
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


void BMAS_cast_ds(const long n,
                  double* x, const long incx,
                  float* y, const long incy);

void BMAS_cast_i8s(const long n,
                   int8_t* x, const long incx,
                   float* y, const long incy);
void BMAS_cast_i16s(const long n,
                    int8_t* x, const long incx,
                    float* y, const long incy);
void BMAS_cast_i32s(const long n,
                    int8_t* x, const long incx,
                    float* y, const long incy);

one_arg_fn(sin);
one_arg_fn(cos);
one_arg_fn(tan);

one_arg_fn(asin);
one_arg_fn(acos);
one_arg_fn(atan);

one_arg_fn(sinh);
one_arg_fn(cosh);
one_arg_fn(tanh);
one_arg_fn(asinh);
one_arg_fn(acosh);
one_arg_fn(atanh);

one_arg_fn(log);
one_arg_fn(log10);
one_arg_fn(log2);
one_arg_fn(log1p);

one_arg_fn(exp);
one_arg_fn(exp2);
one_arg_fn(exp10);
one_arg_fn(expm1);

two_arg_fn(spow,   float, float);
two_arg_fn(satan2, float, float);
two_arg_fn(sadd,   float, float);
two_arg_fn(ssub,   float, float);
two_arg_fn(smul,   float, float);
two_arg_fn(sdiv,   float, float);

two_arg_fn(dpow,   double, double);
two_arg_fn(datan2, double, double);
two_arg_fn(dadd,   double, double);
two_arg_fn(dsub,   double, double);
two_arg_fn(dmul,   double, double);
two_arg_fn(ddiv,   double, double);

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
two_arg_fn(i8add,  int8_t, int8_t);

two_arg_fn(i64sub, int64_t, int64_t);
two_arg_fn(i32sub, int32_t, int32_t);
two_arg_fn(i16sub, int16_t, int16_t);
two_arg_fn(i8sub,  int8_t, int8_t);

one_arg_fn(cbrt);

one_arg_fn(erf);
one_arg_fn(erfc); // u15
one_arg_fn(tgamma);
one_arg_fn(lgamma); // log gamma

one_arg_fn(sinpi);
one_arg_fn(cospi);
one_arg_fn(sincospi);

one_arg_fn(sqrt);

one_arg_fn(trunc);
one_arg_fn(floor);
one_arg_fn(ceil);
one_arg_fn(round);
one_arg_fn(fabs);
