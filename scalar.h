#include <math.h>
#include "sleefinline_purec_scalar.h"

float static inline BMAS_scalar_sadd(float a, float b){return a+b;}
float static inline BMAS_scalar_ssub(float a, float b){return a-b;}
float static inline BMAS_scalar_smul(float a, float b){return a*b;}
float static inline BMAS_scalar_sdiv(float a, float b){return a/b;}

double static inline BMAS_scalar_dadd(double a, double b){return a+b;}
double static inline BMAS_scalar_dsub(double a, double b){return a-b;}
double static inline BMAS_scalar_dmul(double a, double b){return a*b;}
double static inline BMAS_scalar_ddiv(double a, double b){return a/b;}


int64_t static inline BMAS_scalar_i64add(int64_t a, int64_t b){return a+b;}
int32_t static inline BMAS_scalar_i32add(int32_t a, int32_t b){return a+b;}
int16_t static inline BMAS_scalar_i16add(int16_t a, int16_t b){return a+b;}
int8_t  static inline BMAS_scalar_i8add (int8_t a,  int8_t b){return a+b;}

int64_t static inline BMAS_scalar_i64sub(int64_t a, int64_t b){return a-b;}
int32_t static inline BMAS_scalar_i32sub(int32_t a, int32_t b){return a-b;}
int16_t static inline BMAS_scalar_i16sub(int16_t a, int16_t b){return a-b;}
int8_t  static inline BMAS_scalar_i8sub (int8_t a,  int8_t b){return a-b;}

int64_t static inline BMAS_scalar_i64mul(int64_t a, int64_t b){return a*b;}
int32_t static inline BMAS_scalar_i32mul(int32_t a, int32_t b){return a*b;}
int16_t static inline BMAS_scalar_i16mul(int16_t a, int16_t b){return a*b;}
int8_t  static inline BMAS_scalar_i8mul (int8_t a,  int8_t b){return a*b;}

uint64_t static inline BMAS_scalar_u64mul(uint64_t a, uint64_t b){return a*b;}
uint32_t static inline BMAS_scalar_u32mul(uint32_t a, uint32_t b){return a*b;}
uint16_t static inline BMAS_scalar_u16mul(uint16_t a, uint16_t b){return a*b;}
uint8_t  static inline BMAS_scalar_u8mul (uint8_t a,  uint8_t b){return a*b;}

int64_t static inline BMAS_scalar_i64abs(int64_t a){return llabs(a);}
int32_t static inline BMAS_scalar_i32abs(int32_t a){return abs(a);}
int16_t static inline BMAS_scalar_i16abs(int16_t a){return abs(a);}
int8_t  static inline BMAS_scalar_i8abs (int8_t a) {return abs(a);}

float static inline BMAS_scalar_smax(float a, float b){return (a>b)?a:b;}
double static inline BMAS_scalar_dmax(double a, double b){return (a>b)?a:b;}
int64_t static inline BMAS_scalar_i64max(int64_t a, int64_t b){return (a>b)?a:b;}
int32_t static inline BMAS_scalar_i32max(int32_t a, int32_t b){return (a>b)?a:b;}
int16_t static inline BMAS_scalar_i16max(int16_t a, int16_t b){return (a>b)?a:b;}
int8_t  static inline BMAS_scalar_i8max (int8_t a,  int8_t b){return (a>b)?a:b;}
uint64_t static inline BMAS_scalar_u64max(uint64_t a, uint64_t b){return (a>b)?a:b;}
uint32_t static inline BMAS_scalar_u32max(uint32_t a, uint32_t b){return (a>b)?a:b;}
uint16_t static inline BMAS_scalar_u16max(uint16_t a, uint16_t b){return (a>b)?a:b;}
uint8_t  static inline BMAS_scalar_u8max (uint8_t a,  uint8_t b){return (a>b)?a:b;}

float static inline BMAS_scalar_smin(float a, float b){return (a<b)?a:b;}
double static inline BMAS_scalar_dmin(double a, double b){return (a<b)?a:b;}
int64_t static inline BMAS_scalar_i64min(int64_t a, int64_t b){return (a<b)?a:b;}
int32_t static inline BMAS_scalar_i32min(int32_t a, int32_t b){return (a<b)?a:b;}
int16_t static inline BMAS_scalar_i16min(int16_t a, int16_t b){return (a<b)?a:b;}
int8_t  static inline BMAS_scalar_i8min (int8_t a,  int8_t b){return (a<b)?a:b;}
uint64_t static inline BMAS_scalar_u64min(uint64_t a, uint64_t b){return (a<b)?a:b;}
uint32_t static inline BMAS_scalar_u32min(uint32_t a, uint32_t b){return (a<b)?a:b;}
uint16_t static inline BMAS_scalar_u16min(uint16_t a, uint16_t b){return (a<b)?a:b;}
uint8_t  static inline BMAS_scalar_u8min (uint8_t a,  uint8_t b){return (a<b)?a:b;}


int8_t static inline BMAS_scalar_i8and(int8_t a, int8_t b){return a & b;}
int8_t static inline BMAS_scalar_i8or (int8_t a, int8_t b){return a | b;}
int8_t static inline BMAS_scalar_i8not(int8_t a){return ~a;}
int8_t static inline BMAS_scalar_i8xor(int8_t a, int8_t b){return a ^ b;}
int8_t static inline BMAS_scalar_i8andnot(int8_t a, int8_t b){return a & ~b;}

// gcc uses arithmetic shift on signed values and logical shift on unsigned values
int64_t static inline BMAS_scalar_i64sra(int64_t a, int64_t count){return a >> count;}
int32_t static inline BMAS_scalar_i32sra(int32_t a, int32_t count){return a >> count;}
int16_t static inline BMAS_scalar_i16sra(int16_t a, int16_t count){return a >> count;}
int8_t  static inline BMAS_scalar_i8sra(int8_t a, int8_t count){return a >> count;}

uint64_t static inline BMAS_scalar_u64srl(uint64_t a, uint64_t count){return a >> count;}
uint32_t static inline BMAS_scalar_u32srl(uint32_t a, uint32_t count){return a >> count;}
uint16_t static inline BMAS_scalar_u16srl(uint16_t a, uint16_t count){return a >> count;}
uint8_t  static inline BMAS_scalar_u8srl(uint8_t a, uint8_t count){return a >> count;}

uint64_t static inline BMAS_scalar_u64sll(uint64_t a, uint64_t count){return a << count;}
uint32_t static inline BMAS_scalar_u32sll(uint32_t a, uint32_t count){return a << count;}
uint16_t static inline BMAS_scalar_u16sll(uint16_t a, uint16_t count){return a << count;}
uint8_t  static inline BMAS_scalar_u8sll(uint8_t a, uint8_t count){return a << count;}


_Bool static inline BMAS_scalar_slt(float a, float b){return a<b;}
_Bool static inline BMAS_scalar_sle(float a, float b){return a<=b;}
_Bool static inline BMAS_scalar_seq(float a, float b){return a==b;}
_Bool static inline BMAS_scalar_sneq(float a, float b){return a!=b;}
_Bool static inline BMAS_scalar_sge(float a, float b){return a>=b;}
_Bool static inline BMAS_scalar_sgt(float a, float b){return a>b;}

_Bool static inline BMAS_scalar_dlt(double a, double b){return a<b;}
_Bool static inline BMAS_scalar_dle(double a, double b){return a<=b;}
_Bool static inline BMAS_scalar_deq(double a, double b){return a==b;}
_Bool static inline BMAS_scalar_dneq(double a, double b){return a!=b;}
_Bool static inline BMAS_scalar_dge(double a, double b){return a>=b;}
_Bool static inline BMAS_scalar_dgt(double a, double b){return a>b;}

_Bool static inline BMAS_scalar_i8lt(int8_t a, int8_t b){return a<b;}
_Bool static inline BMAS_scalar_i8le(int8_t a, int8_t b){return a<=b;}
_Bool static inline BMAS_scalar_i8eq(int8_t a, int8_t b){return a==b;}
_Bool static inline BMAS_scalar_i8neq(int8_t a, int8_t b){return a!=b;}
_Bool static inline BMAS_scalar_i8ge(int8_t a, int8_t b){return a>=b;}
_Bool static inline BMAS_scalar_i8gt(int8_t a, int8_t b){return a>b;}

_Bool static inline BMAS_scalar_i16lt(int16_t a, int16_t b){return a<b;}
_Bool static inline BMAS_scalar_i16le(int16_t a, int16_t b){return a<=b;}
_Bool static inline BMAS_scalar_i16eq(int16_t a, int16_t b){return a==b;}
_Bool static inline BMAS_scalar_i16neq(int16_t a, int16_t b){return a!=b;}
_Bool static inline BMAS_scalar_i16ge(int16_t a, int16_t b){return a>=b;}
_Bool static inline BMAS_scalar_i16gt(int16_t a, int16_t b){return a>b;}

_Bool static inline BMAS_scalar_i32lt(int32_t a, int32_t b){return a<b;}
_Bool static inline BMAS_scalar_i32le(int32_t a, int32_t b){return a<=b;}
_Bool static inline BMAS_scalar_i32eq(int32_t a, int32_t b){return a==b;}
_Bool static inline BMAS_scalar_i32neq(int32_t a, int32_t b){return a!=b;}
_Bool static inline BMAS_scalar_i32ge(int32_t a, int32_t b){return a>=b;}
_Bool static inline BMAS_scalar_i32gt(int32_t a, int32_t b){return a>b;}

_Bool static inline BMAS_scalar_i64lt(int64_t a, int64_t b){return a<b;}
_Bool static inline BMAS_scalar_i64le(int64_t a, int64_t b){return a<=b;}
_Bool static inline BMAS_scalar_i64eq(int64_t a, int64_t b){return a==b;}
_Bool static inline BMAS_scalar_i64neq(int64_t a, int64_t b){return a!=b;}
_Bool static inline BMAS_scalar_i64ge(int64_t a, int64_t b){return a>=b;}
_Bool static inline BMAS_scalar_i64gt(int64_t a, int64_t b){return a>b;}


_Bool static inline BMAS_scalar_u8lt(uint8_t a, uint8_t b){return a<b;}
_Bool static inline BMAS_scalar_u8le(uint8_t a, uint8_t b){return a<=b;}
_Bool static inline BMAS_scalar_u8eq(uint8_t a, uint8_t b){return a==b;}
_Bool static inline BMAS_scalar_u8neq(uint8_t a, uint8_t b){return a!=b;}
_Bool static inline BMAS_scalar_u8ge(uint8_t a, uint8_t b){return a>=b;}
_Bool static inline BMAS_scalar_u8gt(uint8_t a, uint8_t b){return a>b;}

_Bool static inline BMAS_scalar_u16lt(uint16_t a, uint16_t b){return a<b;}
_Bool static inline BMAS_scalar_u16le(uint16_t a, uint16_t b){return a<=b;}
_Bool static inline BMAS_scalar_u16eq(uint16_t a, uint16_t b){return a==b;}
_Bool static inline BMAS_scalar_u16neq(uint16_t a, uint16_t b){return a!=b;}
_Bool static inline BMAS_scalar_u16ge(uint16_t a, uint16_t b){return a>=b;}
_Bool static inline BMAS_scalar_u16gt(uint16_t a, uint16_t b){return a>b;}

_Bool static inline BMAS_scalar_u32lt(uint32_t a, uint32_t b){return a<b;}
_Bool static inline BMAS_scalar_u32le(uint32_t a, uint32_t b){return a<=b;}
_Bool static inline BMAS_scalar_u32eq(uint32_t a, uint32_t b){return a==b;}
_Bool static inline BMAS_scalar_u32neq(uint32_t a, uint32_t b){return a!=b;}
_Bool static inline BMAS_scalar_u32ge(uint32_t a, uint32_t b){return a>=b;}
_Bool static inline BMAS_scalar_u32gt(uint32_t a, uint32_t b){return a>b;}

_Bool static inline BMAS_scalar_u64lt(uint64_t a, uint64_t b){return a<b;}
_Bool static inline BMAS_scalar_u64le(uint64_t a, uint64_t b){return a<=b;}
_Bool static inline BMAS_scalar_u64eq(uint64_t a, uint64_t b){return a==b;}
_Bool static inline BMAS_scalar_u64neq(uint64_t a, uint64_t b){return a!=b;}
_Bool static inline BMAS_scalar_u64ge(uint64_t a, uint64_t b){return a>=b;}
_Bool static inline BMAS_scalar_u64gt(uint64_t a, uint64_t b){return a>b;}



// trigonometric

float static inline BMAS_scalar_ssin(float a){return sinf(a);}
float static inline BMAS_scalar_scos(float a){return cosf(a);}
float static inline BMAS_scalar_stan(float a){return tanf(a);}

float static inline BMAS_scalar_sasin(float a){return asinf(a);}
float static inline BMAS_scalar_sacos(float a){return acosf(a);}
float static inline BMAS_scalar_satan(float a){return atanf(a);}

float static inline BMAS_scalar_ssinh(float a){return sinhf(a);}
float static inline BMAS_scalar_scosh(float a){return coshf(a);}
float static inline BMAS_scalar_stanh(float a){return tanhf(a);}

float static inline BMAS_scalar_sasinh(float a){return asinhf(a);}
float static inline BMAS_scalar_sacosh(float a){return acoshf(a);}
float static inline BMAS_scalar_satanh(float a){return atanhf(a);}


double static inline BMAS_scalar_dsin(double a){return sin(a);}
double static inline BMAS_scalar_dcos(double a){return cos(a);}
double static inline BMAS_scalar_dtan(double a){return tan(a);}

double static inline BMAS_scalar_dasin(double a){return asin(a);}
double static inline BMAS_scalar_dacos(double a){return acos(a);}
double static inline BMAS_scalar_datan(double a){return atan(a);}

double static inline BMAS_scalar_dsinh(double a){return sinh(a);}
double static inline BMAS_scalar_dcosh(double a){return cosh(a);}
double static inline BMAS_scalar_dtanh(double a){return tanh(a);}

double static inline BMAS_scalar_dasinh(double a){return asinh(a);}
double static inline BMAS_scalar_dacosh(double a){return acosh(a);}
double static inline BMAS_scalar_datanh(double a){return atanh(a);}

// log

float static inline BMAS_scalar_slog(float a)  {return logf(a);}
float static inline BMAS_scalar_slog10(float a){return log10f(a);}
float static inline BMAS_scalar_slog2(float a) {return log2f(a);}
float static inline BMAS_scalar_slog1p(float a){return log1pf(a);}

double static inline BMAS_scalar_dlog(double a)  {return log(a);}
double static inline BMAS_scalar_dlog10(double a){return log10(a);}
double static inline BMAS_scalar_dlog2(double a) {return log2(a);}
double static inline BMAS_scalar_dlog1p(double a){return log1p(a);}

// exp

float static inline BMAS_scalar_sexp(float a)  {return expf(a);}
float static inline BMAS_scalar_sexp10(float a){return Sleef_exp10f1_u10purec(a);}
float static inline BMAS_scalar_sexp2(float a) {return exp2f(a);}
float static inline BMAS_scalar_sexpm1(float a){return expm1f(a);}

double static inline BMAS_scalar_dexp(double a)  {return exp(a);}
double static inline BMAS_scalar_dexp10(double a){return Sleef_exp10d1_u10purec(a);}
double static inline BMAS_scalar_dexp2(double a) {return exp2(a);}
double static inline BMAS_scalar_dexpm1(double a){return expm1(a);}

// pow and atan2

float static inline BMAS_scalar_spow(float x, float y){return powf(x, y);}
float static inline BMAS_scalar_satan2(float x, float y){return atan2f(x, y);}

double static inline BMAS_scalar_dpow(double x, double y){return pow(x, y);}
double static inline BMAS_scalar_datan2(double x, double y){return atan2(x, y);}

// misc

float static inline BMAS_scalar_sfabs(float x)  { return fabsf(x);}
float static inline BMAS_scalar_sceil(float x)  { return ceilf(x);}
float static inline BMAS_scalar_strunc(float x) { return truncf(x);}
float static inline BMAS_scalar_sfloor(float x) { return floorf(x);}
float static inline BMAS_scalar_sround(float x) { return roundf(x);}

double static inline BMAS_scalar_dfabs(double x)  { return fabs(x);}
double static inline BMAS_scalar_dceil(double x)  { return ceil(x);}
double static inline BMAS_scalar_dtrunc(double x) { return trunc(x);}
double static inline BMAS_scalar_dfloor(double x) { return floor(x);}
double static inline BMAS_scalar_dround(double x) { return round(x);}

float  static inline BMAS_scalar_ssqrt(float x) { return sqrtf(x); }
double static inline BMAS_scalar_dsqrt(double x){ return sqrt(x); }

