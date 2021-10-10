
// The way this is currently implemented, a vector-accumulator, having
// a larger otype wouldn't make a difference; for that, the vector-accumulator and
// store-load instructions themselves would need to be changed.

#define sum_fn_body(name, _stride, itype, vec, otype, init_fn, vsum_fn, hsum_fn) \
  otype BMAS_##name(const long n, itype* x, const long incx){           \
    itype* x_end = x + incx * n;                                        \
    vec v;                                                              \
    vec acc = BMAS_vector_##init_fn();                                  \
    const int stride = _stride;                                         \
    if (incx == 1){                                                     \
      itype* simd_end = x + (n/stride)*stride;                          \
      while(x != simd_end){                                             \
        v = vec##_load(x);                                              \
        acc = BMAS_vector_##vsum_fn(acc, v);                            \
        x += stride;                                                    \
      }                                                                 \
    }else{                                                              \
      long i=0;                                                         \
      const long simd_end = (n/stride)*stride;                          \
      while(i != simd_end){                                             \
        v = vec##_make(x, incx, sizeof(itype));                         \
        acc = BMAS_vector_##vsum_fn(acc, v);                            \
        i += stride;                                                    \
        x += stride * incx;                                             \
      }                                                                 \
    }                                                                   \
    otype result = BMAS_vector_##hsum_fn(acc);                          \
    while(x!=x_end){                                                    \
      result += x[0];                                                   \
      x += incx;                                                        \
    }                                                                   \
    return result;                                                      \
  }
