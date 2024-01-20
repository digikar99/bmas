
// The way this is currently implemented, a vector-accumulator, having
// a larger otype wouldn't make a difference; for that, the vector-accumulator and
// store-load instructions themselves would need to be changed.

// An acc (accumulator) below comprises of struct made of a "value" vector
// and one or more "index" vector(s)

#define one_arg_ireduce_fn_body(                                        \
  name, _stride, itype, vec,                                            \
  init_index_fn, init_fn,                                               \
  vreduce_fn, hreduce_fn, reduce_fn_char, sreduce_fn)                   \
                                                                        \
  long BMAS_##name(const long n, itype* x, const long incx){            \
    itype* x_end = x + incx * n;                                        \
    BMAS_##vec v;                                                       \
    struct BMAS_ipair_##vec acc                                         \
      = BMAS_vector_##init_index_fn(BMAS_vector_##init_fn());           \
    const int stride = _stride;                                         \
    long idx = 0;                                                       \
    if (incx == 1){                                                     \
      itype* simd_end = x + (n/stride)*stride;                          \
      while(x != simd_end){                                             \
        v = BMAS_##vec##_load(x);                                       \
        acc = BMAS_vector_##vreduce_fn(acc, v, idx, reduce_fn_char);    \
        x += stride;                                                    \
        idx += stride;                                                  \
      }                                                                 \
    }else{                                                              \
      long i=0;                                                         \
      const long simd_end = (n/stride)*stride;                          \
      while(i != simd_end){                                             \
        v = BMAS_##vec##_make(x, incx, sizeof(itype));                  \
        acc = BMAS_vector_##vreduce_fn(acc, v, idx, reduce_fn_char);    \
        i += stride;                                                    \
        x += stride * incx;                                             \
        idx += stride;                                                  \
      }                                                                 \
    }                                                                   \
    struct BMAS_ipair_##itype result                                    \
      = BMAS_vector_##hreduce_fn(acc, reduce_fn_char);                  \
    while(x!=x_end){                                                    \
      if (BMAS_scalar_##sreduce_fn(x[0], result.value)){                \
        result.idx = idx;                                               \
        result.value = x[0];                                            \
      }                                                                 \
      x += incx;                                                        \
      idx += 1;                                                         \
    }                                                                   \
    return result.idx;                                                  \
  }
