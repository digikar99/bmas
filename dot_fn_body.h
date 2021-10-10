

#define dot_fn_body(name, _stride, itype, vec, otype, init_fn, sum_fn, mul_fn, vec_reduce_fn) \
  otype BMAS_##name(const long n,                                       \
                    itype *x, const long incx,                          \
                    itype *y, const long incy){                         \
    itype *x_end = x + incx * n;                                        \
    vec va, vb, vc;                                                     \
    vec acc = BMAS_vector_##init_fn();                                  \
    const int stride = _stride;                                         \
    if (incx == 1 && incy == 1){                                        \
      itype *simd_end = x + (n/stride)*stride;                          \
      while(x != simd_end){                                             \
        va = vec##_load(x);                                             \
        vb = vec##_load(y);                                             \
        vc = BMAS_vector_##mul_fn(va, vb);                              \
        acc = BMAS_vector_##sum_fn(acc, vc);                            \
        x += stride;                                                    \
        y += stride;                                                    \
      }                                                                 \
    }else if(incy == 1){                                                \
      otype* simd_end = y + (n/stride)*stride;                          \
      while(y != simd_end){                                             \
        va = vec##_make(x, incx, sizeof(itype));                        \
        vb = vec##_load(y);                                             \
        vc = BMAS_vector_##mul_fn(va, vb);                              \
        acc = BMAS_vector_##sum_fn(acc, vc);                            \
        x += stride*incx;                                               \
        y += stride;                                                    \
      }                                                                 \
    }else if(incx == 1){                                                \
      itype* simd_end = x + (n/stride)*stride;                          \
      while(x != simd_end){                                             \
        va = vec##_load(x);                                             \
        vb = vec##_make(y, incy, sizeof(itype));                        \
        vc = BMAS_vector_##mul_fn(va, vb);                              \
        acc = BMAS_vector_##sum_fn(acc, vc);                            \
        x += stride;                                                    \
        y += stride*incy;                                               \
      }                                                                 \
    }else{                                                              \
      long i=0;                                                         \
      const long simd_end = (n/stride)*stride;                          \
      while(i != simd_end){                                             \
        va = vec##_make(x, incx, sizeof(itype));                        \
        vb = vec##_make(y, incy, sizeof(itype));                        \
        vc = BMAS_vector_##mul_fn(va, vb);                              \
        acc = BMAS_vector_##sum_fn(acc, vc);                            \
        i += stride;                                                    \
        x += stride * incx;                                             \
        y += stride * incy;                                             \
      }                                                                 \
    }                                                                   \
                                                                        \
    otype result = BMAS_vector_##vec_reduce_fn(acc);                    \
    while(x!=x_end){                                                    \
      result += x[0] * y[0];                                            \
      x += incx;                                                        \
      y += incy;                                                        \
    }                                                                   \
    return result;                                                      \
  };

