#define copy_fn_body(prefix, _stride, type, vec)        \
  void BMAS_##prefix##copy(const long n,                \
                           type *x, const long incx,    \
                           type* y, const long incy){   \
    type* y_end = y + incy * n;                         \
    vec v;                                              \
    const int stride = _stride;                         \
    if (incx == 1 && incy == 1){                        \
      type* simd_end = y + (n/stride)*stride;           \
      while(y != simd_end){                             \
        v = vec##_load(x);                              \
        vec##_store(y, v);                              \
        x += stride;                                    \
        y += stride;                                    \
      }                                                 \
    }else if(incy == 1){                                \
      type* simd_end = y + (n/stride)*stride;           \
      while(y != simd_end){                             \
        v = vec##_make(x, incx, sizeof(type));         \
        vec##_store(y, v);                              \
        x += stride*incx;                               \
        y += stride;                                    \
      }                                                 \
    }else if(incx == 1){                                \
      type* simd_end = x + (n/stride)*stride;           \
      while(x != simd_end){                             \
        v = vec##_load(x);                              \
        vec##_store_multi(v, y, incy, sizeof(type));   \
        x += stride;                                    \
        y += stride*incy;                               \
      }                                                 \
    }else{                                              \
      long i=0;                                         \
      const long simd_end = (n/stride)*stride;          \
      while(i != simd_end){                             \
        v = vec##_make(x, incx, sizeof(type));         \
        vec##_store_multi(v, y, incy, sizeof(type));   \
        i += stride;                                    \
        x += stride * incx;                             \
        y += stride * incy;                             \
      }                                                 \
    }                                                   \
                                                        \
    while(y!=y_end){                                    \
      y[0] = x[0];                                      \
      x += incx;                                        \
      y += incy;                                        \
    }                                                   \
  };                                                                    
