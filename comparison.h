
#define two_arg_fn_body_comparison(name, stride, type, vec, bool_vec)   \
  void BMAS_##name(const long n,                                        \
                   type *x, const long incx,                            \
                   type *y, const long incy,                            \
                   _Bool *out, const long inc_out){                     \
    _Bool *out_end = out + inc_out * n;                                 \
    vec va, vb; bool_vec vc;                                            \
    if (incx == 1 && incy == 1 && inc_out == 1){                        \
      _Bool *simd_end = out + (n/stride)*stride;                        \
      while(out != simd_end){                                           \
        va = vec##_load(x);                                             \
        vb = vec##_load(y);                                             \
        vc = BMAS_vector_##name(va, vb);                                \
        vec##_store_bool(vc, out, 1);                                   \
        x += stride;                                                    \
        y += stride;                                                    \
        out += stride;                                                  \
      }                                                                 \
    }else if(incy == 1 && inc_out == 1){                                \
      _Bool *simd_end = out + (n/stride)*stride;                        \
      while(out != simd_end){                                           \
        va = vec##_make(x, incx, sizeof(type));                         \
        vb = vec##_load(y);                                             \
        vc = BMAS_vector_##name(va, vb);                                \
        vec##_store_bool(vc, out, 1);                                   \
        x += stride*incx;                                               \
        y += stride;                                                    \
        out += stride;                                                  \
      }                                                                 \
    }else if(incx == 1 && inc_out == 1){                                \
      _Bool *simd_end = out + (n/stride)*stride;                        \
      while(out != simd_end){                                           \
        va = vec##_load(x);                                             \
        vb = vec##_make(y, incy, sizeof(type));                         \
        vc = BMAS_vector_##name(va, vb);                                \
        vec##_store_bool(vc, out, 1);                                   \
        x += stride;                                                    \
        y += stride*incy;                                               \
        out += stride;                                                  \
      }                                                                 \
    }else if(incx == 1 && incy == 1){                                   \
      type *simd_end = x + (n/stride)*stride;                           \
      while(x != simd_end){                                             \
        va = vec##_load(x);                                             \
        vb = vec##_load(y);                                             \
        vc = BMAS_vector_##name(va, vb);                                \
        vec##_store_bool(vc, out, inc_out);                             \
        x += stride;                                                    \
        y += stride;                                                    \
        out += stride*inc_out;                                          \
      }                                                                 \
    }else if(inc_out == 1){                                             \
      _Bool *simd_end = out + (n/stride)*stride;                        \
      while(out != simd_end){                                           \
        va = vec##_make(x, incx, sizeof(type));                         \
        vb = vec##_make(y, incy, sizeof(type));                         \
        vc = BMAS_vector_##name(va, vb);                                \
        vec##_store_bool(vc, out, 1);                                   \
        x += stride*incx;                                               \
        y += stride*incy;                                               \
        out += stride;                                                  \
      }                                                                 \
    }else if(incy == 1){                                                \
      type *simd_end = y + (n/stride)*stride;                           \
      while(y != simd_end){                                             \
        va = vec##_make(x, incx, sizeof(type));                         \
        vb = vec##_load(y);                                             \
        vc = BMAS_vector_##name(va, vb);                                \
        vec##_store_bool(vc, out, inc_out);                             \
        x += stride*incx;                                               \
        y += stride;                                                    \
        out += stride*inc_out;                                          \
      }                                                                 \
    }else if(incx == 1){                                                \
      type *simd_end = x + (n/stride)*stride;                           \
      while(x != simd_end){                                             \
        va = vec##_load(x);                                             \
        vb = vec##_make(y, incy, sizeof(type));                         \
        vc = BMAS_vector_##name(va, vb);                                \
        vec##_store_bool(vc, out, inc_out);                             \
        x += stride;                                                    \
        y += stride*incy;                                               \
        out += stride*inc_out;                                          \
      }                                                                 \
    }else{                                                              \
      long i=0;                                                         \
      const long simd_end = (n/stride)*stride;                          \
      while(i != simd_end){                                             \
        va = vec##_make(x, incx, sizeof(type));                         \
        vb = vec##_make(y, incy, sizeof(type));                         \
        vc = BMAS_vector_##name(va, vb);                                \
        vec##_store_bool(vc, out, inc_out);                             \
        i += stride;                                                    \
        x += stride*incx;                                               \
        y += stride*incy;                                               \
        out += stride*inc_out;                                          \
      }                                                                 \
    }                                                                   \
                                                                        \
    while(out!=out_end){                                                \
      out[0] = BMAS_scalar_##name(x[0], y[0]);                          \
      x += incx;                                                        \
      y += incy;                                                        \
      out += inc_out;                                                   \
    }                                                                   \
  };                                                                    
