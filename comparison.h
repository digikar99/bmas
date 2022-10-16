
#define two_arg_fn_body_comparisonx1(name, _stride, type, vec, bool_vec) \
  void BMAS_##name(const long n,                                        \
                   type *x, const long incx,                            \
                   type *y, const long incy,                            \
                   _Bool *out, const long inc_out){                     \
    _Bool *out_end = out + inc_out * n;                                 \
    vec va, vb; bool_vec vc;                                            \
    const int stride = _stride;                                         \
    if (incx == 1 && incy == 1 && inc_out == 1){                        \
      _Bool *simd_end = out + (n/stride)*stride;                        \
      while(out != simd_end){                                           \
        va = vec##_load(x);                                             \
        vb = vec##_load(y);                                             \
        vc = BMAS_vector_##name(va, vb);                                \
        vec##_store_boolx1(out, vc);                                    \
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
        vec##_store_boolx1(out, vc);                                    \
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
        vec##_store_boolx1(out, vc);                                    \
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
        vec##_store_bool(out, inc_out, vc, sizeof(type));               \
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
        vec##_store_boolx1(out, vc);                                    \
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
        vec##_store_bool(out, inc_out, vc, sizeof(type));               \
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
        vec##_store_bool(out, inc_out, vc, sizeof(type));               \
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
        vec##_store_bool(out, inc_out, vc, sizeof(type));               \
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


#define two_arg_fn_body_comparisonx2(name, _stride, type, vec, bool_vec) \
  void BMAS_##name(const long n,                                        \
                   type *x, const long incx,                            \
                   type *y, const long incy,                            \
                   _Bool *out, const long inc_out){                     \
    _Bool *out_end = out + inc_out * n;                                 \
    vec va, vb; bool_vec vc;                                            \
    const int stride = _stride;                                         \
    if (incx == 1 && incy == 1 && inc_out == 1){                        \
      const int bool_stride = 2*_stride;                                \
      _Bool *simd_end = out + (n/bool_stride)*bool_stride;              \
      while(out != simd_end){                                           \
        BMAS_ivec vci;                                                  \
        vec va1 = vec##_load(x);                                        \
        vec vb1 = vec##_load(y);                                        \
        vec vc1 = BMAS_vector_##name(va1, vb1);                         \
        vec va2 = vec##_load(x+stride);                                 \
        vec vb2 = vec##_load(y+stride);                                 \
        vec vc2 = BMAS_vector_##name(va2, vb2);                         \
        vec##_store_boolx2(out, vc1, vc2);                              \
        x += bool_stride;                                               \
        y += bool_stride;                                               \
        out += bool_stride;                                             \
      }                                                                 \
    }else if(incy == 1 && inc_out == 1){                                \
      const int bool_stride = 2*_stride;                                \
      _Bool *simd_end = out + (n/bool_stride)*bool_stride;              \
      while(out != simd_end){                                           \
        vec va1 = vec##_make(x, incx, sizeof(type));                    \
        vec vb1 = vec##_load(y);                                        \
        vec vc1 = BMAS_vector_##name(va1, vb1);                         \
        vec va2 = vec##_make(x+stride, incx, sizeof(type));             \
        vec vb2 = vec##_load(y+stride);                                 \
        vec vc2 = BMAS_vector_##name(va2, vb2);                         \
        vec##_store_boolx2(out, vc1, vc2);                              \
        x += bool_stride*incx;                                          \
        y += bool_stride;                                               \
        out += bool_stride;                                             \
      }                                                                 \
    }else if(incx == 1 && inc_out == 1){                                \
      const int bool_stride = 2*_stride;                                \
      _Bool *simd_end = out + (n/bool_stride)*bool_stride;              \
      while(out != simd_end){                                           \
        vec va1 = vec##_load(x);                                        \
        vec vb1 = vec##_make(y, incy, sizeof(type));                    \
        vec vc1 = BMAS_vector_##name(va1, vb1);                         \
        vec va2 = vec##_load(x+stride);                                 \
        vec vb2 = vec##_make(y+stride, incy, sizeof(type));             \
        vec vc2 = BMAS_vector_##name(va2, vb2);                         \
        vec##_store_boolx2(out, vc1, vc2);                              \
        x += bool_stride;                                               \
        y += bool_stride*incy;                                          \
        out += bool_stride;                                             \
      }                                                                 \
    }else if(incx == 1 && incy == 1){                                   \
      type *simd_end = x + (n/stride)*stride;                           \
      while(x != simd_end){                                             \
        va = vec##_load(x);                                             \
        vb = vec##_load(y);                                             \
        vc = BMAS_vector_##name(va, vb);                                \
        vec##_store_bool(out, inc_out, vc, sizeof(type));               \
        x += stride;                                                    \
        y += stride;                                                    \
        out += stride*inc_out;                                          \
      }                                                                 \
    }else if(inc_out == 1){                                             \
      const int bool_stride = 2*_stride;                                \
      _Bool *simd_end = out + (n/bool_stride)*bool_stride;              \
      while(out != simd_end){                                           \
        vec va1 = vec##_make(x, incx, sizeof(type));                    \
        vec vb1 = vec##_make(y, incy, sizeof(type));                    \
        vec vc1 = BMAS_vector_##name(va1, vb1);                         \
        vec va2 = vec##_make(x+stride*incx, incx, sizeof(type));        \
        vec vb2 = vec##_make(y+stride*incy, incy, sizeof(type));        \
        vec vc2 = BMAS_vector_##name(va2, vb2);                         \
        vec##_store_boolx2(out, vc1, vc2);                              \
        x += bool_stride*incx;                                          \
        y += bool_stride*incy;                                          \
        out += bool_stride;                                             \
      }                                                                 \
    }else if(incy == 1){                                                \
      type *simd_end = y + (n/stride)*stride;                           \
      while(y != simd_end){                                             \
        va = vec##_make(x, incx, sizeof(type));                         \
        vb = vec##_load(y);                                             \
        vc = BMAS_vector_##name(va, vb);                                \
        vec##_store_bool(out, inc_out, vc, sizeof(type));               \
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
        vec##_store_bool(out, inc_out, vc, sizeof(type));               \
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
        vec##_store_bool(out, inc_out, vc, sizeof(type));               \
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


#define two_arg_fn_body_comparisonx4(name, _stride, type, vec, bool_vec) \
  void BMAS_##name(const long n,                                        \
                   type *x, const long incx,                            \
                   type *y, const long incy,                            \
                   _Bool *out, const long inc_out){                     \
    _Bool *out_end = out + inc_out * n;                                 \
    vec va, vb; bool_vec vc;                                            \
    const int stride = _stride;                                         \
    if (incx == 1 && incy == 1 && inc_out == 1){                        \
      const int bool_stride = 4*_stride;                                \
      _Bool *simd_end = out + (n/bool_stride)*bool_stride;              \
      while(out != simd_end){                                           \
        BMAS_ivec vci;                                                  \
        vec va1 = vec##_load(x);                                        \
        vec vb1 = vec##_load(y);                                        \
        vec vc1 = BMAS_vector_##name(va1, vb1);                         \
        vec va2 = vec##_load(x+stride);                                 \
        vec vb2 = vec##_load(y+stride);                                 \
        vec vc2 = BMAS_vector_##name(va2, vb2);                         \
        vec va3 = vec##_load(x+2*stride);                               \
        vec vb3 = vec##_load(y+2*stride);                               \
        vec vc3 = BMAS_vector_##name(va3, vb3);                         \
        vec va4 = vec##_load(x+3*stride);                               \
        vec vb4 = vec##_load(y+3*stride);                               \
        vec vc4 = BMAS_vector_##name(va4, vb4);                         \
        vec##_store_boolx4(out, vc1, vc2, vc3, vc4, sizeof(type));      \
        x += bool_stride;                                               \
        y += bool_stride;                                               \
        out += bool_stride;                                             \
      }                                                                 \
    }else if(incy == 1 && inc_out == 1){                                \
      const int bool_stride = 4*_stride;                                \
      _Bool *simd_end = out + (n/bool_stride)*bool_stride;              \
      while(out != simd_end){                                           \
        vec va1 = vec##_make(x, incx, sizeof(type));                    \
        vec vb1 = vec##_load(y);                                        \
        vec vc1 = BMAS_vector_##name(va1, vb1);                         \
        vec va2 = vec##_make(x+stride*incx, incx, sizeof(type));        \
        vec vb2 = vec##_load(y+stride);                                 \
        vec vc2 = BMAS_vector_##name(va2, vb2);                         \
        vec va3 = vec##_make(x+2*stride*incx, incx, sizeof(type));      \
        vec vb3 = vec##_load(y+2*stride);                               \
        vec vc3 = BMAS_vector_##name(va3, vb3);                         \
        vec va4 = vec##_make(x+3*stride*incx, incx, sizeof(type));      \
        vec vb4 = vec##_load(y+3*stride);                               \
        vec vc4 = BMAS_vector_##name(va4, vb4);                         \
        vec##_store_boolx4(out, vc1, vc2, vc3, vc4, sizeof(type));      \
        x += bool_stride*incx;                                          \
        y += bool_stride;                                               \
        out += bool_stride;                                             \
      }                                                                 \
    }else if(incx == 1 && inc_out == 1){                                \
      const int bool_stride = 4*_stride;                                \
      _Bool *simd_end = out + (n/bool_stride)*bool_stride;              \
      while(out != simd_end){                                           \
        vec va1 = vec##_load(x);                                        \
        vec vb1 = vec##_make(y, incy, sizeof(type));                    \
        vec vc1 = BMAS_vector_##name(va1, vb1);                         \
        vec va2 = vec##_load(x+stride);                                 \
        vec vb2 = vec##_make(y+stride*incy, incy, sizeof(type));        \
        vec vc2 = BMAS_vector_##name(va2, vb2);                         \
        vec va3 = vec##_load(x+2*stride);                               \
        vec vb3 = vec##_make(y+2*stride*incy, incy, sizeof(type));      \
        vec vc3 = BMAS_vector_##name(va3, vb3);                         \
        vec va4 = vec##_load(x+3*stride);                               \
        vec vb4 = vec##_make(y+3*stride*incy, incy, sizeof(type));      \
        vec vc4 = BMAS_vector_##name(va4, vb4);                         \
        vec##_store_boolx4(out, vc1, vc2, vc3, vc4, sizeof(type));      \
        x += bool_stride;                                               \
        y += bool_stride*incy;                                          \
        out += bool_stride;                                             \
      }                                                                 \
    }else if(incx == 1 && incy == 1){                                   \
      type *simd_end = x + (n/stride)*stride;                           \
      while(x != simd_end){                                             \
        va = vec##_load(x);                                             \
        vb = vec##_load(y);                                             \
        vc = BMAS_vector_##name(va, vb);                                \
        vec##_store_bool(out, inc_out, vc, sizeof(type));               \
        x += stride;                                                    \
        y += stride;                                                    \
        out += stride*inc_out;                                          \
      }                                                                 \
    }else if(inc_out == 1){                                             \
      const int bool_stride = 4*_stride;                                \
      _Bool *simd_end = out + (n/bool_stride)*bool_stride;              \
      while(out != simd_end){                                           \
        vec va1 = vec##_make(x, incx, sizeof(type));                    \
        vec vb1 = vec##_make(y, incy, sizeof(type));                    \
        vec vc1 = BMAS_vector_##name(va1, vb1);                         \
        vec va2 = vec##_make(x+stride*incx, incx, sizeof(type));        \
        vec vb2 = vec##_make(y+stride*incy, incy, sizeof(type));        \
        vec vc2 = BMAS_vector_##name(va2, vb2);                         \
        vec va3 = vec##_make(x+2*stride*incx, incx, sizeof(type));      \
        vec vb3 = vec##_make(y+2*stride*incy, incy, sizeof(type));      \
        vec vc3 = BMAS_vector_##name(va3, vb3);                         \
        vec va4 = vec##_make(x+3*stride*incx, incx, sizeof(type));      \
        vec vb4 = vec##_make(y+3*stride*incy, incy, sizeof(type));      \
        vec vc4 = BMAS_vector_##name(va4, vb4);                         \
        vec##_store_boolx4(out, vc1, vc2, vc3, vc4, sizeof(type));      \
        x += bool_stride*incx;                                          \
        y += bool_stride*incy;                                          \
        out += bool_stride;                                             \
      }                                                                 \
    }else if(incy == 1){                                                \
      type *simd_end = y + (n/stride)*stride;                           \
      while(y != simd_end){                                             \
        va = vec##_make(x, incx, sizeof(type));                         \
        vb = vec##_load(y);                                             \
        vc = BMAS_vector_##name(va, vb);                                \
        vec##_store_bool(out, inc_out, vc, sizeof(type));               \
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
        vec##_store_bool(out, inc_out, vc, sizeof(type));               \
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
        vec##_store_bool(out, inc_out, vc, sizeof(type));               \
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
