#define cast_to_double_body(prefix, from, from_vec, cvt_vec)            \
  void BMAS_cast_##prefix##d(const long n,                              \
                             from* x, const long incx,                  \
                             double* y, const long incy){               \
    double* y_end = y + incy * n;                                       \
    from_vec va;                                                        \
    BMAS_dvec vb;                                                       \
    if (incx == 1 && incy == 1){                                        \
      double* simd_end = y + (n/SIMD_DOUBLE_STRIDE)*SIMD_DOUBLE_STRIDE; \
      while(y != simd_end){                                             \
        va = from_vec##_load(x);                                        \
        vb = cvt_vec(va);                                               \
        BMAS_dvec_store(y, vb);                                         \
        x += SIMD_DOUBLE_STRIDE;                                        \
        y += SIMD_DOUBLE_STRIDE;                                        \
      }                                                                 \
    }else if(incy == 1){                                                \
      double* simd_end = y + (n/SIMD_DOUBLE_STRIDE)*SIMD_DOUBLE_STRIDE; \
      while(y != simd_end){                                             \
        va = from_vec##_make(x, incx, sizeof(from));                    \
        vb = cvt_vec(va);                                               \
        BMAS_dvec_store(y, vb);                                         \
        x += SIMD_DOUBLE_STRIDE*incx;                                   \
        y += SIMD_DOUBLE_STRIDE;                                        \
      }                                                                 \
    }else if(incx == 1){                                                \
      from* simd_end = x + (n/SIMD_DOUBLE_STRIDE)*SIMD_DOUBLE_STRIDE;   \
      while(x != simd_end){                                             \
        va = from_vec##_load(x);                                        \
        vb = cvt_vec(va);                                               \
        BMAS_dvec_store_multi(vb, y, incy, sizeof(double));             \
        x += SIMD_DOUBLE_STRIDE;                                        \
        y += SIMD_DOUBLE_STRIDE*incy;                                   \
      }                                                                 \
    }else{                                                              \
      long i=0;                                                         \
      const long simd_end = (n/SIMD_DOUBLE_STRIDE)*SIMD_DOUBLE_STRIDE;  \
      while(i != simd_end){                                             \
        va = from_vec##_make(x, incx, sizeof(from));                    \
        vb = cvt_vec(va);                                               \
        BMAS_dvec_store_multi(vb, y, incy, sizeof(double));             \
        i += SIMD_DOUBLE_STRIDE;                                        \
        x += SIMD_DOUBLE_STRIDE * incx;                                 \
        y += SIMD_DOUBLE_STRIDE * incy;                                 \
      }                                                                 \
    }                                                                   \
                                                                        \
    while(y!=y_end){                                                    \
      y[0] = (double)(x[0]);                                            \
      x += incx;                                                        \
      y += incy;                                                        \
    }                                                                   \
  }

cast_to_double_body(s,   float,   BMAS_svech,  BMAS_svech_to_dvec);
cast_to_double_body(i8,  int8_t,  BMAS_ivech,  BMAS_ivech_to_dvec_i8);
cast_to_double_body(i16, int16_t, BMAS_ivech,  BMAS_ivech_to_dvec_i16);
cast_to_double_body(i32, int32_t, BMAS_ivech,  BMAS_ivech_to_dvec_i32);
cast_to_double_body(i64, int64_t, BMAS_ivec,   BMAS_ivec_to_dvec_i64);
cast_to_double_body(u8,  uint8_t,  BMAS_ivech, BMAS_ivech_to_dvec_u8);
cast_to_double_body(u16, uint16_t, BMAS_ivech, BMAS_ivech_to_dvec_u16);
cast_to_double_body(u32, uint32_t, BMAS_ivech, BMAS_ivech_to_dvec_u32);
cast_to_double_body(u64, uint64_t, BMAS_ivec,  BMAS_ivec_to_dvec_u64);



#define cast_to_float_body(prefix, len, from, from_vec, to_vec, cvt_vec) \
  void BMAS_cast_##prefix##s(const long n,                              \
                             from* x, const long incx,                  \
                             float* y, const long incy){                \
    float* y_end = y + incy * n;                                        \
    from_vec va;                                                        \
    to_vec vb;                                                          \
    if (incx == 1 && incy == 1){                                        \
      float* simd_end = y + (n/len)*len;                                \
      while(y != simd_end){                                             \
        va = from_vec##_load(x);                                        \
        vb = cvt_vec(va);                                               \
        to_vec##_store(y, vb);                                          \
        x += len;                                                       \
        y += len;                                                       \
      }                                                                 \
    }else if(incy == 1){                                                \
      float* simd_end = y + (n/len)*len;                                \
      while(y != simd_end){                                             \
        va = from_vec##_make(x, incx, sizeof(from));                    \
        vb = cvt_vec(va);                                               \
        to_vec##_store(y, vb);                                          \
        x += len*incx;                                                  \
        y += len;                                                       \
      }                                                                 \
    }else if(incx == 1){                                                \
      from* simd_end = x + (n/len)*len;                                 \
      while(x != simd_end){                                             \
        va = from_vec##_load(x);                                        \
        vb = cvt_vec(va);                                               \
        to_vec##_store_multi(vb, y, incy, sizeof(float));               \
        x += len;                                                       \
        y += len*incy;                                                  \
      }                                                                 \
    }else{                                                              \
      long i=0;                                                         \
      const long simd_end = (n/len)*len;                                \
      while(i != simd_end){                                             \
        va = from_vec##_make(x, incx, sizeof(from));                    \
        vb = cvt_vec(va);                                               \
        to_vec##_store_multi(vb, y, incy, sizeof(float));               \
        i += len;                                                       \
        x += len * incx;                                                \
        y += len * incy;                                                \
      }                                                                 \
    }                                                                   \
                                                                        \
    while(y!=y_end){                                                    \
      y[0] = (float)(x[0]);                                             \
      x += incx;                                                        \
      y += incy;                                                        \
    }                                                                   \
  }
