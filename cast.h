
void BMAS_cast_sd(const long n,
                  float* x, const long incx,
                  double* y, const long incy){
  double* y_end = y + incy * n;
  BMAS_svech va;
  BMAS_dvec vb;
  if (incx == 1 && incy == 1){
    double* simd_end = y + (n/SIMD_DOUBLE_STRIDE)*SIMD_DOUBLE_STRIDE;
    while(y != simd_end){
      va = BMAS_svech_load(x);
      vb = BMAS_svech_to_dvec(va);
      BMAS_dvec_store(y, vb);
      x += SIMD_DOUBLE_STRIDE;
      y += SIMD_DOUBLE_STRIDE;
    }
  }else if(incy == 1){
    double* simd_end = y + (n/SIMD_DOUBLE_STRIDE)*SIMD_DOUBLE_STRIDE;
    while(y != simd_end){
      va = BMAS_svech_make(x, incx, sizeof(float));
      vb = BMAS_svech_to_dvec(va);
      BMAS_dvec_store(y, vb);
      x += SIMD_DOUBLE_STRIDE*incx;
      y += SIMD_DOUBLE_STRIDE;
    }
  }else if(incx == 1){
    float* simd_end = x + (n/SIMD_DOUBLE_STRIDE)*SIMD_DOUBLE_STRIDE;
    while(x != simd_end){
      va = BMAS_svech_load(x);
      vb = BMAS_svech_to_dvec(va);
      BMAS_dvec_store_multi(vb, y, incy, sizeof(double));
      x += SIMD_DOUBLE_STRIDE;
      y += SIMD_DOUBLE_STRIDE*incy;
    }
  }else{
    long i=0;
    const long simd_end = (n/SIMD_DOUBLE_STRIDE)*SIMD_DOUBLE_STRIDE;
    while(i != simd_end){
      va = BMAS_svech_make(x, incx, sizeof(float));
      vb = BMAS_svech_to_dvec(va);
      BMAS_dvec_store_multi(vb, y, incy, sizeof(double));
      i += SIMD_DOUBLE_STRIDE;
      x += SIMD_DOUBLE_STRIDE * incx;
      y += SIMD_DOUBLE_STRIDE * incy;
    }
  }

  while(y!=y_end){
    y[0] = (double)(x[0]);
    x += incx;
    y += incy;
  }
};


void BMAS_cast_ds(const long n,
                  double* x, const long incx,
                  float* y, const long incy){
  float* y_end = y + incy * n;
  BMAS_dvec va;
  BMAS_svech vb;
  if (incx == 1 && incy == 1){
    float* simd_end = y + (n/SIMD_DOUBLE_STRIDE)*SIMD_DOUBLE_STRIDE;
    while(y != simd_end){
      va = BMAS_dvec_load(x);
      vb = BMAS_dvec_to_svech(va);
      BMAS_svech_store(y, vb);
      x += SIMD_DOUBLE_STRIDE;
      y += SIMD_DOUBLE_STRIDE;
    }
  }else if(incy == 1){
    float* simd_end = y + (n/SIMD_DOUBLE_STRIDE)*SIMD_DOUBLE_STRIDE;
    while(y != simd_end){
      va = BMAS_dvec_make(x, incx, sizeof(double));
      vb = BMAS_dvec_to_svech(va);
      BMAS_svech_store(y, vb);
      x += SIMD_DOUBLE_STRIDE*incx;
      y += SIMD_DOUBLE_STRIDE;
    }
  }else if(incx == 1){
    double* simd_end = x + (n/SIMD_DOUBLE_STRIDE)*SIMD_DOUBLE_STRIDE;
    while(x != simd_end){
      va = BMAS_dvec_load(x);
      vb = BMAS_dvec_to_svech(va);
      BMAS_svech_store_multi(vb, y, incy, sizeof(float));
      x += SIMD_DOUBLE_STRIDE;
      y += SIMD_DOUBLE_STRIDE*incy;
    }
  }else{
    long i=0;
    const long simd_end = (n/SIMD_DOUBLE_STRIDE)*SIMD_DOUBLE_STRIDE;
    while(i != simd_end){
      va = BMAS_dvec_make(x, incx, sizeof(double));
      vb = BMAS_dvec_to_svech(va);
      BMAS_svech_store_multi(vb, y, incy, sizeof(float));
      i += SIMD_DOUBLE_STRIDE;
      x += SIMD_DOUBLE_STRIDE * incx;
      y += SIMD_DOUBLE_STRIDE * incy;
    }
  }

  while(y!=y_end){
    y[0] = (float)(x[0]);
    x += incx;
    y += incy;
  }
};
