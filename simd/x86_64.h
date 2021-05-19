#include <immintrin.h>

#if defined(__AVX512F__)

  #include "sleef/sleefinline_avx512f.h"
  #include "x86_64/avx512f.h"

#elif defined(__AVX2__)

  #include "sleef/sleefinline_avx2.h"
  #include "x86_64/avx2.h"

#else

  #include "sleef/sleefinline_sse2.h"
  #include "x86_64/sse2.h"

#endif
