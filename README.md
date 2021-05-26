# BMAS - Basic Mathematical Subprograms

This library intends to provide functions for operating on vectors of numbers.
The function signatures are based on BLAS L1 functions, and comprise of:

```
BMAS_{ITYPE}{name}(long n, {type* ptr, long inc_ptr}*)
```

- n - number of elements to operate upon
- ptr, inc_ptr - one or more pairs of pointer to vector, and stride

For example, the function `BMAS_ssin(n, float* in, long inc_in, float* out, long inc_out)` calculates the sin of the single-floating point numbers in the vector defined by (in, inc\_in) and stores the result in the vector defined by (out, inc\_out).

See the [bmas.h](./bmas.h) for the list of currently provided functions.

## Implementation details

The actual computation is done by simd functions from hardware simd instructions, [SLEEF](https://sleef.org/) and/or libmvec*. These are defined inside the [simd](./simd/) directory. 

*Currently, [libmvec](https://github.com/sgallagher/glibc/blob/master/sysdeps/unix/sysv/linux/x86_64/libmvec.abilist) is used for single-float sine and cosine for AVX2, since they were found to be faster than their SLEEF counterparts.

## Disclaimer

I am not primarily a C developer, and thus might not abide by the C conventions. A simple example is I'm using a `make.sh` instead of a `Makefile`. This may change once (re)learning these things becomes a higher priority for me.

