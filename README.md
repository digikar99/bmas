# BMAS - Basic Mathematical Subprograms

> Run `bash make.sh` to build; modify for non-gcc / non-avx2 platforms.

This library intends to provide functions for operating on vectors of numbers.
The function signatures are based on BLAS L1 functions, and comprise of:

```
BMAS_{ITYPE}{name}(long n, {type* ptr, long inc_ptr}*)
```

- n - number of elements to operate upon
- ptr, inc_ptr - one or more pairs of pointer to vector, and stride

For example, the function `BMAS_ssin(n, float* in, long inc_in, float* out, long inc_out)` calculates the sin of the single-floating point numbers in the vector defined by (in, inc\_in) and stores the result in the vector defined by (out, inc\_out).

See the [bmas.h](./bmas.h) for the list of currently provided functions.

Correctness is checked by the tests in [numericals](https://github.com/digikar99/numericals).

## Implementation details

The actual computation is done by simd functions from hardware simd instructions, [SLEEF](https://sleef.org/) and/or libmvec*. These are defined inside the [simd](./simd/) directory.

*Currently, [libmvec](https://github.com/sgallagher/glibc/blob/master/sysdeps/unix/sysv/linux/x86_64/libmvec.abilist) is used for single-float sine and cosine for AVX2, since they were found to be faster than their SLEEF counterparts.

## AVX2 Status

SSE and AVX512 support exists to a limited extent due to limited developer time.

### Copy and Conversions

| From \ To | float32 | float64 | int64 | int32 | int16 | int8 | uint64 | uint32 | uint16 | uint8 |
|-----------|---------|---------|-------|-------|-------|------|--------|--------|--------|-------|
| float32   | +       | +       | -     | -     | -     | -    | -      | -      | -      | -     |
| float64   | +       | +       | -     | -     | -     | -    | -      | -      | -      | -     |
| int64     | +       | +       | +     | -     | -     | -    | -      | -      | -      | -     |
| int32     | +       | +       | -     | +     | -     | -    | -      | -      | -      | -     |
| int16     | +       | +       | -     | -     | +     | -    | -      | -      | -      | -     |
| int8      | +       | +       | -     | -     | -     | +    | -      | -      | -      | -     |
| uint64    | +       | +       | -     | -     | -     | -    | +      | -      | -      | -     |
| uint32    | +       | +       | -     | -     | -     | -    | -      | +      | -      | -     |
| uint16    | +       | +       | -     | -     | -     | -    | -      | -      | +      | -     |
| uint8     | +       | +       | -     | -     | -     | -    | -      | -      | -      | +     |

### Functions

| Function \ Data type | float32 | float64 | int64 | int32 | int16 | int8 |
|----------------------|---------|---------|-------|-------|-------|------|
| add                  | +       | +       | +     | +     | +     | +    |
| sub                  | +       | +       | +     | +     | +     | +    |
| mul                  | +       | +       | -     | -     | -     | -    |
| div                  | +       | +       | -     | -     | -     | -    |
| lt                   | +       | +       | -     | -     | -     | -    |
| le                   | +       | +       | -     | -     | -     | -    |
| eq                   | +       | +       | -     | -     | -     | -    |
| neq                  | +       | +       | -     | -     | -     | -    |
| gt                   | +       | +       | -     | -     | -     | -    |
| ge                   | +       | +       | -     | -     | -     | -    |
|----------------------|---------|---------|-------|-------|-------|------|
| fabs                 | +       | +       | -     | -     | -     | -    |
| trunc                | +       | +       | NA    | NA    | NA    | NA   |
| floor                | +       | +       | NA    | NA    | NA    | NA   |
| ceil                 | +       | +       | NA    | NA    | NA    | NA   |
| round                | +       | +       | NA    | NA    | NA    | NA   |
|----------------------|---------|---------|-------|-------|-------|------|
| sin                  | +       | +       | NA    | NA    | NA    | NA   |
| cos                  | +       | +       | NA    | NA    | NA    | NA   |
| tan                  | +       | +       | NA    | NA    | NA    | NA   |
| asin                 | +       | +       | NA    | NA    | NA    | NA   |
| acos                 | +       | +       | NA    | NA    | NA    | NA   |
| atan                 | +       | +       | NA    | NA    | NA    | NA   |
| sinh                 | +       | +       | NA    | NA    | NA    | NA   |
| cosh                 | +       | +       | NA    | NA    | NA    | NA   |
| tanh                 | +       | +       | NA    | NA    | NA    | NA   |
| asinh                | +       | +       | NA    | NA    | NA    | NA   |
| acosh                | +       | +       | NA    | NA    | NA    | NA   |
| atanh                | +       | +       | NA    | NA    | NA    | NA   |
|----------------------|---------|---------|-------|-------|-------|------|
| pow                  | +       | +       | -     | -     | -     | -    |
| atan2                | +       | +       | NA    | NA    | NA    | NA   |
| log                  | +       | +       | NA    | NA    | NA    | NA   |
| log2                 | +       | +       | NA    | NA    | NA    | NA   |
| log10                | +       | +       | NA    | NA    | NA    | NA   |
| log1p                | +       | +       | NA    | NA    | NA    | NA   |
| exp                  | +       | +       | NA    | NA    | NA    | NA   |
| exp2                 | +       | +       | NA    | NA    | NA    | NA   |
| exp10                | +       | +       | NA    | NA    | NA    | NA   |
| expm1                | +       | +       | NA    | NA    | NA    | NA   |


## Disclaimer

I am not primarily a C developer, and thus might not abide by the C conventions. A simple example is I'm using a `make.sh` instead of a `Makefile`. This may change once (re)learning these things becomes a higher priority for me.

