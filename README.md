# BMAS - Basic Mathematical Subprograms

> Run `bash make.sh` to build; modify for non-gcc / non-avx2 platforms.
> The interface to this library is not stable yet. Potential future changes include naming of bitwise operators. 

This library intends to provide functions for operating on vectors of numbers.
The function signatures are based on BLAS L1 functions, and comprise of:

```
BMAS_{ITYPE}{name}(long n, {type* ptr, long inc_ptr}*)
```

- ITYPE can be one of `s d i8 i16 i32 i64 u8 u16 u32 u64`
- n - number of elements to operate upon
- ptr, inc_ptr - one or more pairs of pointer to vector, and stride

For example, the function `BMAS_ssin(n, float* in, long inc_in, float* out, long inc_out)` calculates the sin of the single-floating point numbers in the vector defined by (in, inc\_in) and stores the result in the vector defined by (out, inc\_out).

Exceptions for this pattern include `BMAS_cast_{ITYPE}{OTYPE}` function for converting from ITYPE to OTYPE.

See the [bmas.h](./bmas.h) for the list of currently provided functions.

Correctness is checked by the tests in [numericals](https://github.com/digikar99/numericals).

## Implementation details

The actual computation is done by simd functions from hardware simd instructions, [SLEEF](https://sleef.org/) and/or libmvec*. These are defined inside the [simd](./simd/) directory.

*Currently, [libmvec](https://github.com/sgallagher/glibc/blob/master/sysdeps/unix/sysv/linux/x86_64/libmvec.abilist) is used for single-float sine and cosine for AVX2, since they were found to be faster than their SLEEF counterparts.

### Other notes:

- gcc uses arithmetic shift on signed values and logical shift on unsigned values.
- `long` has been used as equivalent to 8 bytes; perhaps this will be fixed in the future to account for machine-OS specificity

## AVX2 Status

SSE and AVX512 support exists to a limited extent due to limited developer time.

### Copy and Conversions

- `BMAS_cast_{ITYPE}{OTYPE}`
- `BMAS_{TYPE}copy`

| From \ To | float32 | float64 | int64 | int32 | int16 | int8 | uint64 | uint32 | uint16 | uint8 |
|----------:|:-------:|:-------:|:-----:|:-----:|:-----:|:----:|:------:|:------:|:------:|:-----:|
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

| Function \ Data type               | float32 | float64 | int64 | int32 | int16 | int8 | uint64 | uint32 | uint16 | uint8 |
|------------------------------------|:-------:|:-------:|:-----:|:-----:|:-----:|:----:|:------:|:------:|:------:|:-----:|
| add                                | +       | +       | +     | +     | +     | +    | -      | -      | -      | -     |
| sub                                | +       | +       | +     | +     | +     | +    | -      | -      | -      | -     |
| mul                                | +       | +       | +     | +     | +     | +    | +      | +      | +      | +     |
| div                                | +       | +       | -     | -     | -     | -    | -      | -      | -      | -     |
| abs (also fabs below)              | -       | -       | +     | +     | +     | +    | -      | -      | -      | -     |
| sum                                | +       | +       | +     | +     | +     | +    | -      | -      | -      | -     |
| dot                                | +       | +       | +     | +     | +     | +    | -      | -      | -      | -     |
| min                                | +       | +       | +     | +     | +     | +    | +      | +      | +      | +     |
| max                                | +       | +       | +     | +     | +     | +    | +      | +      | +      | +     |
| **Function \ Data type**           | float32 | float64 | int64 | int32 | int16 | int8 | uint64 | uint32 | uint16 | uint8 |
| lt                                 | +       | +       | +     | +     | +     | +    | +      | +      | +      | +     |
| le                                 | +       | +       | +     | +     | +     | +    | +      | +      | +      | +     |
| eq                                 | +       | +       | +     | +     | +     | +    | +      | +      | +      | +     |
| neq                                | +       | +       | +     | +     | +     | +    | +      | +      | +      | +     |
| gt                                 | +       | +       | +     | +     | +     | +    | +      | +      | +      | +     |
| ge                                 | +       | +       | +     | +     | +     | +    | +      | +      | +      | +     |
| **Function \ Data type (Bitwise)** | float32 | float64 | int64 | int32 | int16 | int8 | uint64 | uint32 | uint16 | uint8 |
| not                                | -       | -       | -     | -     | -     | +    | -      | -      | -      | -     |
| and                                | -       | -       | -     | -     | -     | +    | -      | -      | -      | -     |
| or                                 | -       | -       | -     | -     | -     | +    | -      | -      | -      | -     |
| nor                                | -       | -       | -     | -     | -     | +    | -      | -      | -      | -     |
| andnot                             | -       | -       | -     | -     | -     | +    | -      | -      | -      | -     |
| srl                                | -       | -       | -     | -     | -     | -    | +      | +      | +      | +     |
| sra                                | -       | -       | +     | +     | +     | +    | -      | -      | -      | -     |



| Function \ Data type | float32 | float64 |
|----------------------|:-------:|:-------:|
| fabs                 | +       | +       |
| trunc                | +       | +       |
| floor                | +       | +       |
| ceil                 | +       | +       |
| round                | +       | +       |
|----------------------|:-------:|:-------:|
| sin                  | +       | +       |
| cos                  | +       | +       |
| tan                  | +       | +       |
| asin                 | +       | +       |
| acos                 | +       | +       |
| atan                 | +       | +       |
| sinh                 | +       | +       |
| cosh                 | +       | +       |
| tanh                 | +       | +       |
| asinh                | +       | +       |
| acosh                | +       | +       |
| atanh                | +       | +       |
|----------------------|:-------:|:-------:|
| pow                  | +       | +       |
| atan2                | +       | +       |
| log                  | +       | +       |
| log2                 | +       | +       |
| log10                | +       | +       |
| log1p                | +       | +       |
| exp                  | +       | +       |
| exp2                 | +       | +       |
| exp10                | +       | +       |
| expm1                | +       | +       |
|----------------------|:-------:|:-------:|


## Disclaimer

I am not primarily a C developer, and thus might not abide by the C conventions. A simple example is I'm using a `make.sh` instead of a `Makefile`. This may change once (re)learning these things becomes a higher priority for me.

## Credits

Include but not limited to:

- https://stackoverflow.com/questions/41144668/how-to-efficiently-perform-double-int64-conversions-with-sse-avx
- https://stackoverflow.com/a/37322570/8957330
