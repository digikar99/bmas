
SHELL=/data/data/com.termux/files/usr/bin/bash
ARCH=$(uname -m)

if [ "$ARCH" == "x86_64" ]; then
    gcc -O3 -mavx2 -mfma -shared -o libbmas.so -fpic bmas.c -lm
    # gcc -O3 -mavx512f -shared -o libbmas.so -fpic bmas.c
    # gcc -O3 -msse2 -shared -o libbmas.so -fpic bmas.c
    # TODO: Prepare according to different architectures
else
    ## FIXME for aarch64
    gcc -O3 -shared -o libbmas.so -fpic bmas.c
fi
