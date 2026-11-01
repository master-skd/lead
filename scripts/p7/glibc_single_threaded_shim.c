#include <stdint.h>

int glibc_28_single_threaded = 1;

__asm__(
    ".symver glibc_28_single_threaded,__libc_single_threaded@GLIBC_2.32"
);
