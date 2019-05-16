#include <stdio.h>
#include "cubin.h"

int main()
{
    // cubin_t cubin = read_cubin("binary/patch_prototype1.cubin");
    // cubin_t cubin = read_cubin("binary/test4.cubin");
    cubin_t cubin = read_cubin("binary/test7.cubin");
    // cubin_function_t *func = cubin.get_function("test");
    // assert(func);
    // printf("regcnt: %d\n", func->register_count());
    // func->set_register_count(32);
    // save_cubin(&cubin, "binary/test4.cubin1");
}
