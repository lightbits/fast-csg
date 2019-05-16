#include <stdio.h>
#include <assert.h>
#include "pas.h"
#include "cubin.h"

uint64_t *find_instruction(uint64_t *instructions, int num_instructions,
                           uint64_t *pattern, int len_pattern)
{
    assert(len_pattern > 0);
    for (int i = 0; i < num_instructions-len_pattern; i++)
    {
        bool match = true;
        for (int j = 0; j < len_pattern; j++)
        {
            if (instructions[i+j] != pattern[j])
                match = false;
        }
        if (match)
            return instructions+i;
    }
    return NULL;
}

int main()
{
    #if 0
    // The source for this binary only has one interesting instruction in the 'test1' call
    // This patch can modify that single instruction, e.g. to verify PAS code generation.
    cubin_t cubin = read_cubin("binary/test4.cubin");
    cubin_function_t *kernel = cubin.get_function("test");
    assert(kernel);
    uint64_t pattern[] = {
        0x001fc000ffe000f1,
        0xeedc200000070200,
        0xe30000000007000f,
        0x5c68100000270200
    };
    int len_pattern = sizeof(pattern)/sizeof(uint64_t);
    uint64_t *i = find_instruction(kernel->instructions(), kernel->num_instructions(), pattern, len_pattern);
    assert(i);

    pas_reset_ctrl();
    pas_stall(0,1);
    pas_stall(1,15);
    pas_stall(2,0);
    *i++ = pas_CTRL();
    *i++ = 0xeedc200000070200; // stg.e [R2], R0
    *i++ = 0xe30000000007000f; // exit
    *i++ = pas_FMUL(0, 2, 2, FMUL_FTZ);

    save_cubin(&cubin, "binary/test4.patched1.cubin");
    #endif

    #if 0
    // The source for this binary performs x*x + y*y + z*z and stores result.
    // For some reason I don't know yet, the first two instructions need a stall of
    // 6, if I change that to 1 the output is incorrect.
    cubin_t cubin = read_cubin("binary/test5.cubin");
    cubin_function_t *kernel = cubin.get_function("test");
    assert(kernel);
    uint64_t pattern[] = {
        0x001fd800fec007ff,
        0xe30000000007000f,
        0x5c68100000470404,
        0x59a0020000570504,
        0x001ffc00ffe007f0,
    };
    int len_pattern = sizeof(pattern)/sizeof(uint64_t);
    uint64_t *i = find_instruction(kernel->instructions(), kernel->num_instructions(), pattern, len_pattern);
    assert(i);
    pas_reset_ctrl();
    pas_stall(0,15);
    pas_stall(1,6);
    pas_stall(2,6);
    *i++ = pas_CTRL();
    *i++ = 0xe30000000007000f; // EXIT
    *i++ = pas_FMUL( 4,  4,  4, FMUL_FTZ);
    *i++ = pas_FFMA( 4,  5,  5,  4, FFMA_FTZ);

    pas_reset_ctrl();
    pas_stall(0,0);
    pas_stall(1,15);
    pas_stall(2,0); pas_yield(2);
    *i++ = pas_CTRL();
    *i++ = pas_FFMA( 4,  6,  6,  4, FFMA_FTZ);
    *i++ = pas_RET();
    *i++ = pas_NOP();

    save_cubin(&cubin, "binary/test5.patched1.cubin");
    #endif

    #if 0
    // The source for this binary performs x*x + y*y + z*z and stores result.
    // It seems like we can modify it and just add stall,6 for every instruction?
    cubin_t cubin = read_cubin("binary/test5.cubin");
    cubin_function_t *kernel = cubin.get_function("test");
    assert(kernel);
    uint64_t pattern[] = {
        0x001fd800fec007ff,
        0xe30000000007000f,
        0x5c68100000470404,
        0x59a0020000570504,
        0x001ffc00ffe007f0,
    };
    int len_pattern = sizeof(pattern)/sizeof(uint64_t);
    uint64_t *i = find_instruction(kernel->instructions(), kernel->num_instructions(), pattern, len_pattern);
    assert(i);

    pas_reset_ctrl();
    pas_stall(0,15);
    pas_stall(1,6);
    pas_stall(2,6);
    *i++ = pas_CTRL();
    *i++ = 0xe30000000007000f; // EXIT
    *i++ = pas_FMAX(0, 4, 5, FMNMX_FTZ);
    // *i++ = pas_FMUL( 0,  4,  4, FMUL_FTZ);
    *i++ = pas_FMUL( 1,  5,  5, FMUL_FTZ);

    pas_reset_ctrl();
    pas_stall(0,6);
    pas_stall(1,6);
    pas_stall(2,15);
    *i++ = pas_CTRL();
    *i++ = pas_FFMA( 1,  6,  6,  0, FFMA_FTZ);
    *i++ = pas_FADD( 4, 1, 1, FADD_FTZ);
    *i++ = pas_RET();

    save_cubin(&cubin, "binary/test5.patched1.cubin");
    #endif

    #if 0
    // The source for this binary performs x*x + y*y + z*z and stores result.
    // Assuming latency = 6 for float instructions, let's actually try and make
    // a pipeline work out here.
    // My working hypothesis is: fp operations take one clock cycle to read an
    // operand register. fp operations take 6 clock cycles to write to the dst
    // register. If a subsequent fp instruction reads from a register written to
    // by the previous, the previous must stall for 6 cycles. If an instruction
    // reads from a register written to by an earlier instruction, the total
    // stall count between them must be >= 6.
    cubin_t cubin = read_cubin("binary/test5.cubin");
    cubin_function_t *kernel = cubin.get_function("test");
    assert(kernel);
    uint64_t pattern[] = {
        0x001fd800fec007ff,
        0xe30000000007000f,
        0x5c68100000470404,
        0x59a0020000570504,
        0x001ffc00ffe007f0,
    };
    int len_pattern = sizeof(pattern)/sizeof(uint64_t);
    uint64_t *i = find_instruction(kernel->instructions(), kernel->num_instructions(), pattern, len_pattern);
    assert(i);
    pas_reset_ctrl();
    pas_stall(0,15);
    pas_stall(1,1); // we need minimum 1 stall? otherwise get illegal instruction
    pas_stall(2,1);
    *i++ = pas_CTRL();
    *i++ = 0xe30000000007000f; // EXIT
    *i++ = pas_FMUL( 4,  4,  5, FMUL_FTZ); // this can read from 5 even though next one writes to it... and the stall is 1
    *i++ = pas_FMUL( 5,  5,  6, FMUL_FTZ);

    pas_reset_ctrl();
    pas_stall(0,6);
    pas_stall(1,0); // this one can have stall 0 because next one is RET?
    pas_stall(2,15);
    *i++ = pas_CTRL();
    *i++ = pas_FMUL( 6,  6,  6, FMUL_FTZ);
    *i++ = pas_FFMA( 4, 4, 5, 6, FFMA_FTZ);
    *i++ = pas_RET();

    save_cubin(&cubin, "binary/test5.patched1.cubin");
    #endif

    #if 0
    // The source for this binary performs x*x + y*y + z*z and stores result.
    // Assuming latency = 6 for float instructions, let's actually try and make
    // a pipeline work out here.
    // My working hypothesis is:
    // * NOPs can be stalled.
    // * CTRLs do not take a cycle?
    cubin_t cubin = read_cubin("binary/test5.cubin");
    cubin_function_t *kernel = cubin.get_function("test");
    assert(kernel);
    uint64_t pattern[] = {
        0x001fd800fec007ff,
        0xe30000000007000f,
        0x5c68100000470404,
        0x59a0020000570504,
        0x001ffc00ffe007f0,
    };
    int len_pattern = sizeof(pattern)/sizeof(uint64_t);
    uint64_t *i = find_instruction(kernel->instructions(), kernel->num_instructions(), pattern, len_pattern);
    assert(i);

    pas_reset_ctrl();
    pas_stall(0,15);
    pas_stall(1,1);
    pas_stall(2,4);
    *i++ = pas_CTRL();
    *i++ = 0xe30000000007000f; // EXIT
    *i++ = pas_FMUL( 5, 5, 5, FMUL_FTZ);
    *i++ = pas_NOP();

    pas_reset_ctrl();
    pas_stall(0,1);
    pas_stall(1,1);
    pas_stall(2,15);
    *i++ = pas_CTRL(); // this apparently takes no cycles?
    *i++ = pas_NOP();
    *i++ = pas_FADD( 4, 5, 6, FADD_FTZ);
    *i++ = pas_RET();

    save_cubin(&cubin, "binary/test5.patched1.cubin");
    #endif

    #if 1
    // The source for this binary performs x*x + y*y + z*z and stores result.
    // Assuming latency = 6 for float instructions, let's actually try and make
    // a pipeline work out here.
    // My working hypothesis is: FFMA need only stall 2 cycles if the next
    // instruction that uses its result is a SQRT.
    cubin_t cubin = read_cubin("binary/test5.cubin");
    cubin_function_t *kernel = cubin.get_function("test");
    assert(kernel);
    uint64_t pattern[] = {
        0x001fd800fec007ff,
        0xe30000000007000f,
        0x5c68100000470404,
        0x59a0020000570504,
        0x001ffc00ffe007f0,
    };
    int len_pattern = sizeof(pattern)/sizeof(uint64_t);
    uint64_t *i = find_instruction(kernel->instructions(), kernel->num_instructions(), pattern, len_pattern);
    assert(i);

    pas_reset_ctrl();
    pas_stall(0,15);
    pas_stall(1,2);
    pas_stall(2,2); // not sure how long we need to stall here, it doesn't seem to matter
    pas_set_write_barrier(2,5);
    *i++ = pas_CTRL();
    *i++ = 0xe30000000007000f; // EXIT
    *i++ = pas_FMUL( 5, 5, 5, FMUL_FTZ);
    *i++ = pas_MUFU_SQRT(5, 5);

    pas_reset_ctrl();
    pas_stall(0,1);
    pas_stall(1,1);
    pas_wait_on_barrier(1,5);
    pas_stall(2,15);
    *i++ = pas_CTRL();
    *i++ = pas_NOP();
    *i++ = pas_FADD( 4, 5, 5, FADD_FTZ);
    *i++ = pas_RET();

    save_cubin(&cubin, "binary/test5.patched1.cubin");
    #endif

    #if 0
    // The source for this binary is a lot more complicated. The patch below
    // modifies the first instruction in the 'test1' function and follows it
    // with RET and NOP, essentially bypassing the rest of the function. This
    // is what I'll eventually need to do when I patch bigger programs. Luckily,
    // it seems like you can just add a ret and nop and it will work. nvdisasm
    // freaks out if you mess up the "one ctrl for every three instructions",
    // but the GPU seems to execute it just fine...?
    cubin_t cubin = read_cubin("binary/test7.cubin");
    cubin_function_t *kernel = cubin.get_function("test");
    assert(kernel);
    uint64_t pattern[] = {
        0x001f8400ffe000f1,
        0xeedc200000070200,
        0xe30000000007000f,
        0x3958503f80070400
    };
    int len_pattern = sizeof(pattern)/sizeof(uint64_t);
    uint64_t *i = find_instruction(kernel->instructions(), kernel->num_instructions(), pattern, len_pattern);
    assert(i);

    pas_reset_ctrl();
    pas_stall(0,1);
    pas_stall(1,15);
    pas_stall(2,1);
    *i++ = pas_CTRL();
    *i++ = 0xeedc200000070200; // STG.E[r2],r0
    *i++ = 0xe30000000007000f; // EXIT
    *i++ = pas_FADD20I( 0,  4, -1.0f, FADD_FTZ|FADD_ABS_A);

    pas_reset_ctrl();
    pas_stall(0,15);
    pas_stall(1,0); pas_yield(1);
    pas_stall(2,0); pas_yield(1);
    *i++ = pas_CTRL();
    *i++ = 0xe32000000007000f; // RET
    *i++ = pas_NOP();
    *i++ = pas_NOP();

    save_cubin(&cubin, "binary/test7.patched1.cubin");
    #endif
}
