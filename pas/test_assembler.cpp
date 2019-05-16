#include <stdio.h>
#include <assert.h>
#include "pas.h"

void print_hex64(uint64_t x)
{
    uint32_t x_lo = x & 0xffffffff;
    uint32_t x_hi = (x >> 32) & 0xffffffff;
    printf("%08x%08x", x_hi, x_lo);
}

#define TEST(a,b) { if ((a) != (b)) { printf("Test failed at line %d\nWas: ", __LINE__); print_hex64(a); printf("\nGot: "); print_hex64(b); printf("\n"); } }

int main()
{
    TEST(0x5ca8148000470a0d, pas_FLOOR32F(0x0d, 0x04));                                                    // F2F.FTZ.F32.F32.FLOOR R13, R4;

    TEST(0x5c58100000a70700, pas_FADD(0x00, 0x07, 0x0a, FADD_FTZ));                                        // FADD.FTZ R0, R7, R10;
    TEST(0x3958103fc0070003, pas_FADD20I(0x03, 0x00, -1.5f, FADD_FTZ));                                    // FADD.FTZ R3, R0, -1.5;
    TEST(0x3858103f80070404, pas_FADD20I(0x04, 0x04, +1.0f, FADD_FTZ));                                    // FADD.FTZ R4, R4, 1;
    TEST(0x3858503f00070409, pas_FADD20I(0x09, 0x04, +0.5f, FADD_FTZ|FADD_ABS_A));                         // FADD.FTZ R9, |R4|.reuse, 0.5;
    TEST(0x0883f33333370408, pas_FADD32I(0x08, 0x04, +0.69999998807907104492f, FADD32I_FTZ));              // FADD32I.FTZ R8, R4.reuse, 0.69999998807907104492;
    TEST(0x088bf4f5c2970809, pas_FADD32I(0x09, 0x08, -0.81000000238418579102f, FADD32I_FTZ));              // FADD32I.FTZ R9, R8, -0.81000000238418579102;
    TEST(0x08cbfe666667030d, pas_FADD32I(0x0d, 0x03, -1.7999999523162841797f, FADD32I_FTZ|FADD32I_ABS_A)); // FADD32I.FTZ R13, |R3|, -1.7999999523162841797;

    // todo: (FMUL d, a, b) where a != b
    // todo: (FMUL d, |a|, b)
    // todo: (FMUL d, a, |b|)
    // todo: (FMUL d, a, -b)
    // todo: (FMUL d, -a, b)
    TEST(0x5c6810000067060a, pas_FMUL(0x0a, 0x06, 0x06, FMUL_FTZ));                                        // FMUL.FTZ R10, R6, R6;
    TEST(0x5c68100000570503, pas_FMUL(0x03, 0x05, 0x05, FMUL_FTZ));                                        // FMUL.FTZ R3, R5, R5;
    TEST(0x3868103f0007000c, pas_FMUL20I(0x0c, 0x00, 0.5f, FMUL_FTZ));                                     // FMUL.FTZ R12, R0.reuse, 0.5;
    TEST(0x3868103f33070004, pas_FMUL20I(0x04, 0x00, 0.69921875f, FMUL_FTZ));                              // FMUL.FTZ R4, R0.reuse, 0.69921875;
    TEST(0x3868103ffff70005, pas_FMUL20I(0x05, 0x00, 1.99951171875f, FMUL_FTZ));                           // FMUL.FTZ R5, R0.reuse, 1.99951171875;
    TEST(0x3968103f00070007, pas_FMUL20I(0x07, 0x00, -0.5f, FMUL_FTZ));                                    // FMUL.FTZ R7, R0.reuse, -0.5;
    TEST(0x3968103f33070008, pas_FMUL20I(0x08, 0x00, -0.69921875f, FMUL_FTZ));                             // FMUL.FTZ R8, R0.reuse, -0.69921875;
    TEST(0x3968103ffff70009, pas_FMUL20I(0x09, 0x00, -1.99951171875f, FMUL_FTZ));                          // FMUL.FTZ R9, R0.reuse, -1.99951171875;
    TEST(0x1e23f33333370407, pas_FMUL32I(0x07, 0x04, 0.69999998807907104492f, FMUL32I_FTZ));               // FMUL32I.FTZ R7, R4.reuse, 0.69999998807907104492;
    TEST(0x1e2bf3235eb70507, pas_FMUL32I(0x07, 0x05, -0.6961352229118347168f, FMUL32I_FTZ));               // FMUL32I.FTZ R7, R5, -0.6961352229118347168;

    // todo: FFMA d, a imm, b, c
    // todo: FFMA d, a, b imm, c
    // todo: FFMA d, a, b, c imm
    TEST(0x59a0050000470408, pas_FFMA(0x08, 0x04, 0x04, 0x0a, FFMA_FTZ));                                  // FFMA.FTZ R8, R4, R4, R10;
    TEST(0x32a0033e80070008, pas_FFMA20I(0x08, 0x00, 0.25f, 0x06, FFMA_FTZ));                              // FFMA.FTZ R8, R0.reuse, 0.25, R6.reuse;
    TEST(0x32a0033f0ff70009, pas_FFMA20I(0x09, 0x00, 0.562255859375f, 0x06, FFMA_FTZ));                    // FFMA.FTZ R9, R0.reuse, 0.562255859375, R6.reuse;
    TEST(0x33a0033e8007000c, pas_FFMA20I(0x0c, 0x00, -0.25f, 0x06, FFMA_FTZ));                             // FFMA.FTZ R12, R0.reuse, -0.25, R6.reuse;

    // todo: FMAX d, a, -b
    // todo: FMAX d, -a, b
    // todo: FMAX d, -a, -b
    // todo: FMIN d, a, -b
    // todo: FMIN d, -a, b
    // todo: FMIN d, -a, -b
    TEST(0x5c60178000a70909, pas_FMAX( 9,  9, 10, FMNMX_FTZ));                                             // FMAX.FTZ R9, R9, R10;
    TEST(0x5c60378000970303, pas_FMAX( 3,  3,  9, FMNMX_FTZ|FMNMX_NEG_B));                                 // FMAX.FTZ R3, R3, -R9;
    TEST(0x5c60138000570004, pas_FMIN( 4,  0,  5, FMNMX_FTZ));                                             // FMIN.FTZ R4, R0, R5;
    TEST(0x386017bf0007000b, pas_FMAX20I(11,  0, 0.5f, FMNMX_FTZ));                                        // FMAX.FTZ R11, R0, 0.5;
    TEST(0x386017bffff70004, pas_FMAX20I( 4,  0, 1.99951171875f, FMNMX_FTZ));                              // FMAX.FTZ R4, R0.reuse, 1.99951171875;
    TEST(0x396017bffff70008, pas_FMAX20I( 8,  0, -1.99951171875f, FMNMX_FTZ));                             // FMAX.FTZ R8, R0.reuse, -1.99951171875;
    TEST(0x386013bf00070009, pas_FMIN20I( 9,  0, 0.5f, FMNMX_FTZ));                                        // FMIN.FTZ R9, R0.reuse, 0.5;
    TEST(0x386013bffff7000c, pas_FMIN20I(12,  0, 1.99951171875f, FMNMX_FTZ));                              // FMIN.FTZ R12, R0.reuse, 1.99951171875;
    TEST(0x396013bffff70007, pas_FMIN20I( 7,  0, -1.99951171875f, FMNMX_FTZ));                             // FMIN.FTZ R7, R0, -1.99951171875;

    TEST(0x508000000087070b, pas_MUFU_SQRT(11,  7));                                                       // MUFU.SQRT R11, R7;

    TEST(0x5c5930000ff70403, pas_NEG(0x03, 0x04, FADD_FTZ));                                               // FADD.FTZ R3, -R4, -RZ;

    pas_reset_ctrl();
    pas_stall(0,2);
    pas_set_write_barrier(0,2);
    pas_stall(1,0);
    pas_stall(2,4);
    pas_set_read_barrier(2,3);
    pas_set_write_barrier(2,4);
    TEST(0x0009d000fe000732, pas_CTRL()); // --:-:2:-:2 [0000] | --:-:-:-:0 [0000] | --:3:4:-:4 [0000]
}
