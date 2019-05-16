// Maxwell and Pascal have two 32-bit floating point instruction types
// that accept immediate values as one or more of their operands.
// FADD32I, FMUL32I, FFMA32I all encode a 32 bit floating point number
// directly in the instruction, taking the place of the B operand.
// (I'm using the convention: <instruction> <d>, <a>, <b>, <c>)
// However, there are also FADD, FMUL, FFMA, FMNMX that can encode
// 20 bits of a fp32 number. This is treated by the ALUs as a 32-bit
// single-precision float, but the instruction truncates the rightmost
// 12 bits of the mantissa to zero.
//   The sign bit is also shifted to a different place in the instruction,
// for both 20-bit and 32-bit immediate versions.
//   To encode a 20-bit float immediate value, apply the mask 0x7FFFF000
// and right-shift the result by 12 bits. To encode a 32-bit float immediate
// apply the mask 0x7FFFFFFF. See respective instructions for where the sign
// bit ends up, and where the value itself is placed inside the instruction
// (usually left-shifted 20 bits in).
#include <stdio.h>
#include <stdint.h>

int main()
{
    const uint32_t mask = 0x7FFFF000;
    // float x = -0.7f; // BF333333 = 10111111001100110011001100110011
    // float x = -0.5f; // BF000000 = 10111111000000000000000000000000
    // float x = -1.0f; // BF800000 = 10111111100000000000000000000000
    // float x = -1.5f; // BFC00000 = 10111111110000000000000000000000
    // float x = 0.000244140625f; // 39800000
    // float x = 32.339f; // 42015B23
    // uint32_t i = 0xBF330000; float x = *(float*)&i;
    // uint32_t i = 0x42010000; float x = *(float*)&i;
    // float x = 0.69921875f;
    uint32_t i = 0xBFFFFFFF & mask; float x = *(float*)&i;
    printf("%08X\t%.20f\n", *(uint32_t*)&x, x);

    // compare precision of 20-bit and 32-bit fp
    {
        uint32_t mask = 0x7FFFF000;
        float x = 0.1f;
        for (int i = 0; i < 32; i++)
        {
            x *= 2.0f;
            uint32_t v = *(uint32_t*)&x;
            v &= mask;
            printf("%f %f\n", x, *(float*)&v);
        }
    }
}


// a = left operand register [0,255]
// b = destination register [0,255]
// v = immediate value
// i = instruction | flags
// Flag bits:
// X1XX -> |a|
// XXX1 -> FTZ
//                                                             iiiii ?vvvv?aabb
// FADD R3, |R4|, -1;                                     /* 0x3958403f80070403 */
// FADD R4, |R7|, -1;                                     /* 0x3958403f80070704 */
// FADD R6, R6, -0.5;                                     /* 0x3958003f00070606 */
// FADD R0, R4, -1;                                       /* 0x3958003f80070400 */ 0x39... -> sub
// FADD.FTZ R0, R7, 1;                                    /* 0x3858103f80070700 */ 0x38... -> add
// FADD.FTZ R5, R4.reuse, 1;                              /* 0x3858103f80070405 */
// FADD.FTZ R0, |R7|, -0.5;                               /* 0x3958503f00070700 */
// FADD.FTZ R3, |R4|, -0.5;                               /* 0x3958503f00070403 */
// FADD.FTZ R5, |R5|, -0.5;                               /* 0x3958503f00070505 */

//                                                             iiivvvvvvvv?aabb
// FADD32I.FTZ R5, R0, -0.69999998807907104492;           /* 0x088bf33333370005 */
// FADD32I.FTZ R3, R6, -0.69999998807907104492;           /* 0x088bf33333370603 */


// test2.nvdisasm constant memory
// a5a4 87a1 d5c9 e13f 0000 00f0 ffff 0f38
// ad4e 8e3f 0000 0080 a5a4 87a1 d5c9 e1bf
