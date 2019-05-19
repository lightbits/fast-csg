#pragma once

namespace backend_sass {

uint64_t FADD_FTZ      = 0x0000100000000000;
uint64_t FADD_NEG_A    = 0x0001000000000000;
uint64_t FADD_NEG_B    = 0x0000200000000000;
uint64_t FADD_ABS_A    = 0x0000400000000000;
uint64_t FADD_ABS_B    = 0x0002000000000000;

uint64_t FADD32I_FTZ   = 0x0080000000000000;
uint64_t FADD32I_ABS_A = 0x0040000000000000;

uint64_t FMUL_FTZ      = 0x0000100000000000;
uint64_t FMUL_NEG_B    = 0x0001000000000000;

uint64_t FMUL32I_FTZ   = 0x0020000000000000;

uint64_t FMNMX_FTZ     = 0x0000100000000000;
uint64_t FMNMX_NEG_A   = 0x0001000000000000;
uint64_t FMNMX_NEG_B   = 0x0000200000000000;
uint64_t FMNMX_ABS_A   = 0x0000400000000000;
uint64_t FMNMX_ABS_B   = 0x0002000000000000;

uint64_t FFMA_FTZ      = 0x0020000000000000;
uint64_t FFMA_NEG_B    = 0x0001000000000000;
uint64_t FFMA_NEG_C    = 0x0002000000000000;

// FADD d, a, b
// d = a+b
uint64_t FADD(uint8_t d, uint8_t a, uint8_t b, uint64_t flags) {
    uint64_t RD  = (uint64_t)(d) << 0;
    uint64_t RA  = (uint64_t)(a) << 8;
    uint64_t RB  = (uint64_t)(b) << 20;
    return 0x5c58000000070000 | flags | RB | RA | RD;
}

// FADD d, -a, -RZ
// d = a+b
uint64_t NEG(uint8_t d, uint8_t a, uint64_t flags) {
    uint64_t RD  = (uint64_t)(d) << 0;
    uint64_t RA  = (uint64_t)(a) << 8;
    // todo: why is NEG_B flag set?
    return 0x5c5930000ff70000 | flags | RA | RD;
}

// FADD d, a, b immediate
// d = a+b
uint64_t FADD20I(uint8_t d, uint8_t a, float b, uint64_t flags) {
    uint64_t b_u64 = *(uint64_t*)&b;
    uint64_t sgn_b = b_u64 & 0x0000000080000000;
    uint64_t NEG_B = sgn_b ? 0x0100000000000000 : 0x0;
    uint64_t B     = ((b_u64 & 0x000000007FFFF000) >> 12) << 20;
    uint64_t RA    = (uint64_t)(a) << 8;
    uint64_t RD    = (uint64_t)(d) << 0;
    return 0x3858000000070000 | flags | NEG_B | B | RA | RD;
}

// FADD32I d, a, b immediate
// d = a+b
uint64_t FADD32I(uint8_t d, uint8_t a, float b, uint64_t flags) {
    uint64_t b_u64 = *(uint64_t*)&b;
    uint64_t sgn_b = b_u64 & 0x0000000080000000;
    uint64_t NEG_B = sgn_b ? 0x0008000000000000 : 0x0;
    uint64_t B     = (b_u64 & 0x000000007FFFFFFF) << 20;
    uint64_t RA    = (uint64_t)(a) << 8;
    uint64_t RD    = (uint64_t)(d) << 0;
    return 0x0880000000070000 | flags | NEG_B | B | RA | RD;
}

// FTF.FTZ.F32.F32.FLOOR d, b
// d = floor(b)
uint64_t FLOOR32F(uint8_t d, uint8_t b) {
    uint64_t RB    = (uint64_t)(b) << 20;
    uint64_t RD    = (uint64_t)(d) << 0;
    return 0x5ca8148000070a00 | RB | RD;
}

// FMUL32I d, a, b immediate
// d = a*b
uint64_t FMUL32I(uint8_t d, uint8_t a, float b, uint64_t flags) {
    uint64_t b_u64 = *(uint64_t*)&b;
    uint64_t sgn_b = b_u64 & 0x0000000080000000;
    uint64_t NEG_B = sgn_b ? 0x0008000000000000 : 0x0;
    uint64_t B     = (b_u64 & 0x000000007FFFFFFF) << 20;
    uint64_t RA    = (uint64_t)(a) << 8;
    uint64_t RD    = (uint64_t)(d) << 0;
    return 0x1e00000000070000 | flags | NEG_B | B | RA | RD;
}

// FMUL d, a, b immediate
// d = a*b
uint64_t FMUL20I(uint8_t d, uint8_t a, float b, uint64_t flags) {
    uint64_t b_u64 = *(uint64_t*)&b;
    uint64_t sgn_b = b_u64 & 0x0000000080000000;
    uint64_t NEG_B = sgn_b ? 0x0100000000000000 : 0x0;
    uint64_t B     = ((b_u64 & 0x000000007FFFF000) >> 12) << 20;
    uint64_t RA    = (uint64_t)(a) << 8;
    uint64_t RD    = (uint64_t)(d) << 0;
    return 0x3868000000070000 | flags | NEG_B | B | RA | RD;
}

// FMUL d, a, b
// d = a*b
uint64_t FMUL(uint8_t d, uint8_t a, uint8_t b, uint64_t flags) {
    uint64_t RD  = (uint64_t)(d) << 0;
    uint64_t RA  = (uint64_t)(a) << 8;
    uint64_t RB  = (uint64_t)(b) << 20;
    return 0x5c68000000070000 | flags | RB | RA | RD;
}

// FFMA d, a, b, c
// d = a*b + c
uint64_t FFMA(uint8_t d, uint8_t a, uint8_t b, uint8_t c, uint64_t flags) {
    uint64_t RD  = (uint64_t)(d) << 0;
    uint64_t RA  = (uint64_t)(a) << 8;
    uint64_t RB  = (uint64_t)(b) << 20;
    uint64_t RC  = (uint64_t)(c) << 39;
    return 0x5980000000070000 | flags | RC | RB | RA | RD;
}

// FFMA d, a, b immediate, c
// d = a*b + c
uint64_t FFMA20I(uint8_t d, uint8_t a, float b, uint8_t c, uint64_t flags) {
    uint64_t b_u64 = *(uint64_t*)&b;
    uint64_t sgn_b = b_u64 & 0x0000000080000000;
    uint64_t NEG_B = sgn_b ? 0x0100000000000000 : 0x0;
    uint64_t B     = ((b_u64 & 0x000000007FFFF000) >> 12) << 20;
    uint64_t RC    = (uint64_t)(c) << 39;
    uint64_t RA    = (uint64_t)(a) << 8;
    uint64_t RD    = (uint64_t)(d) << 0;
    return 0x3280000000070000 | flags | NEG_B | RC | B | RA | RD;
}

// FMNMX d, a, b, !PT
// d = max(a,b)
uint64_t FMAX(uint8_t d, uint8_t a, uint8_t b, uint64_t flags) {
    uint64_t RD  = (uint64_t)(d) << 0;
    uint64_t RA  = (uint64_t)(a) << 8;
    uint64_t RB  = (uint64_t)(b) << 20;
    return 0x5c60078000070000 | flags | RB | RA | RD;
}

// FMNMX d, a, b, PT
// d = min(a,b)
uint64_t FMIN(uint8_t d, uint8_t a, uint8_t b, uint64_t flags) {
    uint64_t RD  = (uint64_t)(d) << 0;
    uint64_t RA  = (uint64_t)(a) << 8;
    uint64_t RB  = (uint64_t)(b) << 20;
    return 0x5c60038000070000 | flags | RB | RA | RD;
}

// FMNMX d, a, b immediate, !PT
// d = min(a,b)
uint64_t FMAX20I(uint8_t d, uint8_t a, float b, uint64_t flags) {
    uint64_t b_u64 = *(uint64_t*)&b;
    uint64_t sgn_b = b_u64 & 0x0000000080000000;
    uint64_t NEG_B = sgn_b ? 0x0100000000000000 : 0x0;
    uint64_t B     = ((b_u64 & 0x000000007FFFF000) >> 12) << 20;
    uint64_t RA    = (uint64_t)(a) << 8;
    uint64_t RD    = (uint64_t)(d) << 0;
    return 0x3860078000070000 | NEG_B | flags | B | RA | RD;
}

// FMNMX d, a, b immediate, PT
// d = min(a,b)
uint64_t FMIN20I(uint8_t d, uint8_t a, float b, uint64_t flags) {
    uint64_t b_u64 = *(uint64_t*)&b;
    uint64_t sgn_b = b_u64 & 0x0000000080000000;
    uint64_t NEG_B = sgn_b ? 0x0100000000000000 : 0x0;
    uint64_t B     = ((b_u64 & 0x000000007FFFF000) >> 12) << 20;
    uint64_t RA    = (uint64_t)(a) << 8;
    uint64_t RD    = (uint64_t)(d) << 0;
    return 0x3860038000070000 | NEG_B | flags | B | RA | RD;
}

// MUFU.SQRT d, a
// d = sqrt(a)
uint64_t MUFU_SQRT(uint8_t d, uint8_t a) {
    uint64_t RD  = (uint64_t)(d) << 0;
    uint64_t RA  = (uint64_t)(a) << 8;
    return 0x5080000000870000 | RA | RD;
}

// NOP should be issued along with --:-:-:Y:0 control codes
uint64_t NOP() { return 0x50b0000000070f00; }
// RET should be issued along with --:-:-:-:f control codes
uint64_t RET() { return 0xe32000000007000f; }

struct control_flags_t
{
    uint8_t reuse;
    uint8_t yield;
    uint8_t stall;
    uint8_t wrtdb;
    uint8_t readb;
    uint8_t watdb;
};

static control_flags_t ctrl[3];

// watdb:readb:wrtdb:yield:stall [reuse]
// read and write barriers are numbered 1...6
void wait_on_barrier(uint8_t op, uint8_t barrier_number) {
    ctrl[op].watdb |= (1 << (barrier_number-1));
}
void set_write_barrier(uint8_t op, uint8_t barrier_number) {
    ctrl[op].wrtdb = barrier_number-1;
}
void set_read_barrier(uint8_t op, uint8_t barrier_number) {
    ctrl[op].readb = barrier_number-1;
}
void yield(uint8_t op) { // enables yield on instruction number op
    ctrl[op].yield = 0; // zero means enable
}
void stall(uint8_t op, uint8_t count) {
    ctrl[op].stall = count;
}
void reuse(uint8_t op, bool ra, bool rb, bool rc, bool rd) {
    ctrl[op].reuse = 0;
    if (ra) ctrl[op].reuse |= 0x1;
    if (rb) ctrl[op].reuse |= 0x2;
    if (rc) ctrl[op].reuse |= 0x4;
    if (rd) ctrl[op].reuse |= 0x8;
}
void reset_ctrl() {
    for (int op = 0; op < 3; op++)
    {
        ctrl[op].watdb = 0x00;
        ctrl[op].readb = 7;
        ctrl[op].wrtdb = 7;
        ctrl[op].yield = 1;
        ctrl[op].stall = 0;
    }
}
uint64_t CTRL() {
    uint64_t ret = 0;
    for (int op = 0; op < 3; op++) {
        uint64_t stall = (((uint64_t)ctrl[op].stall) & 0x0f) << 0;
        uint64_t yield = (((uint64_t)ctrl[op].yield) & 0x01) << 4;
        uint64_t wrtdb = (((uint64_t)ctrl[op].wrtdb) & 0x07) << 5;
        uint64_t readb = (((uint64_t)ctrl[op].readb) & 0x07) << 8;
        uint64_t watdb = (((uint64_t)ctrl[op].watdb) & 0x3f) << 11;
        uint64_t reuse = (((uint64_t)ctrl[op].reuse) & 0x0f) << 17;
        uint64_t ctrl = reuse|watdb|readb|wrtdb|yield|stall;
        ret |= ctrl << (op*21);
    }
    return ret;
}

void print_ctrl_segment(uint64_t x) {
    uint8_t stall = (uint8_t)((x & 0x0000f) >> 0);
    uint8_t yield = (uint8_t)((x & 0x00010) >> 4);
    uint8_t wrtdb = (uint8_t)((x & 0x000e0) >> 5); // 7 = no dependency
    uint8_t readb = (uint8_t)((x & 0x00700) >> 8); // 7 = no dependency
    uint8_t watdb = (uint8_t)((x & 0x1f800) >> 11);
    if (watdb)    printf("%02x:", watdb); else printf("--:");
    if (readb==7) printf("-:");           else printf("%d:", readb+1);
    if (wrtdb==7) printf("-:");           else printf("%d:", wrtdb+1);
    if (yield)    printf("-:");           else printf("Y:");
    printf("%x", stall);
}

void print_ctrl(uint64_t x) {
    uint64_t ctrl1 =  (x & 0x000000000001ffff) >> 0;
    uint64_t ctrl2 =  (x & 0x0000003fffe00000) >> 21;
    uint64_t ctrl3 =  (x & 0x07fffc0000000000) >> 42;
    uint64_t reuse1 = (x & 0x00000000001e0000) >> 17;
    uint64_t reuse2 = (x & 0x000003c000000000) >> 38;
    uint64_t reuse3 = (x & 0x7800000000000000) >> 59;
    print_ctrl_segment(ctrl1); printf(" | ");
    print_ctrl_segment(ctrl2); printf(" | ");
    print_ctrl_segment(ctrl3);
}

}

/*
Notes

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                               IMMEDIATE VALUES

FADD20I, FMUL20I and FFMA20I are immediate versions of their respective instructions,
except the rightmost 12 bits of the single-precision mantissa are masked to zero. If
you need full 23-bit mantissa precision you can use FADD32I and FMUL32I, which encode
the entire float. FFMA does not have a 32-bit immediate version, but it can load from
constant memory.

*20I appear to be treated the same (flag-wise) as their non-immediate counterparts.

FMNMX d, a, b, !PT -> MAX(a,b)
FMNMX d, a, b, PT  -> MIN(a,b)

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                REGISTER BANKS

Maxwell has four register banks per thread. The assignment of registers to banks is easy:
  Bank = Register number mod 4 (e.g. R0 and R4 are bank0, R3 and R7 are bank3)
On Maxwell and Pascal, instructions can only access one value from each memory bank?

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                REGISTER REUSE

Maxwell and Pascal have 4 register reuse caches and 4 source operand slots. Each of the
4 reuse flag bits correspond to one of the 8-byte slots. The LSB in reuse flags controls
the cache for the first source operand slot (a?), while the MSB is for the fourth.
e.g. instruction dst, op0 ("first"), op1, op2, op3 ("last")
e.g. FFMA.FTZ R3, R4, R4, R0.reuse -> has reuse flag 0100
e.g. FFMA.FTZ R3, R4.reuse, R4, R0 -> has reuse flag 0001
*/

