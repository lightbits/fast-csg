#pragma once

namespace backend_sass {

enum latency_constants_
{
    // All the 32-bit floating point instructions (except sqrt) take exactly
    // 6 cycles before the result is written to and valid. Subsequent instructions
    // that read from this result must therefore be executed atleast six cycles
    // after the first one began. The scheduler tries to fill the gap between one
    // instruction and one that depends on its results by looking for others that
    // do not depend on its results. We conveniently structure our input code into
    // 'blocks' that are entirely independent from other blocks, but the instructions
    // within a block cannot be reordered. If the scheduler can't find enough
    // instructions to fill the pipeline, it will have to insert 'stalls', which
    // do nothing for a given number of clock cycles.
    LATENCY_X32T  = 6,

    // sqrt is a variable latency instruction and needs to set a write barrier
    // which dependent instructions must wait on. The later that instruction
    // actually does the wait, the more likely it is that the sqrt is finished,
    // and the barrier does not incur a stall. We work under the assumption that
    // sqrt finishes after 'LATENCY_SQRT' cycles.
    LATENCY_SQRT  = 8,

    // Setting the write barrier takes non-zero clock cycles.
    LATENCY_WRTDB = 1,
};

enum instruction_type_t
{
    INSTRUCTION_FFMA=0,
    INSTRUCTION_FMUL,
    INSTRUCTION_FADD,
    INSTRUCTION_FFMA20I,
    INSTRUCTION_FMUL20I,
    INSTRUCTION_FADD20I,
    INSTRUCTION_FADD20I_ABS_A,
    INSTRUCTION_FMIN,
    INSTRUCTION_FMAX,
    INSTRUCTION_FMAX_NEG_B,
    INSTRUCTION_SQRT
};

struct instruction_t
{
    instruction_type_t type;
    named_register_t a,b,c; // source registers ("operands")
    named_register_t d;     // destination register
    float imm_b;            // immediate value in b-slot

    // filled in by scheduler
    uint8_t ra,rb,rc,rd;
    uint8_t reuse;       // register reuse flags
    uint8_t yield;       // can relinquish control to other warp or not
    uint8_t stall;       // number of cycles to wait before continuing
    uint8_t wrtdb;       // write dependencies
    uint8_t readb;       // read dependencies
    uint8_t watdb;       // wait dependencies
};

enum { MAX_INSTRUCTIONS_PER_BLOCK = 64 };
struct instruction_block_t
// An instruction block is a list of instructions that implements a single basic
// AST opcode, either a primitive or an operator. During code generation (parsing
// the AST), we create a list of instruction blocks, evaluating the AST bottom-up.
// During this, we assign to each block up to three register addresses.
// A destination register, where the output of the block is to be stored, and
// a left- and right-child register (for boolean operators).
{
    instruction_t instructions[MAX_INSTRUCTIONS_PER_BLOCK];
    int num_instructions;
    int d,d_left,d_right;
};

enum { MAX_INSTRUCTION_BLOCKS = 128 };
struct instruction_blocks_t
{
    instruction_block_t *blocks;
    int num_blocks;
};

void print_instruction(instruction_t in)
{
    int n = 0;
         if (in.type==INSTRUCTION_FFMA)          n+=printf("FFMA r%d,  r%d , r%d, r%d", in.rd, in.ra, in.rb, in.rc);
    else if (in.type==INSTRUCTION_FMUL)          n+=printf("FMUL r%d,  r%d , r%d", in.rd, in.ra, in.rb);
    else if (in.type==INSTRUCTION_FADD)          n+=printf("FADD r%d,  r%d , r%d", in.rd, in.ra, in.rb);
    else if (in.type==INSTRUCTION_FFMA20I)       n+=printf("FFMA r%d,  r%d , %5.2ff, r%d", in.rd, in.ra, in.imm_b, in.rc);
    else if (in.type==INSTRUCTION_FMUL20I)       n+=printf("FMUL r%d,  r%d , %5.2ff", in.rd, in.ra, in.imm_b);
    else if (in.type==INSTRUCTION_FADD20I)       n+=printf("FADD r%d,  r%d , %5.2ff", in.rd, in.ra, in.imm_b);
    else if (in.type==INSTRUCTION_FADD20I_ABS_A) n+=printf("FADD r%d, |r%d|, %5.2ff", in.rd, in.ra, in.imm_b);
    else if (in.type==INSTRUCTION_FMIN)          n+=printf("FMIN r%d,  r%d , r%d", in.rd, in.ra, in.rb);
    else if (in.type==INSTRUCTION_FMAX)          n+=printf("FMAX r%d,  r%d , r%d", in.rd, in.ra, in.rb);
    else if (in.type==INSTRUCTION_FMAX_NEG_B)    n+=printf("FMAX r%d, -r%d , r%d", in.rd, in.ra, in.rb);
    else if (in.type==INSTRUCTION_SQRT)          n+=printf("SQRT r%d,  r%d", in.rd, in.ra);
    else assert(false);

    for (int i = n; i < 30; i++)
        printf(" ");

    if (in.watdb)    printf("%02x:", in.watdb); else printf("--:");
    if (in.readb==7) printf("-:");              else printf("%d:", in.readb+1);
    if (in.wrtdb==7) printf("-:");              else printf("%d:", in.wrtdb+1);
    if (in.yield)    printf("-:");              else printf("Y:");
    printf("%x", in.stall);
    if (in.reuse)
        printf(" reuse: %s%s%s",
            (in.reuse & 1) ? "a" : " ",
            (in.reuse & 2) ? "b" : " ",
            (in.reuse & 4) ? "c" : " ");
    printf("\n");
}

}
