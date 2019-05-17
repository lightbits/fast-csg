// Developed by Simen Haugo.
// See LICENSE.txt for copyright and licensing details (standard MIT License).
//
// This file contains the machine code generation backend for NVIDIA SASS (Shader
// Assembly) ISA. Unlike the PTX backend, this directly outputs to binary code that
// can be patched into a Cubin binary module and loaded immediately with the Cuda
// Driver API (see NVRTC example in SDK). This avoids the slow PTX compiler provided
// in CUDA.
//
// This backend is for devices of compute capability 6.x, such as the Maxwell and
// Pascal GPU families. It does not support Volta or Turing families (which have
// compute capability 7.x).
//
// SASS code generation consists of the following major steps
//
// 1. Generate instruction blocks
//      the input frep tree is parsed to produce independent sequences of temporary
//      SASS instructions (not binary). These are assigned virtual register names,
//      which must be assigned to physical registers in the next step.
//
// 2. Schedule instructions and assign physical registers
//
// 3. Generate SASS binary
//      With the physical registers assigned, we can now generate the actual binary
//      instructions that go into the final ELF executable.
//
// 4. Link SASS ELF executable (a "Cubin" module)
//

#pragma once
#include "frep.h"
#include <stdint.h>
#include <assert.h>

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

enum named_register_t
{
    // used to indicate immediate values
    NO_REGISTER=0,

    // input position is stored in these
    REGISTER_X0=0,
    REGISTER_Y0,
    REGISTER_Z0,

    // an instruction block is allocated these by the scheduler
    REGISTER_X,
    REGISTER_Y,
    REGISTER_Z,
    REGISTER_W,
    REGISTER_D,       // result is to be stored here
    REGISTER_D_LEFT,  // result from left child in SDF tree is stored here
    REGISTER_D_RIGHT, // result from right child in SDF tree is stored here

    // constant zero
    REGISTER_RZ=0xff
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

enum { max_instructions_per_block = 64 };
struct instruction_block_t
// An instruction block is a list of instructions that implements a single basic
// AST opcode, either a primitive or an operator. During code generation (parsing
// the AST), we create a list of instruction blocks, evaluating the AST bottom-up.
// During this, we assign to each block up to three register addresses.
// A destination register, where the output of the block is to be stored, and
// a left- and right-child register (for boolean operators).
{
    instruction_t instructions[max_instructions_per_block];
    int num_instructions;
    int d,d_left,d_right;
};

enum { max_instruction_blocks = 128 };
struct instruction_blocks_t
{
    instruction_block_t *blocks;
    int num_blocks;
    int registers_used;
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

instruction_t *
schedule_blocks(instruction_blocks_t blocks, int *return_num_instructions)
// This function performs physical register allocation and instruction scheduling.
// Register allocation maps the virtual register names used by each instruction to
// physical register addresses (0 to 255). Instruction scheduling makes sure that
// enough clock cycles passes between instructions so that the results are ready.
{
    enum { max_instructions = 1024 };
    static instruction_t out[max_instructions];
    int num_out = 0;

    enum { max_registers = 256 };
    enum { num_wait_barriers = 6 };
    enum { max_temp_registers = 24 };

    struct wait_barrier_t
    {
        uint8_t barrier_on_register[max_registers];
        bool is_barrier_active[num_wait_barriers];
        void init()
        {
            for (int i = 0; i < num_wait_barriers; i++)
                is_barrier_active[i] = false;
            for (int i = 0; i < max_registers; i++)
                barrier_on_register[i] = 7;
        }
        bool is_set(uint8_t reg) { return barrier_on_register[reg] != 7; }
        uint8_t set(uint8_t reg) // return wrtdb flag
        {
            for (int i = 0; i < num_wait_barriers; i++)
            {
                if (!is_barrier_active[i])
                {
                    uint8_t barrier = (uint8_t)(i);
                    barrier_on_register[reg] = barrier;
                    is_barrier_active[i] = true;
                    return barrier;
                }
            }
            assert(false && "Ran out of wait barriers");
            return 7;
        }
        uint8_t wait(uint8_t reg) // return watdb flag (to be OR'd with current flag)
        {
            uint8_t barrier = barrier_on_register[reg];
            assert(barrier != 7 && "Tried to wait on a register that had no wait barrier set.");
            uint8_t watdb = 1 << barrier;
            is_barrier_active[barrier] = false;
            barrier_on_register[reg] = 7;
            return watdb;
        }
    };

    static wait_barrier_t wait_barrier;
    wait_barrier.init();

    for (int i = 0; i < blocks.num_blocks; i++)
    {
        int d = blocks.blocks[i].d;
        assert(d < max_temp_registers);
        int d_left = blocks.blocks[i].d_left;
        int d_right = blocks.blocks[i].d_right;

        static uint8_t register_map[256] = {0};
        // register_map[NO_REGISTER]      =
        register_map[REGISTER_X0]      = 0x00;
        register_map[REGISTER_Y0]      = 0x01;
        register_map[REGISTER_Z0]      = 0x02;
        register_map[REGISTER_X]       = 0x03;
        register_map[REGISTER_Y]       = 0x04;
        register_map[REGISTER_Z]       = 0x05;
        register_map[REGISTER_W]       = 0x06;
        register_map[REGISTER_D]       = 0x07 + d;
        register_map[REGISTER_D_LEFT]  = 0x07 + d_left;
        register_map[REGISTER_D_RIGHT] = 0x07 + d_right;
        register_map[REGISTER_RZ]      = 0xff;

        for (int j = 0; j < blocks.blocks[i].num_instructions; j++)
        {
            instruction_t *in = &blocks.blocks[i].instructions[j];
            in->ra = register_map[in->a];
            in->rb = register_map[in->b];
            in->rc = register_map[in->c];
            in->rd = register_map[in->d];
            in->reuse = 0;
            in->watdb = 0;
            in->readb = 7;
            in->wrtdb = 7;
            in->yield = 0;
            if (wait_barrier.is_set(in->ra)) { in->watdb |= wait_barrier.wait(in->ra); }
            if (wait_barrier.is_set(in->rb)) { in->watdb |= wait_barrier.wait(in->rb); }
            if (wait_barrier.is_set(in->rc)) { in->watdb |= wait_barrier.wait(in->rc); }

            // if we the instruction doesn't have a stall count set already
            // we set it to the latency of the instruction.
            if (in->stall == 0)
            {
                if (in->type == INSTRUCTION_SQRT) in->stall = 1+LATENCY_WRTDB;
                else                              in->stall = LATENCY_X32T;
            }

            if (in->type == INSTRUCTION_SQRT) in->wrtdb = wait_barrier.set(in->rd);

            // simple reuse tactic
            #if 1
            if (j > 0)
            {
                instruction_t *last = &blocks.blocks[i].instructions[j-1];
                if (last->a != NO_REGISTER && last->ra == in->ra && last->rd != in->ra) in->reuse |= 1 << 0;
                if (last->b != NO_REGISTER && last->rb == in->rb && last->rd != in->rb) in->reuse |= 1 << 1;
                if (last->c != NO_REGISTER && last->rc == in->rc && last->rd != in->rc) in->reuse |= 1 << 2;
            }
            #endif

            out[num_out++] = *in;
            assert(num_out <= max_instructions);
        }
    }

    *return_num_instructions = num_out;
    return out;
}

#include <memory.h>
#define CLEAR()           memset(&block->instructions[block->num_instructions], 0, sizeof(instruction_t))
#define TYPE(Expression)  block->instructions[block->num_instructions].type = INSTRUCTION_##Expression
#define RA(Expression)    block->instructions[block->num_instructions].a = REGISTER_##Expression
#define RB(Expression)    block->instructions[block->num_instructions].b = REGISTER_##Expression
#define RC(Expression)    block->instructions[block->num_instructions].c = REGISTER_##Expression
#define RD(Expression)    block->instructions[block->num_instructions].d = REGISTER_##Expression
#define STALL(Expression) block->instructions[block->num_instructions].stall = Expression;
#define IMMB(Expression)  block->instructions[block->num_instructions].imm_b = Expression;
#define NEXT()            block->num_instructions++; assert(block->num_instructions <= max_instructions_per_block);

#if 0 // sequential transform code
// (x,y,z) = R_root_to_this*((x0,y0,z0) - T_this_rel_root)
//         = Rz(rz)*Ry(ry)*Rx(rx)*((x0-tx, y0-ty, z0-tz))
void emit_transform(instruction_block_t *block, frep_mat3_t R_root_to_this, frep_vec3_t T_this_rel_root)
{
    // Convert to final rotation into euler angles
    // (need less registers to do three sequential
    // euler rotations, than a full 3x3 matrix multiply, I think...?)
    float rx,ry,rz;
    frep_so3_to_ypr(R_root_to_this, &rz, &ry, &rx);
    float tx = T_this_rel_root[0];
    float ty = T_this_rel_root[1];
    float tz = T_this_rel_root[2];
    float cx = cosf(rx); float sx = sinf(rx);
    float cy = cosf(ry); float sy = sinf(ry);
    float cz = cosf(rz); float sz = sinf(rz);
                                                                     // translate:
    CLEAR(); TYPE(FADD20I); RD(X); RA(X0); IMMB(-tx);        NEXT(); // FADD x, x0, (-tx)
    CLEAR(); TYPE(FADD20I); RD(Y); RA(Y0); IMMB(-ty);        NEXT(); // FADD y, y0, (-ty)
    CLEAR(); TYPE(FADD20I); RD(Z); RA(Z0); IMMB(-tz);        NEXT(); // FADD z, z0, (-tz)
                                                                     // rotate_x: x=x, y=c*y - s*z, z=s*y + c*z
    CLEAR(); TYPE(FMUL20I); RD(W); RA(Y);  IMMB(+sx);        NEXT(); // FMUL w, y, (s)
    CLEAR(); TYPE(FMUL20I); RD(Y); RA(Y);  IMMB(+cx);        NEXT(); // FMUL y, y.reuse, (c)
    CLEAR(); TYPE(FFMA20I); RD(Y); RA(Z);  IMMB(-sx); RC(Y); NEXT(); // FFMA y, z, (-s), y
    CLEAR(); TYPE(FFMA20I); RD(Z); RA(Z);  IMMB(+cx); RC(W); NEXT(); // FFMA z, z.reuse, (c), w
                                                                     // rotate_y: x=c*x + s*z, y=y, z=-s*x + c*z
    CLEAR(); TYPE(FMUL20I); RD(W); RA(X);  IMMB(-sy);        NEXT(); // FMUL w, x, (-s)
    CLEAR(); TYPE(FMUL20I); RD(X); RA(X);  IMMB(+cy);        NEXT(); // FMUL x, x.reuse, (c)
    CLEAR(); TYPE(FFMA20I); RD(X); RA(Z);  IMMB(+sy); RC(X); NEXT(); // FFMA x, z, (s), x
    CLEAR(); TYPE(FFMA20I); RD(Z); RA(Z);  IMMB(+cy); RC(W); NEXT(); // FFMA z, z.reuse, (c), w
                                                                     // rotate_z: x=c*x - s*y, y=s*x + c*y, z=z
    CLEAR(); TYPE(FMUL20I); RD(W); RA(X);  IMMB(+sz);        NEXT(); // FMUL w, x, (s)
    CLEAR(); TYPE(FMUL20I); RD(X); RA(X);  IMMB(+cz);        NEXT(); // FMUL x, x.reuse, (c)
    CLEAR(); TYPE(FFMA20I); RD(X); RA(Y);  IMMB(-sz); RC(X); NEXT(); // FFMA x, y, (-s), x
    CLEAR(); TYPE(FFMA20I); RD(Y); RA(Y);  IMMB(+cz); RC(W); NEXT(); // FFMA y, y.reuse, (c), w
}
#else
void emit_transform(instruction_block_t *block, frep_mat3_t R/*_root_to_this*/, frep_vec3_t T/*_this_rel_root*/)
{
    // This path is a stall-count optimized version of the above.
    // The generated code computes the following:
    // (x,y,z) = R_root_to_this*((x0,y0,z0) - T_this_rel_root)
    // x = R00*(x0-Tx) + R01*(y0-Ty) + R02*(z0-Tz)
    //   = R00*x0 + R01*y0 + R02*z0 + (-R00*Tx - R01*Ty - R02*Tz)
    //   = R00*x0 + R01*y0 + R02*z0 + dx
    // etc...

    float dx = -(R.at(0,0)*T[0] + R.at(0,1)*T[1] + R.at(0,2)*T[2]);
    float dy = -(R.at(1,0)*T[0] + R.at(1,1)*T[1] + R.at(1,2)*T[2]);
    float dz = -(R.at(2,0)*T[0] + R.at(2,1)*T[1] + R.at(2,2)*T[2]);

    CLEAR(); TYPE(FADD20I); RD(X); RA(RZ); IMMB(dx);               STALL(1); NEXT(); // 1 FADD x, RZ, dx
    CLEAR(); TYPE(FADD20I); RD(Y); RA(RZ); IMMB(dy);               STALL(1); NEXT(); // 1 FADD y, RZ, dy
    CLEAR(); TYPE(FADD20I); RD(Z); RA(RZ); IMMB(dz);               STALL(4); NEXT(); // 4 FADD z, RZ, dz
    CLEAR(); TYPE(FFMA20I); RD(X); RA(X0); IMMB(R.at(0,0)); RC(X); STALL(1); NEXT(); // 1 FFMA x, x0, (R00), x // Q) Why not have dx here?
    CLEAR(); TYPE(FFMA20I); RD(Y); RA(X0); IMMB(R.at(1,0)); RC(Y); STALL(1); NEXT(); // 1 FFMA y, x0, (R10), y
    CLEAR(); TYPE(FFMA20I); RD(Z); RA(X0); IMMB(R.at(2,0)); RC(Z); STALL(4); NEXT(); // 4 FFMA z, x0, (R20), z
    CLEAR(); TYPE(FFMA20I); RD(X); RA(Y0); IMMB(R.at(0,1)); RC(X); STALL(1); NEXT(); // 1 FFMA x, y0, (R01), x
    CLEAR(); TYPE(FFMA20I); RD(Y); RA(Y0); IMMB(R.at(1,1)); RC(Y); STALL(1); NEXT(); // 1 FFMA y, y0, (R11), y
    CLEAR(); TYPE(FFMA20I); RD(Z); RA(Y0); IMMB(R.at(2,1)); RC(Z); STALL(4); NEXT(); // 4 FFMA z, y0, (R21), z
    CLEAR(); TYPE(FFMA20I); RD(X); RA(Z0); IMMB(R.at(0,2)); RC(X); STALL(1); NEXT(); // 1 FFMA x, z0, (R02), x
    CLEAR(); TYPE(FFMA20I); RD(Y); RA(Z0); IMMB(R.at(1,2)); RC(Y); STALL(1); NEXT(); // 1 FFMA y, z0, (R12), y
    CLEAR(); TYPE(FFMA20I); RD(Z); RA(Z0); IMMB(R.at(2,2)); RC(Z); STALL(4); NEXT(); // 4 FFMA z, z0, (R22), z
}
#endif

// cylinder: max(sqrt(x*x + z*z) - R, abs(y)-H)
void emit_cylinder(instruction_block_t *block, frep_mat3_t R, frep_vec3_t T, float r, float h)
{
    emit_transform(block, R, T);
    CLEAR(); TYPE(FMUL);          RD(W); RA(X); RB(X);        NEXT(); // FMUL w, x, x
    CLEAR(); TYPE(FFMA);          RD(W); RA(Z); RB(Z); RC(W); NEXT(); // FFMA w, z, z, w
    CLEAR(); TYPE(SQRT);          RD(W); RA(W); RB(W);        NEXT(); // SQRT w, w
    CLEAR(); TYPE(FADD20I_ABS_A); RD(Y); RA(Y); IMMB(-h);     NEXT(); // FADD y, |y|, -H
    CLEAR(); TYPE(FADD20I);       RD(W); RA(W); IMMB(-r);     NEXT(); // FADD w, w, -R
    CLEAR(); TYPE(FMAX);          RD(D); RA(W); RB(Y);        NEXT(); // FMAX d, w, y
}

// sphere: sqrt(x*x + y*y + z*z) - R
void emit_sphere(instruction_block_t *block, frep_mat3_t R, frep_vec3_t T, float r)
{
    #if 1
    CLEAR(); TYPE(FADD20I); RD(X); RA(X0); IMMB(-T[0]);  STALL(1); NEXT(); // 1 FADD x, x0, (-tx)
    CLEAR(); TYPE(FADD20I); RD(Y); RA(Y0); IMMB(-T[1]);  STALL(1); NEXT(); // 1 FADD y, y0, (-ty)
    CLEAR(); TYPE(FADD20I); RD(Z); RA(Z0); IMMB(-T[2]);  STALL(4); NEXT(); // 4 FADD z, z0, (-tz)
    CLEAR(); TYPE(FMUL);    RD(W); RA(X); RB(X);                   NEXT(); // 6 FMUL w, x, x
    CLEAR(); TYPE(FFMA);    RD(W); RA(Y); RB(Y); RC(W);            NEXT(); // 6 FFMA w, y, y, w
    CLEAR(); TYPE(FFMA);    RD(W); RA(Z); RB(Z); RC(W);            NEXT(); // 6 FFMA w, z, z, w
    CLEAR(); TYPE(SQRT);    RD(W); RA(W); RB(W);                   NEXT(); // 8 SQRT w, w
    CLEAR(); TYPE(FADD20I); RD(D); RA(W); IMMB(-r);                NEXT(); // 6 FADD d, w, -R
    #else
    emit_transform(block, R, T);
    CLEAR(); TYPE(FMUL);    RD(W); RA(X); RB(X);        NEXT(); // FMUL w, x, x
    CLEAR(); TYPE(FFMA);    RD(W); RA(Y); RB(Y); RC(W); NEXT(); // FFMA w, y, y, w
    CLEAR(); TYPE(FFMA);    RD(W); RA(Z); RB(Z); RC(W); NEXT(); // FFMA w, z, z, w
    CLEAR(); TYPE(SQRT);    RD(W); RA(W); RB(W);        NEXT(); // SQRT w, w
    CLEAR(); TYPE(FADD20I); RD(D); RA(W); IMMB(-r);     NEXT(); // FADD d, w, -R
    #endif
}

void emit_box(instruction_block_t *block, frep_mat3_t R, frep_vec3_t T, float bx, float by, float bz)
{
    assert(false && "fBox is not implemented yet");
}

// box: max(max(|x|-wx, |y|-wy), |z|-wz)
void emit_box_cheap(instruction_block_t *block, frep_mat3_t R, frep_vec3_t T, float bx, float by, float bz)
{
    emit_transform(block, R, T);
    CLEAR(); TYPE(FADD20I_ABS_A); RD(X); RA(X); IMMB(-bx); STALL(1); NEXT(); // 1 FADD x, |x|, -wx
    CLEAR(); TYPE(FADD20I_ABS_A); RD(Y); RA(Y); IMMB(-by); STALL(1); NEXT(); // 1 FADD y, |y|, -wy
    CLEAR(); TYPE(FADD20I_ABS_A); RD(Z); RA(Z); IMMB(-bz); STALL(5); NEXT(); // 5 FADD z, |z|, -wz
    CLEAR(); TYPE(FMAX);          RD(W); RA(X); RB(Y);               NEXT(); // 6 FMAX w, x, y
    CLEAR(); TYPE(FMAX);          RD(D); RA(W); RB(Z);               NEXT(); // 6 FMAX d, w, z
}

void emit_plane(instruction_block_t *block, frep_mat3_t R, frep_vec3_t T, float px)
{
    #if 0
    // optimized version
    float rx,ry,rz;
    frep_so3_to_ypr(R, &rz, &ry, &rx);
    float cx = cosf(rx); float sx = sinf(rx);
    float cy = cosf(ry); float sy = sinf(ry);
    float cz = cosf(rz); float sz = sinf(rz);
    float rtx = -((cy*cz)*T[0] + (cz*sx*sy - cx*sz)*T[1] + (sx*sz + cx*cz*sy)*T[2]);

    CLEAR(); TYPE(FMUL20I); RD(X); RA(X0); IMMB((cy*cz));                   NEXT(); // 6 FMUL x, x0, (cy*cz)
    CLEAR(); TYPE(FFMA20I); RD(X); RA(Y0); IMMB((cz*sx*sy-cx*sz));   RC(X); NEXT(); // 6 FFMA x, y0, (cz*sx*sy-cx*sz), x
    CLEAR(); TYPE(FFMA20I); RD(X); RA(Z0); IMMB((sx*sz + cx*cz*sy)); RC(X); NEXT(); // 6 FFMA x, z0, (sx*sz + cx*cz*sy), x
    CLEAR(); TYPE(FADD20I); RD(D); RA(X);  IMMB(rtx-px);                    NEXT(); // 6 FADD d, x, rtx-px
    #else
    emit_transform(block, R, T);
                                                    // plane: x - px
    CLEAR(); TYPE(FADD20I); RD(D); RA(X); IMMB(-px); NEXT(); // FADD d, x, -px
    #endif
}

void emit_union(instruction_block_t *block)     { CLEAR(); TYPE(FMIN); RD(D); RA(D_LEFT); RB(D_RIGHT);       NEXT(); }
void emit_intersect(instruction_block_t *block) { CLEAR(); TYPE(FMAX); RD(D); RA(D_LEFT); RB(D_RIGHT);       NEXT(); }
void emit_subtract(instruction_block_t *block)  { CLEAR(); TYPE(FMAX_NEG_B); RD(D); RA(D_LEFT); RB(D_RIGHT); NEXT(); }
void emit_blend(instruction_block_t *block, float alpha)
{
                                                                        // blend: alpha*d_left + (1-alpha)*d_right
    CLEAR(); TYPE(FMUL20I); RD(D); RA(D_LEFT);  IMMB(alpha);             NEXT(); // FMUL d, d_left, (alpha)
    CLEAR(); TYPE(FFMA20I); RD(D); RA(D_RIGHT); IMMB(1.0f-alpha); RC(D); NEXT(); // FFMA d, d_right, (1-alpha), d
}

#undef TYPE
#undef RA
#undef RB
#undef RC
#undef RD
#undef IMMB
#undef NEXT
#undef STALL
#undef CLEAR

void
_generate_sass_blocks(instruction_block_t *blocks,
               int &num_blocks,
               frep_t *node,
               int destination,
               frep_mat3_t R_root_to_parent=frep_identity_3x3,
               frep_vec3_t T_parent_rel_root=frep_null_3x1)
// You can do much smarter register allocation here. The register allocation
// may also need to change if we do smarter scheduling. E.g. block reordering.
{
    assert(node);

    frep_mat3_t R_root_to_this;
    frep_vec3_t T_this_rel_root;
    frep_get_global_transform(node, &R_root_to_this, &T_this_rel_root, R_root_to_parent, T_parent_rel_root);

    if (frep_is_boolean(node))
    {
        assert(node->left);
        assert(node->right);

        int d_left = destination;
        int d_right = destination+1;
        _generate_sass_blocks(blocks, num_blocks, node->left, d_left, R_root_to_this, T_this_rel_root);
        _generate_sass_blocks(blocks, num_blocks, node->right, d_right, R_root_to_this, T_this_rel_root);

        instruction_block_t *b = &blocks[num_blocks++];
        b->num_instructions = 0;
        b->d_left = d_left;
        b->d_right = d_right;
        b->d = destination;
             if (node->opcode == FREP_UNION)     emit_union(b);
        else if (node->opcode == FREP_INTERSECT) emit_intersect(b);
        else if (node->opcode == FREP_SUBTRACT)  emit_subtract(b);
        else if (node->opcode == FREP_BLEND)     emit_blend(b, node->blend.alpha);
        assert(num_blocks <= max_instruction_blocks);
    }
    else if (frep_is_primitive(node))
    {
        instruction_block_t *b = &blocks[num_blocks++];
        b->num_instructions = 0;
        frep_mat3_t R = R_root_to_this;
        frep_vec3_t T = T_this_rel_root;
        b->d = destination;
             if (node->opcode == FREP_BOX)       emit_box(b, R, T, node->box.width, node->box.height, node->box.depth);
        else if (node->opcode == FREP_BOX_CHEAP) emit_box_cheap(b, R, T, node->box.width, node->box.height, node->box.depth);
        else if (node->opcode == FREP_SPHERE)    emit_sphere(b, R, T, node->sphere.radius);
        else if (node->opcode == FREP_CYLINDER)  emit_cylinder(b, R, T, node->cylinder.radius, node->cylinder.height);
        else if (node->opcode == FREP_PLANE)     emit_plane(b, R, T, node->plane.offset);
        assert(num_blocks <= max_instruction_blocks);
    }
    else
    {
        assert(false && "Unexpected node type");
    }
}

instruction_blocks_t generate_sass_blocks(frep_t *node)
// This function generates a list of instruction blocks that evaluates the
// tree and stores the resulting distance value in register[0]. Each block
// is assigned registers during the recursive tree parsing.
{
    assert(node);

    static instruction_block_t blocks[max_instruction_blocks];
    int num_blocks = 0;
    int destination = 0;

    _generate_sass_blocks(blocks, num_blocks, node, destination);

    instruction_blocks_t result;
    result.blocks = blocks;
    result.num_blocks = num_blocks;
    return result;
}
