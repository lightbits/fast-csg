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
#include "../frep.h"
#include <stdint.h>
#include <assert.h>
#include <stdio.h>
#include <memory.h>
#include "registers.h"
#include "instruction.h"
#include "scheduler.h"
#include "blocks.h"
#include "bytecode.h"

#if 0
void *frep_compile_to_sass(frep_t *tree, size_t *length)
{
    using namespace backend_sass;
    instruction_blocks_t blocks = generate_sass_blocks(tree);

    int num_instructions;
    instruction_t *instructions = schedule_blocks(blocks, &num_instructions);
}
#endif
