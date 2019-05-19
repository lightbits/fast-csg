#pragma once

namespace backend_sass {

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

}
