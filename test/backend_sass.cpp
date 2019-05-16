#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "../src/frep.h"
#include "../src/frep_builder.h"
#include "../src/frep_eval.h"
#include "../src/backend_sass.h"

struct pas_simulator_t
{
    bool debug;
    int t;
    float reg[256];

    // writes in progress
    struct job_t
    {
        uint8_t dst;
        float val;
        int t_write;
    };
    enum { max_write_jobs = 1024 };
    job_t writes[max_write_jobs];
    int num_writes_waiting;

    // barriers
    enum { num_write_barriers = 6 };
    int register_on_barrier[num_write_barriers];

    void init(bool _debug)
    {
        reg[REGISTER_RZ] = 0.0f;
        num_writes_waiting = 0;
        t = 0;
        debug = _debug;
        for (int i = 0; i < num_write_barriers; i++)
            register_on_barrier[i] = -1;
    }
    void _step(int cycles)
    {
        t += cycles;
        for (int i = 0; i < num_writes_waiting; i++)
        {
            if (t >= writes[i].t_write)
            {
                reg[writes[i].dst] = writes[i].val;

                // if a write barrier was set on the register we can take it down
                for (int j = 0; j < 6; j++)
                {
                    if (register_on_barrier[j] == writes[i].dst)
                        register_on_barrier[j] = -1;
                }

                writes[i] = writes[--num_writes_waiting];
                i--;
            }
        }
    }
    void _set_write_barrier(uint8_t reg, uint8_t barrier)
    {
        assert(barrier >= 0 && barrier <= num_write_barriers-1);
        assert(register_on_barrier[barrier] == -1 && "overwrote an existing write barrier.");
        register_on_barrier[barrier] = reg;
    }
    void _wait_on_barrier(uint8_t barrier)
    {
        if (register_on_barrier[barrier] == -1)
            return;
        assert(barrier >= 0 && barrier <= num_write_barriers-1);
        bool resolved = false;
        for (int i = 0; i < num_writes_waiting; i++)
        {
            if (writes[i].dst == (uint8_t)register_on_barrier[barrier])
            {
                int t_to_wait = writes[i].t_write - t;
                if (t_to_wait > 0)
                {
                    if (debug) printf("waited %d cycles on barrier\n", t_to_wait);
                    _step(t_to_wait);
                }
                resolved = true;
                register_on_barrier[barrier] = -1;
            }
        }
        assert(resolved && "waited on a barrier which is not resolved by any on-going writes.");
    }
    float _read_reg(uint8_t src)
    {
        for (int i = 0; i < num_writes_waiting; i++)
            if (writes[i].dst == src && debug)
                printf("read-before-write conflict on r%d\n", src);
        return reg[src];
    }
    void _write_reg(uint8_t dst, float val, int latency)
    {
        assert(num_writes_waiting+1 <= max_write_jobs);
        writes[num_writes_waiting].dst = dst;
        writes[num_writes_waiting].val = val;
        writes[num_writes_waiting].t_write = t + latency;
        num_writes_waiting++;
    }
    void execute(instruction_t in)
    {
        bool is_immediate =
            in.type == INSTRUCTION_FFMA20I ||
            in.type == INSTRUCTION_FMUL20I ||
            in.type == INSTRUCTION_FADD20I ||
            in.type == INSTRUCTION_FADD20I_ABS_A;

        if (in.watdb)
        {
            if (in.watdb & 1)  _wait_on_barrier(0);
            if (in.watdb & 2)  _wait_on_barrier(1);
            if (in.watdb & 4)  _wait_on_barrier(2);
            if (in.watdb & 8)  _wait_on_barrier(3);
            if (in.watdb & 16) _wait_on_barrier(4);
            if (in.watdb & 32) _wait_on_barrier(5);
        }

        if (in.wrtdb != 7) _set_write_barrier(in.rd, in.wrtdb);

        float a = _read_reg(in.ra);
        float b = is_immediate ? in.imm_b : _read_reg(in.rb);
        float c = _read_reg(in.rc);

        float d;
        int lat;
             if (in.type==INSTRUCTION_FFMA)          { lat = LATENCY_X32T; d = a*b + c; }
        else if (in.type==INSTRUCTION_FMUL)          { lat = LATENCY_X32T; d = a*b; }
        else if (in.type==INSTRUCTION_FADD)          { lat = LATENCY_X32T; d = a + b; }
        else if (in.type==INSTRUCTION_FFMA20I)       { lat = LATENCY_X32T; d = a*b + c; }
        else if (in.type==INSTRUCTION_FMUL20I)       { lat = LATENCY_X32T; d = a*b; }
        else if (in.type==INSTRUCTION_FADD20I)       { lat = LATENCY_X32T; d = a + b; }
        else if (in.type==INSTRUCTION_FADD20I_ABS_A) { lat = LATENCY_X32T; d = fabsf(a) + b; }
        else if (in.type==INSTRUCTION_FMIN)          { lat = LATENCY_X32T; d = (a < b) ? a : b; }
        else if (in.type==INSTRUCTION_FMAX)          { lat = LATENCY_X32T; d = (a > b) ? a : b; }
        else if (in.type==INSTRUCTION_FMAX_NEG_B)    { lat = LATENCY_X32T; d = (a > -b) ? a : -b; }
        else if (in.type==INSTRUCTION_SQRT)          { lat = LATENCY_SQRT; d = sqrtf(a); }
        else assert(false && "unhandled instruction");

        _write_reg(in.rd, d, lat);
        _step(in.stall);

        if (debug) print_instruction(in);
    }
};

float frep_eval_sim(float x0, float y0, float z0, instruction_t *instructions, int num_instructions, bool debug=false)
{
    static pas_simulator_t sim = {0};
    sim.init(debug);
    sim.reg[REGISTER_X0] = x0;
    sim.reg[REGISTER_Y0] = y0;
    sim.reg[REGISTER_Z0] = z0;
    for (int i = 0; i < num_instructions; i++)
        sim.execute(instructions[i]);
    return sim.reg[REGISTER_D+0];
}

void run_test(int test_number, frep_t *tree)
{
    instruction_blocks_t blocks = generate_sass(tree);

    int num_instructions;
    instruction_t *instructions = schedule_blocks(blocks, &num_instructions);

    printf("%d. running test...\n", test_number);

    frep_eval_sim(0.0f,0.0f,0.0f, instructions, num_instructions, true);

    for (int i = -4; i <= 4; i++)
    for (int j = -4; j <= 4; j++)
    for (int k = -4; k <= 4; k++)
    {
        float x0 = i/4.0f;
        float y0 = j/4.0f;
        float z0 = k/4.0f;
        float f_sass = frep_eval_sim(x0,y0,z0, instructions, num_instructions);
        float f_true = frep_eval(tree, x0,y0,z0);
        if (fabsf(f_sass - f_true) > 0.00001f)
        {
            printf("\nEvaluation mismatch!\n");
            printf("true: f(%.2f,%.2f,%.2f) = %f\n", x0,y0,z0,f_true);
            printf("sass: f(%.2f,%.2f,%.2f) = %f\n", x0,y0,z0,f_sass);
            printf("Instructions were:\n");
            frep_eval_sim(x0,y0,z0, instructions, num_instructions, true);
            exit(1);
        }
    }
    printf("ok!\n");
}

int main()
{
    frep_t *tree;

    tree = fBoxCheap(0.9f,0.6f,0.3f);
    run_test(0, tree);

    tree = fSphere(0.3f);
    run_test(1, tree);

    tree = fCylinder(0.6f,0.3f);
    run_test(2, tree);

    tree = fPlane(1.0f, 0.3f);
    pOpRotate(tree, 0.3f,0.5f,0.4f);
    pOpTranslate(tree, 0.2f,0.5f,0.4f);
    run_test(3, tree);

    frep_t *d1 = fBoxCheap(1.0f,0.5f,0.25f);
    pOpRotate(d1, 0.1f,0.4f,0.3f);
    pOpTranslate(d1, 0.5f,0.25f,0.25f);
    frep_t *d2 = fSphere(0.8f);
    pOpTranslate(d2, 1.0f,0,0);
    frep_t *d3 = fCylinder(0.4f, 0.2f);
    pOpTranslate(d3, 1.0f, 1.0f, 0.3f);
    tree = fOpUnion(fOpUnion(d1, d2), d3);
    run_test(4, tree);
}
