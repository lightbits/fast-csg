#include <stdint.h>
#include "../pas/pas.h"

static const uint64_t PATTERN_FADD      = 0x5c58000000070000; bool is_FADD(uint64_t x)     { return (x & 0xfff8000000000000) == 0x5c58000000000000; }
static const uint64_t PATTERN_FADD20I   = 0x3858000000070000; bool is_FADD20I(uint64_t x)  { return (x & 0xfff8000000000000) == 0x3858000000000000; }
static const uint64_t PATTERN_FLOOR32F  = 0x5ca8148000070a00; bool is_FLOOR32F(uint64_t x) { return (x & 0xfff8000000000000) == 0x5ca8000000000000; }
static const uint64_t PATTERN_FMUL20I   = 0x3868000000070000; bool is_FMUL20I(uint64_t x)  { return (x & 0xfff8000000000000) == 0x3868000000000000; }
static const uint64_t PATTERN_FMUL      = 0x5c68000000070000; bool is_FMUL(uint64_t x)     { return (x & 0xfff8000000000000) == 0x5c68000000000000; }
static const uint64_t PATTERN_FFMA      = 0x5980000000070000; bool is_FFMA(uint64_t x)     { return (x & 0xfff8000000000000) == 0x5980000000000000; }
static const uint64_t PATTERN_FFMA20I   = 0x3280000000070000; bool is_FFMA20I(uint64_t x)  { return (x & 0xfff8000000000000) == 0x3280000000000000; }
static const uint64_t PATTERN_FMAX      = 0x5c60078000070000; bool is_FMAX(uint64_t x)     { return (x & 0xfff8000000000000) == 0x5c60000000000000; }
static const uint64_t PATTERN_FMIN      = 0x5c60038000070000; bool is_FMIN(uint64_t x)     { return (x & 0xfff8000000000000) == 0x5c60000000000000; }
static const uint64_t PATTERN_FMAX20I   = 0x3860078000070000; bool is_FMAX20I(uint64_t x)  { return (x & 0xfff8000000000000) == 0x3860000000000000; }
static const uint64_t PATTERN_FMIN20I   = 0x3860038000070000; bool is_FMIN20I(uint64_t x)  { return (x & 0xfff8000000000000) == 0x3860000000000000; }
static const uint64_t PATTERN_MUFU_SQRT = 0x5080000000870000; bool is_SQRT(uint64_t x)     { return (x & 0xfff8000000000000) == 0x5080000000000000; }

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
    void execute(uint64_t in)
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
             if (in.type==INSTRUCTION_FFMA)          { lat = 6; d = a*b + c; }
        else if (in.type==INSTRUCTION_FMUL)          { lat = 6; d = a*b; }
        else if (in.type==INSTRUCTION_FADD)          { lat = 6; d = a + b; }
        else if (in.type==INSTRUCTION_FFMA20I)       { lat = 6; d = a*b + c; }
        else if (in.type==INSTRUCTION_FMUL20I)       { lat = 6; d = a*b; }
        else if (in.type==INSTRUCTION_FADD20I)       { lat = 6; d = a + b; }
        else if (in.type==INSTRUCTION_FADD20I_ABS_A) { lat = 6; d = fabsf(a) + b; }
        else if (in.type==INSTRUCTION_FMIN)          { lat = 6; d = (a < b) ? a : b; }
        else if (in.type==INSTRUCTION_FMAX)          { lat = 6; d = (a > b) ? a : b; }
        else if (in.type==INSTRUCTION_FMAX_NEG_B)    { lat = 6; d = (a > -b) ? a : -b; }
        else if (in.type==INSTRUCTION_SQRT)          { lat = 8; d = sqrtf(a); }
        else assert(false && "unhandled instruction");

        _write_reg(in.rd, d, lat);
        _step(in.stall);

        if (debug) print_instruction(in);
    }
};
