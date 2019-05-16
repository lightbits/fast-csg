#include <iostream>
#include <math.h>
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include "ptxjit.h"

#define SDF_APPROX_BOX_EVAL
#define PTX_CODEGEN_FP20
#include "../sdf/sdf.h"
#include "../sdf/sdf_codegen_ptx.h"
#include "../sdf/test_models.h"

const float finite_difference_epsilon = 0.001f;

// This generates a PTX program roughly equivalent to:
//   main(vec4 *input, vec4 *outputs)
//   {
//       int tid = threadIdx.x + blockDim.x*blockIdx.x;
//       vec4 p = input[tid];
//       for (int i = 0; i < num_models; i++)
//       {
//           outputs[tid + num_points*i].x = sdf_eval(models[i], p.x, p.y, p.z);
//       }
//   }
// out_length _DOES NOT_ include the null-terminator.
char *sdf_codegen_ptx_program(sdf_node_t **models, int num_models, int num_points, size_t *out_length)
{
    static char buffer[10*1024*1024];

    // write header
    // todo: this assumes a 64-bit machine, sm_60 and ptx isa version 6.0
    char *stream = buffer;
    stream += sprintf(stream,
        ".version 6.0\n"
        ".target sm_60\n"
        ".address_size 64\n"
    );

    // write each model function
    for (int i = 0; i < num_models; i++)
    {
        int result_register;
        char *ptx = sdf_codegen_ptx(models[i], &result_register);
        stream += sprintf(stream, ".func (.reg.f32 f%d) model%d(.reg.f32 x0, .reg.f32 y0, .reg.f32 z0)\n{\n", result_register, i);
        stream += sprintf(stream, ".reg.f32 f<%d>;\n", result_register); // declare registers f0, ..., f_(result_register-1)
        stream += sprintf(stream, "%sret.uni;\n}\n", ptx);
    }

    printf("%s\n", buffer);

    // write preamble of main function
    // (compute global tid, load point, and compute destination address)
    // TODO: this has hardcoded constant (16=sizeof(vec4)) for size of input and output elements
    stream += sprintf(stream,
        ".visible.entry map_to_error(.param .u64 param0, .param .u64 param1)\n"
        "{\n"
        "\t.reg.f32 x0;\n"
        "\t.reg.f32 y0;\n"
        "\t.reg.f32 z0;\n"
        "\t.reg.f32 w0;\n"
        "\t.reg .b32 %%r<5>;\n"
        "\t.reg .b64 %%rd<9>;\n"
        "\tld.param.u64 %%rd1, [param0];\n"
        "\tld.param.u64 %%rd2, [param1];\n"
        "\tcvta.to.global.u64  %%rd3, %%rd2;\n"
        "\tcvta.to.global.u64  %%rd4, %%rd1;\n"
        "\tmov.u32 %%r1, %%tid.x;\n"             // threadIdx.x
        "\tmov.u32 %%r2, %%ctaid.x;\n"           // blockIdx.x
        "\tmov.u32 %%r3, %%ntid.x;\n"            // blockDim.x
        "\tmad.lo.s32 %%r4, %%r3, %%r2, %%r1;\n" // blockDim.x*blockIdx.x + threadIdx.x
        "\tmul.wide.s32 %%rd5, %%r4, 16;\n"      // sizeof(vec4)*(...)
        "\tadd.s64 %%rd6, %%rd4, %%rd5;\n"       // param0 + sizeof(vec4)*(...)
        "\tld.global.v4.f32 {x0, y0, z0, w0}, [%%rd6];\n"
        "\tmul.wide.s32 %%rd7, %%r4, 16;\n"      // sizeof(vec4)*(...)
        "\tadd.s64 %%rd8, %%rd3, %%rd7;\n"       // param1 + sizeof(float)*(...)
    );

    #if 0

    // This path only evaluates the distance of the model and stores the result
    // in the output array.

    stream += sprintf(stream, "\t.reg.f32 d;\n");
    for (int o = 0; o < num_models; o++)
    {
        int st_byte_offset = (o*num_points*4 + 3)*sizeof(float);
        stream += sprintf(stream, "\tcall.uni (d), model%d, (x0,y0,z0);\n", o);
        stream += sprintf(stream, "\tst.global.f32 [%%rd8 + %d], d;\n", st_byte_offset);
    }

    #else

    // This path evaluates both distance and estimates the normal by finite difference
    // and stores both in the output array, in the order (distance, normal_x, normal_y, normal_z).

    // compute perturbed input coordinates for finite differencing below
    stream += sprintf(stream,
        "\t.reg.f32 x0p;\n"
        "\t.reg.f32 y0p;\n"
        "\t.reg.f32 z0p;\n"
        "\tadd.ftz.f32 x0p, x0, 0f%08x;\n"
        "\tadd.ftz.f32 y0p, y0, 0f%08x;\n"
        "\tadd.ftz.f32 z0p, z0, 0f%08x;\n"
        ,
        ptx_encode_f32(finite_difference_epsilon),
        ptx_encode_f32(finite_difference_epsilon),
        ptx_encode_f32(finite_difference_epsilon)
    );

    // declare registers
    stream += sprintf(stream,
        "\t.reg.f32 d;\n"
        "\t.reg.f32 nx;\n"
        "\t.reg.f32 ny;\n"
        "\t.reg.f32 nz;\n"
        "\t.reg.f32 ilen;\n"
        "\t.reg.f32 err;\n"
    );

    for (int model_index = 0; model_index < num_models; model_index++)
    {
        // todo: can probably use some goto's here to reduce code?
        stream += sprintf(stream,
            // evaluate distance at center
            "\tcall.uni (d), model%d, (x0,y0,z0);\n"
            // evaluate distance at finite-difference coordinates
            "\tcall.uni (nx), model%d, (x0p,y0,z0);\n"
            "\tcall.uni (ny), model%d, (x0,y0p,z0);\n"
            "\tcall.uni (nz), model%d, (x0,y0,z0p);\n"
            // evaluate dfdx dfdy dfdz
            "\tsub.ftz.f32 nx, nx, d;\n"
            "\tsub.ftz.f32 ny, ny, d;\n"
            "\tsub.ftz.f32 nz, nz, d;\n"
#if 0 // Normalization is a problem (div by zero). Don't do it for now.
            // compute 1/length(dfdx,dfdy,dfdz)
            "\tmul.ftz.f32 ilen, nx, nx;\n"
            "\tfma.rn.ftz.f32 ilen, ny, ny, ilen;\n"
            "\tfma.rn.ftz.f32 ilen, nz, nz, ilen;\n"
            "\trsqrt.approx.ftz.f32 ilen, ilen;\n"
            // normalize (dfdx,dfdy,dfdz)
            "\tmul.ftz.f32 nx, nx, ilen;\n"
            "\tmul.ftz.f32 ny, ny, ilen;\n"
            "\tmul.ftz.f32 nz, nz, ilen;\n"
#endif
            ,
            model_index, model_index, model_index, model_index
        );

        // todo: byte offset overflow?
        // todo: how big can these be??? uint32?
        // todo: add asserts here?
        uint64_t st_byte_offset_nx = (uint64_t)((model_index*num_points*4 + 0)*sizeof(float));
        uint64_t st_byte_offset_ny = (uint64_t)((model_index*num_points*4 + 1)*sizeof(float));
        uint64_t st_byte_offset_nz = (uint64_t)((model_index*num_points*4 + 2)*sizeof(float));
        uint64_t st_byte_offset_d  = (uint64_t)((model_index*num_points*4 + 3)*sizeof(float));
        stream += sprintf(stream,
            "\tst.global.f32 [%%rd8 + %"PRIu64"], d;\n"
            "\tst.global.f32 [%%rd8 + %"PRIu64"], nx;\n"
            "\tst.global.f32 [%%rd8 + %"PRIu64"], ny;\n"
            "\tst.global.f32 [%%rd8 + %"PRIu64"], nz;\n"
            ,
            st_byte_offset_d,
            st_byte_offset_nx,
            st_byte_offset_ny,
            st_byte_offset_nz);
    }
    #endif

    stream += sprintf(stream,
        "\tret;\n"
        "}\n"
    );

    *out_length = (stream - buffer);

    return buffer;
}

int main(int argc, char **argv)
{
    // Note: input and output arrays should be aligned to vec4 boundaries.
    sdf_node_t *models[] =
    {
        model_complex04(),
        model_simple01(),
        // model_simple03()
    };

    enum { num_points_x = 4, num_points_y = 4, num_points_z = 4 };
    enum { num_points = num_points_x*num_points_y*num_points_z };
    enum { num_models = sizeof(models)/sizeof(models[0]) };
    enum { num_threads = 32 };
    enum { num_blocks  = num_points/num_threads };
    enum { num_output = num_models };
    enum { sizeof_input = num_points*4*sizeof(float) };
    enum { sizeof_output = num_points*4*sizeof(float)*num_output };
    float *output = (float*)malloc(sizeof_output); assert(output);
    float *cpu_output = (float*)malloc(sizeof_output); assert(cpu_output);
    float *points = (float*)malloc(num_points*4*sizeof(float));
    float *input = points;

    size_t ptx_source_length;
    char *ptx_source = sdf_codegen_ptx_program(models, num_models, num_points, &ptx_source_length);


    // generate point cloud (regular grid sampling in 3D) and evaluate distance at each point using
    // "ground-truth" CPU evaluator.
    {
        // Assert(num_models == 1 && "haven't implemented proper support for multi models");

        // generate regular voxel grid sampling
        float *p = points;
        for (int zi = 0; zi < num_points_z; zi++)
        for (int yi = 0; yi < num_points_y; yi++)
        for (int xi = 0; xi < num_points_x; xi++)
        {
            p[0] = (-1.0f + 2.0f*xi/num_points_x);
            p[1] = (-1.0f + 2.0f*yi/num_points_y);
            p[2] = (-1.0f + 2.0f*zi/num_points_z);
            p[3] = 0.0f;
            p += 4;
        }

        // evaluate distance and normal with CPU
        for (int o = 0; o < num_models; o++)
        for (int i = 0; i < num_points; i++)
        {
            float x = points[4*i + 0];
            float y = points[4*i + 1];
            float z = points[4*i + 2];

            sdf_node_t *model = models[o];
            float d = sdf_eval(model, x, y, z);
            float eps = finite_difference_epsilon;
            float dpx = sdf_eval(model, x+eps, y, z) - d;
            float dpy = sdf_eval(model, x, y+eps, z) - d;
            float dpz = sdf_eval(model, x, y, z+eps) - d;
            // float iln = 1.0f/sqrtf(dpx*dpx + dpy*dpy + dpz*dpz);
            // float nx = dpx*iln;
            // float ny = dpy*iln;
            // float nz = dpz*iln;

            cpu_output[4*(i + o*num_points) + 0] = dpx;
            cpu_output[4*(i + o*num_points) + 1] = dpy;
            cpu_output[4*(i + o*num_points) + 2] = dpz;
            cpu_output[4*(i + o*num_points) + 3] = d;
        }
    }

    RunPTX(input, sizeof_input,
           output, sizeof_output,
           ptx_source, ptx_source_length,
           "map_to_error",
           num_blocks, num_threads);

    // verify that output matches CPU evaluation
    for (int o = 0; o < num_output; o++)
    for (int i = 0; i < num_points; i++)
    {
        float nx_cpu = cpu_output[4*(i + o*num_points) + 0];
        float ny_cpu = cpu_output[4*(i + o*num_points) + 1];
        float nz_cpu = cpu_output[4*(i + o*num_points) + 2];
        float d_cpu  = cpu_output[4*(i + o*num_points) + 3];

        float nx_ptx = output[4*(i + o*num_points) + 0];
        float ny_ptx = output[4*(i + o*num_points) + 1];
        float nz_ptx = output[4*(i + o*num_points) + 2];
        float d_ptx  = output[4*(i + o*num_points) + 3];

        // printf("%.9g %.9g\n", nx_cpu, nx_ptx);
        assert(fabsf(nx_cpu - nx_ptx) < 0.01f);
        assert(fabsf(ny_cpu - ny_ptx) < 0.01f);
        assert(fabsf(nz_cpu - nz_ptx) < 0.01f);
        assert(fabsf(d_cpu - d_ptx) < 0.01f);
    }

    free(output);

    return 0;
}
