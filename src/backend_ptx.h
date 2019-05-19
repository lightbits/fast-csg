// Developed by Simen Haugo.
// See LICENSE.txt for copyright and licensing details (standard MIT License).

/*
This is the code generation backend for NVIDIA PTX, which is not
a machine code target, but a fake assembly language (stored as text)
which gets compiled into native target-architecture instructions by
the CUDA driver. Note that this compilation can take a long time.
If you need to be able to rapidly compile and upload trees to the
GPU, look at the SASS backend, where we implement our own native
machine code generation.
*/

#pragma once

#include "frep.h"
#include <stdint.h>
#include <assert.h>

/*
Generates a string containing newline-seperated PTX instructions
which evaluate f(x0, y0, z0) and stores the result in a register
named "f%d" % result_register (e.g. "f3"). The input coordinates
are assumed to be in registers named "x0", "y0", and "z0".

See test/backend_ptx.cpp for an example of a complete PTX program
that uses the generated output.
*/
char *frep_compile_to_ptx(frep_t *f, int *result_register);

//////////////////////////////////////////////////////////////////
//                       Implementation
//////////////////////////////////////////////////////////////////

namespace backend_ptx {

/*
Nodes in the FRep AST have constants (such as sphere radius) that
are involved in the expression for that node's membership function.
When generating code to execute the membership function, constants
can either be placed in Constants Memory (and must be fetched with
an additional load), or be baked directly into the instructions.

For example, the PTX instruction
    add.ftz.f32 x, x, 0f3F000000; // x <- x + 0.5
uses +0.5 as an immediate value. In the generated machine code for
e.g. Maxwell architectures, this instruction may look like this:
    0x3858503f00070409
            ^^^^^
        immediate value (note that last 12 bits are truncated).

However, not all instructions can use full 32-bit floating point
immediate values. Notably, min, max and fused-multiply-add (FFMA)
on Maxwell/Pascal target architectures. But all do support 20-bit
floating point immediates, where the last 12 bits of the mantissa
are truncated (assumed to be zero).

You can choose whether you want to preserve 32-bit floating point
constants at the expense of speed, or if you want to truncate the
last 12 bits and use 20-bit floating point constants.
*/
uint32_t encode_f32(float x)
{
    #if defined(PTX_FP32_IMMEDIATE)
    return (*(uint32_t*)&x);
    #elif defined(PTX_FP20_IMMEDIATE)
    // Note: PTX immediate values preserve their sign bit, unlike
    // SASS immediate values, which encode the sign bit elsewhere
    // in the instruction.
    return (*(uint32_t*)&x) & 0xFFFFF000;
    #else
    #error "You must #define either PTX_FP32_IMMEDIATE or PTX_FP20_IMMEDIATE before including this file."
    #endif
}

struct ptx_t
{
    int next_register;
    char *stream;
};

void emit_transform(ptx_t &s, frep_mat3_t R/*root_to_this*/, frep_vec3_t T/*this_rel_root*/)
{
    // emit transform code: p_this = R_root_to_this*(p_root - T_this_rel_root)
    int x = s.next_register++;
    int y = s.next_register++;
    int z = s.next_register++;

    // compute R_root_to_this*(-T_this_rel_root)
    float tx = -(R.at(0,0)*T[0] + R.at(0,1)*T[1] + R.at(0,2)*T[2]);
    float ty = -(R.at(1,0)*T[0] + R.at(1,1)*T[1] + R.at(1,2)*T[2]);
    float tz = -(R.at(2,0)*T[0] + R.at(2,1)*T[1] + R.at(2,2)*T[2]);

    // emit instructions for R_root_to_this*p_root + R_root_to_this*(-T_this_rel_root)
    s.stream += sprintf(s.stream, "fma.rn.ftz.f32 f%d, x0, 0f%08x, 0f%08x;\n", x, encode_f32(R.at(0,0)), encode_f32(tx));
    s.stream += sprintf(s.stream, "fma.rn.ftz.f32 f%d, x0, 0f%08x, 0f%08x;\n", y, encode_f32(R.at(1,0)), encode_f32(ty));
    s.stream += sprintf(s.stream, "fma.rn.ftz.f32 f%d, x0, 0f%08x, 0f%08x;\n", z, encode_f32(R.at(2,0)), encode_f32(tz));
    s.stream += sprintf(s.stream, "fma.rn.ftz.f32 f%d, y0, 0f%08x, f%d;\n",    x, encode_f32(R.at(0,1)), x);
    s.stream += sprintf(s.stream, "fma.rn.ftz.f32 f%d, y0, 0f%08x, f%d;\n",    y, encode_f32(R.at(1,1)), y);
    s.stream += sprintf(s.stream, "fma.rn.ftz.f32 f%d, y0, 0f%08x, f%d;\n",    z, encode_f32(R.at(2,1)), z);
    s.stream += sprintf(s.stream, "fma.rn.ftz.f32 f%d, z0, 0f%08x, f%d;\n",    x, encode_f32(R.at(0,2)), x);
    s.stream += sprintf(s.stream, "fma.rn.ftz.f32 f%d, z0, 0f%08x, f%d;\n",    y, encode_f32(R.at(1,2)), y);
    s.stream += sprintf(s.stream, "fma.rn.ftz.f32 f%d, z0, 0f%08x, f%d;\n",    z, encode_f32(R.at(2,2)), z);
}

int emit_box(ptx_t &s, frep_t *node, frep_mat3_t R/*root_to_this*/, frep_vec3_t T/*this_rel_root*/)
{
    assert(false && "Box is not implemented in PTX backend yet");
    return 0;
}

int emit_box_cheap(ptx_t &s, frep_t *node, frep_mat3_t R/*root_to_this*/, frep_vec3_t T/*this_rel_root*/)
{
    // mathematical expression: Box(p, width,height,depth)
    // (x,y,z) = R*(p - T)
    // d = max( |x|-width, |y|-height, |z|-depth )

    // ptx template:
    // <transform template>
    // abs.ftz.f32 x, x;
    // abs.ftz.f32 y, y;
    // abs.ftz.f32 z, z;
    // sub.ftz.f32 x, x, (width);
    // sub.ftz.f32 y, y, (height);
    // sub.ftz.f32 z, z, (depth);
    // max.ftz.f32 d, x, y;
    // max.ftz.f32 d, d, z;

    // emitted instructions:
    emit_transform(s, R, T); // todo: inline here and optimize for each primitive
    int x = s.next_register - 3;
    int y = s.next_register - 2;
    int z = s.next_register - 1;
    int d = s.next_register++;
    s.stream += sprintf(s.stream, "abs.ftz.f32 f%d,f%d;\n", x, x);
    s.stream += sprintf(s.stream, "abs.ftz.f32 f%d,f%d;\n", y, y);
    s.stream += sprintf(s.stream, "abs.ftz.f32 f%d,f%d;\n", z, z);
    s.stream += sprintf(s.stream, "add.ftz.f32 f%d,f%d,0f%08x;\n", x, x, encode_f32(-node->box.width));
    s.stream += sprintf(s.stream, "add.ftz.f32 f%d,f%d,0f%08x;\n", y, y, encode_f32(-node->box.height));
    s.stream += sprintf(s.stream, "add.ftz.f32 f%d,f%d,0f%08x;\n", z, z, encode_f32(-node->box.depth));
    s.stream += sprintf(s.stream, "max.ftz.f32 f%d,f%d,f%d;\n", d, x, y);
    s.stream += sprintf(s.stream, "max.ftz.f32 f%d,f%d,f%d;\n", d, d, z);
    return d;
}

int emit_sphere(ptx_t &s, frep_t *node, frep_mat3_t R/*root_to_this*/, frep_vec3_t T/*this_rel_root*/)
{
    // mathematical expression:
    // d = length(p_this) - r
    //   = length(R*(p_root - T)) - r
    //   = length(p_root - T) - r

    // ptx template:
    // add.ftz.f32 x, x0, (-tx);
    // add.ftz.f32 y, y0, (-ty);
    // add.ftz.f32 z, z0, (-tz);
    // mul.ftz.f32 d, x, x;
    // fma.rn.ftz.f32 d, y, y, d;
    // fma.rn.ftz.f32 d, z, z, d;
    // sqrt.approx.ftz.f32 d, d;
    // sub.f32 d, d, (r);

    // emitted instructions:
    int x = s.next_register++;
    int y = s.next_register++;
    int z = s.next_register++;
    int d = s.next_register++;
    s.stream += sprintf(s.stream, "add.ftz.f32 f%d,x0,0f%08x;\n", x, encode_f32(-T[0]));                // x <- x0 - (Tx)
    s.stream += sprintf(s.stream, "add.ftz.f32 f%d,y0,0f%08x;\n", y, encode_f32(-T[1]));                // y <- y0 - (Ty)
    s.stream += sprintf(s.stream, "add.ftz.f32 f%d,z0,0f%08x;\n", z, encode_f32(-T[2]));                // z <- z0 - (Tz)
    s.stream += sprintf(s.stream, "mul.ftz.f32 f%d,f%d,f%d;\n", d, x, x);                               // d <- x*x
    s.stream += sprintf(s.stream, "fma.rn.ftz.f32 f%d,f%d,f%d,f%d;\n", d, y, y, d);                     // d <- y*y + d
    s.stream += sprintf(s.stream, "fma.rn.ftz.f32 f%d,f%d,f%d,f%d;\n", d, z, z, d);                     // d <- z*z + d
    s.stream += sprintf(s.stream, "sqrt.approx.ftz.f32 f%d,f%d;\n", d, d);                              // d <- sqrt(d)
    s.stream += sprintf(s.stream, "add.f32 f%d,f%d,0f%08x;\n", d, d, encode_f32(-node->sphere.radius)); // d <- d - (r)
    return d;
}

int emit_cylinder(ptx_t &s, frep_t *node, frep_mat3_t R/*root_to_this*/, frep_vec3_t T/*this_rel_root*/)
{
    // mathematical expression: cylinder(p, 2*height, radius)
    // (x,y,z) = R*(p - T)
    // d = max( sqrt(x*x + z*z) - radius, abs(y) - height )

    // ptx template
    // <transform template>
    // mul.ftz.f32 d, x, x;
    // fma.rn.ftz.f32 d, z, z, d;
    // sqrt.approx.ftz.f32 d, d;
    // abs.ftz.f32 y, y;
    // add.ftz.f32 y, y, (-height);
    // add.ftz.f32 d, d, (-radius);
    // max.ftz.f32 d, d, y;

    // emitted instructions:
    emit_transform(s, R, T); // todo: inline here and optimize for each primitive
    int x = s.next_register - 3;
    int y = s.next_register - 2;
    int z = s.next_register - 1;
    int d = s.next_register++;
    s.stream += sprintf(s.stream, "mul.ftz.f32 f%d,f%d,f%d;\n", d, x, x);
    s.stream += sprintf(s.stream, "fma.rn.ftz.f32 f%d,f%d,f%d,f%d;\n", d, z, z, d);
    s.stream += sprintf(s.stream, "sqrt.approx.ftz.f32 f%d,f%d;\n", d, d);
    s.stream += sprintf(s.stream, "abs.ftz.f32 f%d,f%d;\n", y, y);
    s.stream += sprintf(s.stream, "add.ftz.f32 f%d,f%d,0f%08x;\n", y, y, encode_f32(-node->cylinder.height));
    s.stream += sprintf(s.stream, "add.ftz.f32 f%d,f%d,0f%08x;\n", d, d, encode_f32(-node->cylinder.radius));
    s.stream += sprintf(s.stream, "max.ftz.f32 f%d,f%d,f%d;\n", d, d, y);
    return d;
}

int emit_plane(ptx_t &s, frep_t *node, frep_mat3_t R/*root_to_this*/, frep_vec3_t T/*this_rel_root*/)
{
    // mathematical expression:
    // (x,y,z) = R*(p - T)
    // d = x - plane.x
    //   = R00*(x0 - Tx) + R01*(y0 - Ty) + R02*(z0 - Tz) - plane.x
    //   = R00*x0 + R01*y0 + R02*z0 + (-plane.x - R00*Tx - R01*Ty - R02*Tz)
    //   = R00*x0 + R01*y0 + R02*z0 + k

    // ptx template:
    // mul.ftz.f32 d, x0, (R00);
    // fma.rn.ftz.f32 d, y0, (R01), d;
    // fma.rn.ftz.f32 d, z0, (R02), d;
    // add.ftz.f32 d, d, (k)

    // emitted instructions:
    float k = -(R.at(0,0)*T[0] + R.at(0,1)*T[1] + R.at(0,2)*T[2] + node->plane.offset);
    int d = s.next_register++;
    s.stream += sprintf(s.stream, "mul.ftz.f32 f%d,x0,0f%08x;\n", d, encode_f32(R.at(0,0)));
    s.stream += sprintf(s.stream, "fma.rn.ftz.f32 f%d,y0,0f%08x,f%d;\n", d, encode_f32(R.at(0,1)), d);
    s.stream += sprintf(s.stream, "fma.rn.ftz.f32 f%d,z0,0f%08x,f%d;\n", d, encode_f32(R.at(0,2)), d);
    s.stream += sprintf(s.stream, "add.ftz.f32 f%d,f%d,0f%08x;\n", d, d, encode_f32(k));
    return d;
}

int emit_union(ptx_t &s, int left, int right)
{
    int d = s.next_register++;
    s.stream += sprintf(s.stream, "min.ftz.f32 f%d,f%d,f%d;\n", d, left, right);
    return d;
}

int emit_intersect(ptx_t &s, int left, int right)
{
    int d = s.next_register++;
    s.stream += sprintf(s.stream, "max.ftz.f32 f%d,f%d,f%d;\n", d, left, right);
    return d;
}

int emit_subtract(ptx_t &s, int left, int right)
{
    int d = s.next_register++;
    s.stream += sprintf(s.stream, "neg.ftz.f32 f%d,f%d;\n", right, right);
    s.stream += sprintf(s.stream, "max.ftz.f32 f%d,f%d,f%d;\n", d, left, right);
    return d;
}

int emit_blend(ptx_t &s, int left, int right, float blend_alpha)
{
    int d = s.next_register++;
    s.stream += sprintf(s.stream, "mul.ftz.f32 f%d,f%d,0f%08x;\n", d, left, encode_f32(blend_alpha));
    s.stream += sprintf(s.stream, "fma.rn.ftz.f32 f%d,f%d,0f%08x,f%d;\n", d, right, encode_f32(1.0f-blend_alpha), d);
    return d;
}

int _frep_compile_to_ptx(
    frep_t *node,
    ptx_t &state,
    frep_mat3_t R_root_to_parent=frep_identity_3x3,
    frep_vec3_t T_parent_rel_root=frep_null_3x1)
{
    assert(node);

    frep_mat3_t R_root_to_this;
    frep_vec3_t T_this_rel_root;
    frep_get_global_transform(node, &R_root_to_this, &T_this_rel_root, R_root_to_parent, T_parent_rel_root);

    int result = -1;
    if (frep_is_boolean(node))
    {
        assert(node->left);
        assert(node->right);
        int left = _frep_compile_to_ptx(node->left, state, R_root_to_this, T_this_rel_root);
        int right = _frep_compile_to_ptx(node->right, state, R_root_to_this, T_this_rel_root);
        switch (node->opcode)
        {
            case FREP_UNION:     return emit_union(state, left, right);
            case FREP_INTERSECT: return emit_intersect(state, left, right);
            case FREP_SUBTRACT:  return emit_subtract(state, left, right);
            case FREP_BLEND:     return emit_blend(state, left, right, node->blend.alpha);
        }
    }
    else if (frep_is_primitive(node))
    {
        switch (node->opcode)
        {
            case FREP_BOX:       return emit_box(state, node, R_root_to_this, T_this_rel_root);
            case FREP_BOX_CHEAP: return emit_box_cheap(state, node, R_root_to_this, T_this_rel_root);
            case FREP_SPHERE:    return emit_sphere(state, node, R_root_to_this, T_this_rel_root);
            case FREP_CYLINDER:  return emit_cylinder(state, node, R_root_to_this, T_this_rel_root);
            case FREP_PLANE:     return emit_plane(state, node, R_root_to_this, T_this_rel_root);
        }
    }

    assert(false && "Unexpected node opcode");
    return -1;
}

}

char *frep_compile_to_ptx(frep_t *node, int *result_register)
{
    using namespace backend_ptx;
    static char *buffer = (char*)malloc(10*1024*1024);
    assert(buffer && "Failed to allocate buffer to contain PTX output");
    ptx_t s;
    s.stream = buffer;
    s.next_register = 0;
    *result_register = _frep_compile_to_ptx(node, s);
    return buffer;
}
