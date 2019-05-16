#include "frep.h"
#include "frep_transform.h"
#include <stdint.h>

namespace backend {
namespace ptx {

#if defined(PTX_CODEGEN_FP32)
uint32_t encode_f32(float x)
{
    return (*(uint32_t*)&x);
}
#elif defined(PTX_CODEGEN_FP20)
uint32_t encode_f32(float x)
{
    // If we truncate the last 3 bytes of 32-bit fp mantissa, we
    // can use the 20-bit immediate versions of instructions. There
    // are 32-bit immediate versions for some, but not e.g. max/min.

    // Note: ptx immediate values preserve their sign bit, unlike
    // SASS immediate values, which encode the sign bit elsewhere
    // in the instruction.
    return (*(uint32_t*)&x) & 0xFFFFF000;
}
#else
#error "You must #define either PTX_CODEGEN_FP32 or PTX_CODEGEN_FP20 before including this file."
// PTX_CODEGEN_FP20 truncates the last 3 bytes of 32-bit mantissa to use 20-bit immediate
// versions of instructions (in Maxwell and Pascal SASS). There are not 32-bit immediate
// versions for some instructions, like max/min.
#endif

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

int _generate_ptx(frep_t *node,
                 ptx_t &state,
                 frep_mat3_t R_root_to_parent=frep_identity_3x3,
                 frep_vec3_t T_parent_rel_root=frep_null_3x1)
{
    assert(node);

    frep_mat3_t R_root_to_this;
    frep_vec3_t T_this_rel_root;
    frep_computeGlobalTransform(node, &R_root_to_this, &T_this_rel_root, R_root_to_parent, T_parent_rel_root);

    int result = -1;
    if (frep_isBoolean(node))
    {
        assert(node->left);
        assert(node->right);
        int left = _generate_ptx(node->left, state, R_root_to_this, T_this_rel_root);
        int right = _generate_ptx(node->right, state, R_root_to_this, T_this_rel_root);
             if (node->opcode == FREP_UNION)         result = emit_union(state, left, right);
        else if (node->opcode == FREP_INTERSECT)     result = emit_intersect(state, left, right);
        else if (node->opcode == FREP_SUBTRACT)      result = emit_subtract(state, left, right);
        else if (node->opcode == FREP_BLEND)         result = emit_blend(state, left, right, node->blend.alpha);
        else assert(false && "Unexpected node opcode");
    }
    else if (frep_isPrimitive(node))
    {
             if (node->opcode == FREP_BOX)       result = emit_box(state, node, R_root_to_this, T_this_rel_root);
        else if (node->opcode == FREP_BOX_CHEAP) result = emit_box_cheap(state, node, R_root_to_this, T_this_rel_root);
        else if (node->opcode == FREP_SPHERE)    result = emit_sphere(state, node, R_root_to_this, T_this_rel_root);
        else if (node->opcode == FREP_CYLINDER)  result = emit_cylinder(state, node, R_root_to_this, T_this_rel_root);
        else if (node->opcode == FREP_PLANE)     result = emit_plane(state, node, R_root_to_this, T_this_rel_root);
        else assert(false && "Unexpected note opcode");
    }
    else
    {
        assert(false && "Unexpected node opcode");
    }
    return result;
}

char *generate_ptx(frep_t *node, int *result_register)
{
    static char buffer[10*1024*1024];
    ptx_t s;
    s.stream = buffer;
    s.next_register = 0;
    *result_register = _generate_ptx(node, s);
    return buffer;
}

} // namespace ptx
} // namespace backend
