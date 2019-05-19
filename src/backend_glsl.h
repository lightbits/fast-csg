// Developed by Simen Haugo.
// See LICENSE.txt for copyright and licensing details (standard MIT License).

// This is the code generation backend for GLSL (GL Shading Language).
// The output is a stripped GLSL source code, meaning you must insert
// it into a GLSL shader as necessary for your application.

#pragma once
#include "frep.h"
#include <assert.h>

// Generates a null-terminated string of GLSL code that computes
//
// Variables are expected to be defined:
//     vec3 p0;
//
// Output is stored in:
//     float d1 = f(p0.x, p0.y, p0.z);
//
// The following functions must be declared and linked into the GLSL:
//     float fBox(vec3 p, vec3 dim);
//     float fBoxCheap(vec3 p, vec3 dim);
//     float fCylinder(vec3 p, float r, float h);
//     float fSphere(vec3 p, float r, float h);
//
char *frep_compile_to_glsl(frep_t *f);

//////////////////////////////////////////////////////////////////
//                       Implementation
//////////////////////////////////////////////////////////////////

namespace backend_glsl {

struct glsl_t
{
    int destination;
    char *stream;
};

int _frep_compile_to_glsl(frep_t *node,
                   glsl_t &s,
                   frep_mat3_t R_root_to_parent=frep_identity_3x3,
                   frep_vec3_t T_parent_rel_root=frep_null_3x1)
{
    assert(node);

    frep_mat3_t R_root_to_this;
    frep_vec3_t T_this_rel_root;
    frep_get_global_transform(node, &R_root_to_this, &T_this_rel_root, R_root_to_parent, T_parent_rel_root);

    int my_index = s.destination++;

    // p^this = R_root_to_this*(p^0 - T_this_rel_root)
    //        = R_root_to_this*p^0 + (-R_root_to_this*T_this_rel_root)
    {
        #define R(row,col) R_root_to_this.at(row,col)
        #define T(i) T_this_rel_root[i]
        float dtx = -(R(0,0)*T(0) + R(0,1)*T(1) + R(0,2)*T(2));
        float dty = -(R(1,0)*T(0) + R(1,1)*T(1) + R(1,2)*T(2));
        float dtz = -(R(2,0)*T(0) + R(2,1)*T(1) + R(2,2)*T(2));
        s.stream += sprintf(s.stream,
            "vec3 p%d = "
            "vec3(%f,%f,%f)*p0.x + "
            "vec3(%f,%f,%f)*p0.y + "
            "vec3(%f,%f,%f)*p0.z + "
            "vec3(%f,%f,%f);\n",
            my_index,
            R(0,0), R(1,0), R(2,0),
            R(0,1), R(1,1), R(2,1),
            R(0,2), R(1,2), R(2,2),
            dtx, dty, dtz
        );
        #undef R
        #undef T
    }

    if (frep_is_boolean(node))
    {
        assert(node->left);
        assert(node->right);

        int i_left = _frep_compile_to_glsl(node->left, s, R_root_to_this, T_this_rel_root);
        int i_right = _frep_compile_to_glsl(node->right, s, R_root_to_this, T_this_rel_root);

        s.stream += sprintf(s.stream, "float d%d = ", my_index);

        switch (node->opcode)
        {
            case FREP_UNION: s.stream += sprintf(s.stream, "min(d%d,d%d);\n", i_left, i_right); break;
            case FREP_INTERSECT: s.stream += sprintf(s.stream, "max(d%d,d%d);\n", i_left, i_right); break;
            case FREP_SUBTRACT: s.stream += sprintf(s.stream, "max(d%d,-d%d);\n", i_left, i_right); break;
            case FREP_BLEND: s.stream += sprintf(s.stream, "%f*d%d + %f*d%d;\n", node->blend.alpha, i_left, 1.0f-node->blend.alpha, i_right); break;
            default: assert(false && "Unexpected opcode");
        }
    }
    else if (frep_is_primitive(node))
    {
        s.stream += sprintf(s.stream, "float d%d = ", my_index);

        switch (node->opcode)
        {
            case FREP_BOX:       s.stream += sprintf(s.stream, "fBox(p%d, vec3(%f, %f, %f));\n", my_index, node->box.width, node->box.height, node->box.depth); break;
            case FREP_BOX_CHEAP: s.stream += sprintf(s.stream, "fBoxCheap(p%d, vec3(%f, %f, %f));\n", my_index, node->box.width, node->box.height, node->box.depth); break;
            case FREP_SPHERE:    s.stream += sprintf(s.stream, "fSphere(p%d, %f);\n", my_index, node->sphere.radius); break;
            case FREP_CYLINDER:  s.stream += sprintf(s.stream, "fCylinder(p%d, %f, %f);\n", my_index, node->cylinder.radius, node->cylinder.height); break;
            case FREP_PLANE:     s.stream += sprintf(s.stream, "p%d.x - %f;\n", my_index, node->plane.offset); break;
            default: assert(false && "Unexpected opcode");
        }
    }
    else
    {
        assert(false && "Unexpected node type");
    }
    return my_index;
}

}

char *frep_compile_to_glsl(frep_t *node)
{
    using namespace backend_glsl;
    static char *buffer = (char*)malloc(10*1024*1024);
    assert(buffer && "Failed to allocate buffer to contain GLSL output");
    glsl_t s;
    s.stream = buffer;
    s.destination = 1;
    _frep_compile_to_glsl(node, s);
    return buffer;
}
