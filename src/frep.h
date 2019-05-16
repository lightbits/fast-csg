#pragma once
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

typedef int frep_opcode_t;
enum frep_opcode_ {
    FREP_INVALID = 0,

    FREP_BOX,
    FREP_BOX_CHEAP,
    FREP_SPHERE,
    FREP_CYLINDER,
    FREP_PLANE,
    FREP_UNION,
    FREP_INTERSECT,
    FREP_SUBTRACT,
    FREP_BLEND,
};

struct frep_box_t      { float width, height, depth; };
struct frep_sphere_t   { float radius; };
struct frep_cylinder_t { float radius, height; };
struct frep_plane_t    { float sign, offset; };
struct frep_blend_t    { float alpha; };

/*
Each frep node has a rigid-body transform associated with it.
It can be the identity. If so, it gets optimized out in the
backend. The transformation parameters relate the point argument
of the child node to its parent node by:

    p^parent = Rx(rx)*Ry(ry)*Rz(rz)*p^child + (tx,ty,tz)

*/
struct frep_t {
    frep_opcode_t opcode;
    frep_t *left;
    frep_t *right;
    float rx,ry,rz,tx,ty,tz;
    union {
        frep_box_t box;
        frep_sphere_t sphere;
        frep_cylinder_t cylinder;
        frep_plane_t plane;
        frep_blend_t blend;
    };
};

/*
Node creation and deletion utilities
*/
frep_t *frep_malloc() {
    frep_t *f = (frep_t*)malloc(sizeof(frep_t));
    return f;
}
frep_t *frep_calloc() {
    frep_t *f = (frep_t*)calloc(1, sizeof(frep_t));
    return f;
}
void frep_free(frep_t *f) {
    if (!f) return;
    frep_free(f->left);
    frep_free(f->right);
    free(f);
}
frep_t *frep_copy(frep_t *f) {
    if (!f) return NULL;
    frep_t *f1 = frep_malloc();
    *f1 = *f;
    f1->left = frep_copy(f->left);
    f1->right = frep_copy(f->right);
    return f1;
}

/*
Other utilities
*/
bool frep_is_primitive(frep_t *f) {
    return f->opcode == FREP_BOX ||
           f->opcode == FREP_BOX_CHEAP ||
           f->opcode == FREP_SPHERE ||
           f->opcode == FREP_CYLINDER ||
           f->opcode == FREP_PLANE;
}
bool frep_is_boolean(frep_t *f) {
    return f->opcode == FREP_UNION ||
           f->opcode == FREP_INTERSECT ||
           f->opcode == FREP_SUBTRACT;
}
int frep_get_num_nodes(frep_t *f) {
    if (!f) return 0;
    return 1 + frep_get_num_nodes(f->left) + frep_get_num_nodes(f->right);
}

int frep_get_depth(frep_t *f) {
    if (!f) return 0;
    int l = frep_get_depth(f->left);
    int r = frep_get_depth(f->right);
    int max_lr = (l > r ? l : r);
    return 1 + max_lr;
}
frep_t *frep_find_node(frep_t *a, int find_i, frep_t **out_parent, int *out_depth, frep_t *parent=NULL, int depth=0)
{
    assert(a);
    assert(find_i >= 0);

    static int i = 0;
    if (!parent) i = 0;
    else i++;

    if (i == find_i)
    {
        *out_depth = depth;
        *out_parent = parent;
        return a;
    }
    else if (frep_is_boolean(a))
    {
        frep_t *left = frep_find_node(a->left, find_i, out_parent, out_depth, a, depth+1);
        if (left) return left;
        frep_t *right = frep_find_node(a->right, find_i, out_parent, out_depth, a, depth+1);
        if (right) return right;
    }
    return NULL;
}

/*
Utility routines for computing rigid-body transform from root node to a specific child.
*/
struct frep_mat3_t { float d[3*3]; float &at(int row, int col) { return d[col + row*3]; } };
struct frep_vec3_t { float d[3]; float &operator[](int i) { return d[i]; } };
static frep_mat3_t frep_identity_3x3 = { 1,0,0, 0,1,0, 0,0,1 };
static frep_vec3_t frep_null_3x1 = { 0,0,0 };

// d = a*b
frep_mat3_t frep_mat_mul(frep_mat3_t a, frep_mat3_t b) {
    frep_mat3_t d = {0};
    for (int row = 0; row < 3; row++)
    for (int col = 0; col < 3; col++)
    {
        d.at(row,col) = 0.0f;
        for (int i = 0; i < 3; i++)
            d.at(row,col) += a.at(row,i)*b.at(i,col);
    }
    return d;
}

// d = transpose(a) * b
frep_vec3_t frep_mat_mul_transpose(frep_mat3_t a, frep_vec3_t b) {
    frep_vec3_t d = {0};
    for (int row = 0; row < 3; row++)
    {
        d[row] = 0.0f;
        for (int i = 0; i < 3; i++)
            d[row] += a.at(i,row)*b[i];
    }
    return d;
}

frep_vec3_t frep_mat_add(frep_vec3_t a, frep_vec3_t b) {
    frep_vec3_t d = { a[0]+b[0], a[1]+b[1], a[2]+b[2] };
    return d;
}

void frep_get_global_transform(frep_t *node,
                              frep_mat3_t *R_root_to_this,
                              frep_vec3_t *T_this_rel_root,
                              frep_mat3_t R_root_to_parent,
                              frep_vec3_t T_parent_rel_root) {
    float cx = cosf(-node->rx); float sx = sinf(-node->rx);
    float cy = cosf(-node->ry); float sy = sinf(-node->ry);
    float cz = cosf(-node->rz); float sz = sinf(-node->rz);

    //    R_this_to_parent = Rx(rx)*Ry(ry)*Rz(rz)
    // -> R_parent_to_this = Rz(-rz)*Ry(-ry)*Rx(-rx)
    frep_mat3_t R_parent_to_this =
    {
        cy*cz, cz*sx*sy - cx*sz, sx*sz + cx*cz*sy,
        cy*sz, cx*cz + sx*sy*sz, cx*sy*sz - cz*sx,
        -sy, cy*sx, cx*cy
    };
    frep_vec3_t T_this_rel_parent = { node->tx, node->ty, node->tz };

    *R_root_to_this = frep_mat_mul(R_parent_to_this,R_root_to_parent);
    *T_this_rel_root = frep_mat_add(T_parent_rel_root, frep_mat_mul_transpose(R_root_to_parent, T_this_rel_parent));
}

