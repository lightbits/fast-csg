#pragma once
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

typedef int ast_opcode_t;
enum ast_opcode_ {
    AST_INVALID = 0,

    AST_BOX,
    AST_BOX_CHEAP,
    AST_SPHERE,
    AST_CYLINDER,
    AST_PLANE,
    AST_UNION,
    AST_INTERSECT,
    AST_SUBTRACT,
    AST_BLEND,
};

struct ast_box_t      { float width, height, depth; };
struct ast_sphere_t   { float radius; };
struct ast_cylinder_t { float radius, height; };
struct ast_plane_t    { float sign, offset; };
struct ast_blend_t    { float alpha; };

/*
Each frep node has a rigid-body transform associated with it.
It can be the identity. If so, it gets optimized out in the
backend. The transformation parameters relate the point argument
of the child node to its parent node by:

    p^parent = Rx(rx)*Ry(ry)*Rz(rz)*p^child + (tx,ty,tz)

*/
struct ast_t {
    ast_opcode_t opcode;
    ast_t *left;
    ast_t *right;
    float rx,ry,rz,tx,ty,tz;
    union {
        ast_box_t box;
        ast_sphere_t sphere;
        ast_cylinder_t cylinder;
        ast_plane_t plane;
        ast_blend_t blend;
    };
};

/*
Node creation and deletion utilities
*/
ast_t *ast_malloc() {
    ast_t *f = (ast_t*)malloc(sizeof(ast_t));
    return f;
}
ast_t *ast_calloc() {
    ast_t *f = (ast_t*)calloc(1, sizeof(ast_t));
    return f;
}
void ast_free(ast_t *f) {
    if (!f) return;
    ast_free(f->left);
    ast_free(f->right);
    free(f);
}
ast_t *ast_copy(ast_t *f) {
    if (!f) return NULL;
    ast_t *f1 = ast_malloc();
    *f1 = *f;
    f1->left = ast_copy(f->left);
    f1->right = ast_copy(f->right);
    return f1;
}

/*
Other utilities
*/
bool ast_is_primitive(ast_t *f) {
    return f->opcode == AST_BOX ||
           f->opcode == AST_BOX_CHEAP ||
           f->opcode == AST_SPHERE ||
           f->opcode == AST_CYLINDER ||
           f->opcode == AST_PLANE;
}
bool ast_is_boolean(ast_t *f) {
    return f->opcode == AST_UNION ||
           f->opcode == AST_INTERSECT ||
           f->opcode == AST_SUBTRACT;
}
int ast_get_num_nodes(ast_t *f) {
    if (!f) return 0;
    return 1 + ast_get_num_nodes(f->left) + ast_get_num_nodes(f->right);
}

int ast_get_depth(ast_t *f) {
    if (!f) return 0;
    int l = ast_get_depth(f->left);
    int r = ast_get_depth(f->right);
    int max_lr = (l > r ? l : r);
    return 1 + max_lr;
}
ast_t *ast_find_node(ast_t *a, int find_i, ast_t **out_parent, int *out_depth, ast_t *parent=NULL, int depth=0)
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
    else if (ast_is_boolean(a))
    {
        ast_t *left = ast_find_node(a->left, find_i, out_parent, out_depth, a, depth+1);
        if (left) return left;
        ast_t *right = ast_find_node(a->right, find_i, out_parent, out_depth, a, depth+1);
        if (right) return right;
    }
    return NULL;
}

/*
Utility routines for computing rigid-body transform from root node to a specific child.
*/
struct ast_mat3_t { float d[3*3]; float &at(int row, int col) { return d[col + row*3]; } };
struct ast_vec3_t { float d[3]; float &operator[](int i) { return d[i]; } };
static ast_mat3_t ast_identity_3x3 = { 1,0,0, 0,1,0, 0,0,1 };
static ast_vec3_t ast_null_3x1 = { 0,0,0 };

// d = a*b
ast_mat3_t ast_mat_mul(ast_mat3_t a, ast_mat3_t b) {
    ast_mat3_t d = {0};
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
ast_vec3_t ast_mat_mul_transpose(ast_mat3_t a, ast_vec3_t b) {
    ast_vec3_t d = {0};
    for (int row = 0; row < 3; row++)
    {
        d[row] = 0.0f;
        for (int i = 0; i < 3; i++)
            d[row] += a.at(i,row)*b[i];
    }
    return d;
}

ast_vec3_t ast_mat_add(ast_vec3_t a, ast_vec3_t b) {
    ast_vec3_t d = { a[0]+b[0], a[1]+b[1], a[2]+b[2] };
    return d;
}

void ast_get_global_transform(ast_t *node,
                              ast_mat3_t *R_root_to_this,
                              ast_vec3_t *T_this_rel_root,
                              ast_mat3_t R_root_to_parent,
                              ast_vec3_t T_parent_rel_root) {
    float cx = cosf(-node->rx); float sx = sinf(-node->rx);
    float cy = cosf(-node->ry); float sy = sinf(-node->ry);
    float cz = cosf(-node->rz); float sz = sinf(-node->rz);

    //    R_this_to_parent = Rx(rx)*Ry(ry)*Rz(rz)
    // -> R_parent_to_this = Rz(-rz)*Ry(-ry)*Rx(-rx)
    ast_mat3_t R_parent_to_this =
    {
        cy*cz, cz*sx*sy - cx*sz, sx*sz + cx*cz*sy,
        cy*sz, cx*cz + sx*sy*sz, cx*sy*sz - cz*sx,
        -sy, cy*sx, cx*cy
    };
    ast_vec3_t T_this_rel_parent = { node->tx, node->ty, node->tz };

    *R_root_to_this = ast_mat_mul(R_parent_to_this,R_root_to_parent);
    *T_this_rel_root = ast_mat_add(T_parent_rel_root, ast_mat_mul_transpose(R_root_to_parent, T_this_rel_parent));
}

