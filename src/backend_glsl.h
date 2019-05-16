#include "sdf.h"
#if 1
char *sdf_codegen_glsl(sdf_node_t *node,
                       char *stream=NULL,
                       int *return_my_index=NULL,
                       sdf_mat3_t R_root_to_parent=sdf_identity_3x3,
                       sdf_vec3_t T_parent_rel_root=sdf_null_3x1)
{
    assert(node);

    static int next_index = 0;
    static char buffer[1024*1024];
    if (!return_my_index)
    {
        next_index = 1;
        stream = buffer;
    }
    int my_index = next_index++;

    sdf_mat3_t R_root_to_this;
    sdf_vec3_t T_this_rel_root;
    sdf_compute_global_transform(node, &R_root_to_this, &T_this_rel_root, R_root_to_parent, T_parent_rel_root);

    // p^this = R_root_to_this*(p^0 - T_this_rel_root)
    //        = R_root_to_this*p^0 + (-R_root_to_this*T_this_rel_root)
    {
        #define R(row,col) R_root_to_this.at(row,col)
        #define T(i) T_this_rel_root[i]
        float dtx = -(R(0,0)*T(0) + R(0,1)*T(1) + R(0,2)*T(2));
        float dty = -(R(1,0)*T(0) + R(1,1)*T(1) + R(1,2)*T(2));
        float dtz = -(R(2,0)*T(0) + R(2,1)*T(1) + R(2,2)*T(2));
        stream += sprintf(stream,
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

    if (sdf_node_is_boolean(node))
    {
        assert(node->left);
        assert(node->right);

        int i_left,i_right;
        stream = sdf_codegen_glsl(node->left, stream, &i_left, R_root_to_this, T_this_rel_root);
        stream = sdf_codegen_glsl(node->right, stream, &i_right, R_root_to_this, T_this_rel_root);

        stream += sprintf(stream, "float d%d = ", my_index);

        if (node->type == sdf_node_union)     stream += sprintf(stream, "min(d%d,d%d);\n", i_left, i_right);
        if (node->type == sdf_node_intersect) stream += sprintf(stream, "max(d%d,d%d);\n", i_left, i_right);
        if (node->type == sdf_node_subtract)  stream += sprintf(stream, "max(d%d,-d%d);\n", i_left, i_right);
        if (node->type == sdf_node_blend)     stream += sprintf(stream, "%f*d%d + %f*d%d;\n", node->blend.alpha, i_left, 1.0f-node->blend.alpha, i_right);
    }
    else if (sdf_node_is_primitive(node))
    {
        stream += sprintf(stream, "float d%d = ", my_index);

        #if defined(SDF_APPROX_BOX_EVAL)
        if (node->type == sdf_node_box)      stream += sprintf(stream, "max(max(abs(p%d.x)-%f, abs(p%d.y)-%f), abs(p%d.z)-%f);\n", my_index, node->box.w, my_index, node->box.h, my_index, node->box.d);
        #elif defined(SDF_EXACT_BOX_EVAL)
        if (node->type == sdf_node_box)      stream += sprintf(stream, "box(p%d, vec3(%f,%f,%f));\n", my_index, node->box.w, node->box.h, node->box.d);
        #else
        #error "You must define either SDF_APPROX_BOX_EVAL or SDF_EXACT_BOX_EVAL."
        #endif
        if (node->type == sdf_node_sphere)   stream += sprintf(stream, "length(p%d) - %f;\n", my_index, node->sphere.r);
        if (node->type == sdf_node_cylinder) stream += sprintf(stream, "max(length(p%d.xz) - %f, abs(p%d.y) - %f);\n", my_index, node->cylinder.r, my_index, node->cylinder.h);
        if (node->type == sdf_node_plane)    stream += sprintf(stream, "p%d.x - %f;\n", my_index, node->plane.x);
    }
    else
    {
        assert(false && "Unexpected node type");
    }

    if (return_my_index)
    {
        *return_my_index = my_index;
        return stream;
    }
    else
    {
        return buffer;
    }
}
#else
// This one computes transforms differently; instead of pre-computing the absolute transform
// from root to child (as above) it composes transformations from parent to child iteratively.
// This could lead to more pipeline stalls as the instructions are sequentially dependent.
char *sdf_codegen_glsl(sdf_node_t *root, int level=0, sdf_node_t *parent=NULL, char *stream=NULL, int parent_p=0, int *return_d_this=NULL)
{
    static int num_p;
    static int num_d;
    static char buffer[1024*1024];
    if (level == 0)
    {
        stream = buffer;
        num_p = 1;
        num_d = 1;
    }

    assert(root);
    bool has_rotate = root->rx != 0.0f || root->ry != 0.0f || root->rz != 0.0f;
    bool has_translate = root->tx != 0.0f || root->ty != 0.0f || root->tz != 0.0f;
    bool has_transform = has_rotate || has_translate;

    if (has_transform)
    {
        stream += sprintf(stream, "vec3 p%d = p%d;\n", num_p, parent_p);
        parent_p = num_p++;

        if (has_translate)
            stream += sprintf(stream, "p%d -= vec3(%f,%f,%f);\n", parent_p, root->tx, root->ty, root->tz);

        // implements p.zy = cos(-rx)*p.zy + sin(-rx)*vec2(p.y, -p.z)
        if (root->rx != 0.0f)
            stream += sprintf(stream, "p%d.zy = %f*p%d.zy + %f*vec2(p%d.y, -p%d.z);\n", parent_p, cosf(-root->rx), parent_p, sinf(-root->rx), parent_p, parent_p);

        // implements p.xz = cos(-ry)*p.xz + sin(-ry)*vec2(p.z, -p.x)
        if (root->ry != 0.0f)
            stream += sprintf(stream, "p%d.xz = %f*p%d.xz + %f*vec2(p%d.z, -p%d.x);\n", parent_p, cosf(-root->ry), parent_p, sinf(-root->ry), parent_p, parent_p);

        // implements p.xy = cos(-rz)*p.xy + sin(-rz)*vec2(-p.y, p.x)
        if (root->rz != 0.0f)
            stream += sprintf(stream, "p%d.xy = %f*p%d.xy + %f*vec2(-p%d.y, p%d.x);\n", parent_p, cosf(-root->rz), parent_p, sinf(-root->rz), parent_p, parent_p);
    }

    int d_this = num_d++;
    if (return_d_this)
        *return_d_this = d_this;

    if (sdf_node_is_boolean(root))
    {
        assert(root->left);
        assert(root->right);

        int d_left,d_right;
        stream = sdf_codegen_glsl(root->left, level+1, root, stream, parent_p, &d_left);
        stream = sdf_codegen_glsl(root->right, level+1, root, stream, parent_p, &d_right);

        stream += sprintf(stream, "float d%d = ", d_this);

        if (root->type == sdf_node_union)     stream += sprintf(stream, "min(d%d,d%d);\n", d_left, d_right);
        if (root->type == sdf_node_intersect) stream += sprintf(stream, "max(d%d,d%d);\n", d_left, d_right);
        if (root->type == sdf_node_subtract)  stream += sprintf(stream, "max(d%d,-d%d);\n", d_left, d_right);
        if (root->type == sdf_node_blend)     stream += sprintf(stream, "%f*d%d + %f*d%d;\n", blend_alpha, d_left, 1.0f-blend_alpha, d_right);
    }
    else if (sdf_node_is_primitive(root))
    {
        stream += sprintf(stream, "float d%d = ", d_this);

        if (root->type == sdf_node_box)           stream += sprintf(stream, "box(p%d, vec3(%f,%f,%f));\n", parent_p, root->box.w, root->box.h, root->box.d);
        else if (root->type == sdf_node_sphere)   stream += sprintf(stream, "sphere(p%d, %f);\n", parent_p, root->sphere.r);
        else if (root->type == sdf_node_cylinder) stream += sprintf(stream, "cylinder(p%d, %f,%f);\n", parent_p, root->cylinder.r, root->cylinder.h);
        else if (root->type == sdf_node_plane)    stream += sprintf(stream, "p%d.x - %f;\n", parent_p, root->plane.x);
    }
    else
    {
        assert(false && "Unexpected node type");
    }

    if (level == 0)
        return buffer;
    else
        return stream;
}
#endif
