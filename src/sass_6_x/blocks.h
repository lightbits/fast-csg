#pragma once

namespace backend_sass {

#define CLEAR()           memset(&block->instructions[block->num_instructions], 0, sizeof(instruction_t))
#define TYPE(Expression)  block->instructions[block->num_instructions].type = INSTRUCTION_##Expression
#define RA(Expression)    block->instructions[block->num_instructions].a = REGISTER_##Expression
#define RB(Expression)    block->instructions[block->num_instructions].b = REGISTER_##Expression
#define RC(Expression)    block->instructions[block->num_instructions].c = REGISTER_##Expression
#define RD(Expression)    block->instructions[block->num_instructions].d = REGISTER_##Expression
#define STALL(Expression) block->instructions[block->num_instructions].stall = Expression;
#define IMMB(Expression)  block->instructions[block->num_instructions].imm_b = Expression;
#define NEXT()            block->num_instructions++; assert(block->num_instructions <= MAX_INSTRUCTIONS_PER_BLOCK);

#if 0 // sequential transform code
// (x,y,z) = R_root_to_this*((x0,y0,z0) - T_this_rel_root)
//         = Rz(rz)*Ry(ry)*Rx(rx)*((x0-tx, y0-ty, z0-tz))
void emit_transform(instruction_block_t *block, frep_mat3_t R_root_to_this, frep_vec3_t T_this_rel_root)
{
    // Convert to final rotation into euler angles
    // (need less registers to do three sequential
    // euler rotations, than a full 3x3 matrix multiply, I think...?)
    float rx,ry,rz;
    frep_so3_to_ypr(R_root_to_this, &rz, &ry, &rx);
    float tx = T_this_rel_root[0];
    float ty = T_this_rel_root[1];
    float tz = T_this_rel_root[2];
    float cx = cosf(rx); float sx = sinf(rx);
    float cy = cosf(ry); float sy = sinf(ry);
    float cz = cosf(rz); float sz = sinf(rz);
                                                                     // translate:
    CLEAR(); TYPE(FADD20I); RD(X); RA(X0); IMMB(-tx);        NEXT(); // FADD x, x0, (-tx)
    CLEAR(); TYPE(FADD20I); RD(Y); RA(Y0); IMMB(-ty);        NEXT(); // FADD y, y0, (-ty)
    CLEAR(); TYPE(FADD20I); RD(Z); RA(Z0); IMMB(-tz);        NEXT(); // FADD z, z0, (-tz)
                                                                     // rotate_x: x=x, y=c*y - s*z, z=s*y + c*z
    CLEAR(); TYPE(FMUL20I); RD(W); RA(Y);  IMMB(+sx);        NEXT(); // FMUL w, y, (s)
    CLEAR(); TYPE(FMUL20I); RD(Y); RA(Y);  IMMB(+cx);        NEXT(); // FMUL y, y.reuse, (c)
    CLEAR(); TYPE(FFMA20I); RD(Y); RA(Z);  IMMB(-sx); RC(Y); NEXT(); // FFMA y, z, (-s), y
    CLEAR(); TYPE(FFMA20I); RD(Z); RA(Z);  IMMB(+cx); RC(W); NEXT(); // FFMA z, z.reuse, (c), w
                                                                     // rotate_y: x=c*x + s*z, y=y, z=-s*x + c*z
    CLEAR(); TYPE(FMUL20I); RD(W); RA(X);  IMMB(-sy);        NEXT(); // FMUL w, x, (-s)
    CLEAR(); TYPE(FMUL20I); RD(X); RA(X);  IMMB(+cy);        NEXT(); // FMUL x, x.reuse, (c)
    CLEAR(); TYPE(FFMA20I); RD(X); RA(Z);  IMMB(+sy); RC(X); NEXT(); // FFMA x, z, (s), x
    CLEAR(); TYPE(FFMA20I); RD(Z); RA(Z);  IMMB(+cy); RC(W); NEXT(); // FFMA z, z.reuse, (c), w
                                                                     // rotate_z: x=c*x - s*y, y=s*x + c*y, z=z
    CLEAR(); TYPE(FMUL20I); RD(W); RA(X);  IMMB(+sz);        NEXT(); // FMUL w, x, (s)
    CLEAR(); TYPE(FMUL20I); RD(X); RA(X);  IMMB(+cz);        NEXT(); // FMUL x, x.reuse, (c)
    CLEAR(); TYPE(FFMA20I); RD(X); RA(Y);  IMMB(-sz); RC(X); NEXT(); // FFMA x, y, (-s), x
    CLEAR(); TYPE(FFMA20I); RD(Y); RA(Y);  IMMB(+cz); RC(W); NEXT(); // FFMA y, y.reuse, (c), w
}
#else
void emit_transform(instruction_block_t *block, frep_mat3_t R/*_root_to_this*/, frep_vec3_t T/*_this_rel_root*/)
{
    // This path is a stall-count optimized version of the above.
    // The generated code computes the following:
    // (x,y,z) = R_root_to_this*((x0,y0,z0) - T_this_rel_root)
    // x = R00*(x0-Tx) + R01*(y0-Ty) + R02*(z0-Tz)
    //   = R00*x0 + R01*y0 + R02*z0 + (-R00*Tx - R01*Ty - R02*Tz)
    //   = R00*x0 + R01*y0 + R02*z0 + dx
    // etc...

    float dx = -(R.at(0,0)*T[0] + R.at(0,1)*T[1] + R.at(0,2)*T[2]);
    float dy = -(R.at(1,0)*T[0] + R.at(1,1)*T[1] + R.at(1,2)*T[2]);
    float dz = -(R.at(2,0)*T[0] + R.at(2,1)*T[1] + R.at(2,2)*T[2]);

    CLEAR(); TYPE(FADD20I); RD(X); RA(RZ); IMMB(dx);               STALL(1); NEXT(); // 1 FADD x, RZ, dx
    CLEAR(); TYPE(FADD20I); RD(Y); RA(RZ); IMMB(dy);               STALL(1); NEXT(); // 1 FADD y, RZ, dy
    CLEAR(); TYPE(FADD20I); RD(Z); RA(RZ); IMMB(dz);               STALL(4); NEXT(); // 4 FADD z, RZ, dz
    CLEAR(); TYPE(FFMA20I); RD(X); RA(X0); IMMB(R.at(0,0)); RC(X); STALL(1); NEXT(); // 1 FFMA x, x0, (R00), x // Q) Why not have dx here?
    CLEAR(); TYPE(FFMA20I); RD(Y); RA(X0); IMMB(R.at(1,0)); RC(Y); STALL(1); NEXT(); // 1 FFMA y, x0, (R10), y
    CLEAR(); TYPE(FFMA20I); RD(Z); RA(X0); IMMB(R.at(2,0)); RC(Z); STALL(4); NEXT(); // 4 FFMA z, x0, (R20), z
    CLEAR(); TYPE(FFMA20I); RD(X); RA(Y0); IMMB(R.at(0,1)); RC(X); STALL(1); NEXT(); // 1 FFMA x, y0, (R01), x
    CLEAR(); TYPE(FFMA20I); RD(Y); RA(Y0); IMMB(R.at(1,1)); RC(Y); STALL(1); NEXT(); // 1 FFMA y, y0, (R11), y
    CLEAR(); TYPE(FFMA20I); RD(Z); RA(Y0); IMMB(R.at(2,1)); RC(Z); STALL(4); NEXT(); // 4 FFMA z, y0, (R21), z
    CLEAR(); TYPE(FFMA20I); RD(X); RA(Z0); IMMB(R.at(0,2)); RC(X); STALL(1); NEXT(); // 1 FFMA x, z0, (R02), x
    CLEAR(); TYPE(FFMA20I); RD(Y); RA(Z0); IMMB(R.at(1,2)); RC(Y); STALL(1); NEXT(); // 1 FFMA y, z0, (R12), y
    CLEAR(); TYPE(FFMA20I); RD(Z); RA(Z0); IMMB(R.at(2,2)); RC(Z); STALL(4); NEXT(); // 4 FFMA z, z0, (R22), z
}
#endif

// cylinder: max(sqrt(x*x + z*z) - R, abs(y)-H)
void emit_cylinder(instruction_block_t *block, frep_mat3_t R, frep_vec3_t T, float r, float h)
{
    emit_transform(block, R, T);
    CLEAR(); TYPE(FMUL);          RD(W); RA(X); RB(X);        NEXT(); // FMUL w, x, x
    CLEAR(); TYPE(FFMA);          RD(W); RA(Z); RB(Z); RC(W); NEXT(); // FFMA w, z, z, w
    CLEAR(); TYPE(SQRT);          RD(W); RA(W); RB(W);        NEXT(); // SQRT w, w
    CLEAR(); TYPE(FADD20I_ABS_A); RD(Y); RA(Y); IMMB(-h);     NEXT(); // FADD y, |y|, -H
    CLEAR(); TYPE(FADD20I);       RD(W); RA(W); IMMB(-r);     NEXT(); // FADD w, w, -R
    CLEAR(); TYPE(FMAX);          RD(D); RA(W); RB(Y);        NEXT(); // FMAX d, w, y
}

// sphere: sqrt(x*x + y*y + z*z) - R
void emit_sphere(instruction_block_t *block, frep_mat3_t R, frep_vec3_t T, float r)
{
    #if 1
    CLEAR(); TYPE(FADD20I); RD(X); RA(X0); IMMB(-T[0]);  STALL(1); NEXT(); // 1 FADD x, x0, (-tx)
    CLEAR(); TYPE(FADD20I); RD(Y); RA(Y0); IMMB(-T[1]);  STALL(1); NEXT(); // 1 FADD y, y0, (-ty)
    CLEAR(); TYPE(FADD20I); RD(Z); RA(Z0); IMMB(-T[2]);  STALL(4); NEXT(); // 4 FADD z, z0, (-tz)
    CLEAR(); TYPE(FMUL);    RD(W); RA(X); RB(X);                   NEXT(); // 6 FMUL w, x, x
    CLEAR(); TYPE(FFMA);    RD(W); RA(Y); RB(Y); RC(W);            NEXT(); // 6 FFMA w, y, y, w
    CLEAR(); TYPE(FFMA);    RD(W); RA(Z); RB(Z); RC(W);            NEXT(); // 6 FFMA w, z, z, w
    CLEAR(); TYPE(SQRT);    RD(W); RA(W); RB(W);                   NEXT(); // 8 SQRT w, w
    CLEAR(); TYPE(FADD20I); RD(D); RA(W); IMMB(-r);                NEXT(); // 6 FADD d, w, -R
    #else
    emit_transform(block, R, T);
    CLEAR(); TYPE(FMUL);    RD(W); RA(X); RB(X);        NEXT(); // FMUL w, x, x
    CLEAR(); TYPE(FFMA);    RD(W); RA(Y); RB(Y); RC(W); NEXT(); // FFMA w, y, y, w
    CLEAR(); TYPE(FFMA);    RD(W); RA(Z); RB(Z); RC(W); NEXT(); // FFMA w, z, z, w
    CLEAR(); TYPE(SQRT);    RD(W); RA(W); RB(W);        NEXT(); // SQRT w, w
    CLEAR(); TYPE(FADD20I); RD(D); RA(W); IMMB(-r);     NEXT(); // FADD d, w, -R
    #endif
}

void emit_box(instruction_block_t *block, frep_mat3_t R, frep_vec3_t T, float bx, float by, float bz)
{
    assert(false && "fBox is not implemented yet");
}

// box: max(max(|x|-wx, |y|-wy), |z|-wz)
void emit_box_cheap(instruction_block_t *block, frep_mat3_t R, frep_vec3_t T, float bx, float by, float bz)
{
    emit_transform(block, R, T);
    CLEAR(); TYPE(FADD20I_ABS_A); RD(X); RA(X); IMMB(-bx); STALL(1); NEXT(); // 1 FADD x, |x|, -wx
    CLEAR(); TYPE(FADD20I_ABS_A); RD(Y); RA(Y); IMMB(-by); STALL(1); NEXT(); // 1 FADD y, |y|, -wy
    CLEAR(); TYPE(FADD20I_ABS_A); RD(Z); RA(Z); IMMB(-bz); STALL(5); NEXT(); // 5 FADD z, |z|, -wz
    CLEAR(); TYPE(FMAX);          RD(W); RA(X); RB(Y);               NEXT(); // 6 FMAX w, x, y
    CLEAR(); TYPE(FMAX);          RD(D); RA(W); RB(Z);               NEXT(); // 6 FMAX d, w, z
}

void emit_plane(instruction_block_t *block, frep_mat3_t R, frep_vec3_t T, float px)
{
    #if 0
    // optimized version
    float rx,ry,rz;
    frep_so3_to_ypr(R, &rz, &ry, &rx);
    float cx = cosf(rx); float sx = sinf(rx);
    float cy = cosf(ry); float sy = sinf(ry);
    float cz = cosf(rz); float sz = sinf(rz);
    float rtx = -((cy*cz)*T[0] + (cz*sx*sy - cx*sz)*T[1] + (sx*sz + cx*cz*sy)*T[2]);

    CLEAR(); TYPE(FMUL20I); RD(X); RA(X0); IMMB((cy*cz));                   NEXT(); // 6 FMUL x, x0, (cy*cz)
    CLEAR(); TYPE(FFMA20I); RD(X); RA(Y0); IMMB((cz*sx*sy-cx*sz));   RC(X); NEXT(); // 6 FFMA x, y0, (cz*sx*sy-cx*sz), x
    CLEAR(); TYPE(FFMA20I); RD(X); RA(Z0); IMMB((sx*sz + cx*cz*sy)); RC(X); NEXT(); // 6 FFMA x, z0, (sx*sz + cx*cz*sy), x
    CLEAR(); TYPE(FADD20I); RD(D); RA(X);  IMMB(rtx-px);                    NEXT(); // 6 FADD d, x, rtx-px
    #else
    emit_transform(block, R, T);
                                                    // plane: x - px
    CLEAR(); TYPE(FADD20I); RD(D); RA(X); IMMB(-px); NEXT(); // FADD d, x, -px
    #endif
}

void emit_union(instruction_block_t *block)     { CLEAR(); TYPE(FMIN); RD(D); RA(D_LEFT); RB(D_RIGHT);       NEXT(); }
void emit_intersect(instruction_block_t *block) { CLEAR(); TYPE(FMAX); RD(D); RA(D_LEFT); RB(D_RIGHT);       NEXT(); }
void emit_subtract(instruction_block_t *block)  { CLEAR(); TYPE(FMAX_NEG_B); RD(D); RA(D_LEFT); RB(D_RIGHT); NEXT(); }
void emit_blend(instruction_block_t *block, float alpha)
{
                                                                        // blend: alpha*d_left + (1-alpha)*d_right
    CLEAR(); TYPE(FMUL20I); RD(D); RA(D_LEFT);  IMMB(alpha);             NEXT(); // FMUL d, d_left, (alpha)
    CLEAR(); TYPE(FFMA20I); RD(D); RA(D_RIGHT); IMMB(1.0f-alpha); RC(D); NEXT(); // FFMA d, d_right, (1-alpha), d
}

#undef TYPE
#undef RA
#undef RB
#undef RC
#undef RD
#undef IMMB
#undef NEXT
#undef STALL
#undef CLEAR

void _generate_blocks(
    instruction_blocks_t *s,
    frep_t *node,
    int destination=0,
    frep_mat3_t R_root_to_parent=frep_identity_3x3,
    frep_vec3_t T_parent_rel_root=frep_null_3x1)
// You can do much smarter register allocation here. The register allocation
// may also need to change if we do smarter scheduling. E.g. block reordering.
{
    assert(node);

    frep_mat3_t R_root_to_this;
    frep_vec3_t T_this_rel_root;
    frep_get_global_transform(node, &R_root_to_this, &T_this_rel_root, R_root_to_parent, T_parent_rel_root);

    if (frep_is_boolean(node))
    {
        assert(node->left);
        assert(node->right);

        int d_left = destination;
        int d_right = destination+1;
        _generate_blocks(s, node->left, d_left, R_root_to_this, T_this_rel_root);
        _generate_blocks(s, node->right, d_right, R_root_to_this, T_this_rel_root);

        instruction_block_t *b = &s->blocks[s->num_blocks++];
        b->num_instructions = 0;
        b->d_left = d_left;
        b->d_right = d_right;
        b->d = destination;
             if (node->opcode == FREP_UNION)     emit_union(b);
        else if (node->opcode == FREP_INTERSECT) emit_intersect(b);
        else if (node->opcode == FREP_SUBTRACT)  emit_subtract(b);
        else if (node->opcode == FREP_BLEND)     emit_blend(b, node->blend.alpha);
        assert(s->num_blocks <= MAX_INSTRUCTION_BLOCKS);
    }
    else if (frep_is_primitive(node))
    {
        instruction_block_t *b = &s->blocks[s->num_blocks++];
        b->num_instructions = 0;
        frep_mat3_t R = R_root_to_this;
        frep_vec3_t T = T_this_rel_root;
        b->d = destination;
             if (node->opcode == FREP_BOX)       emit_box(b, R, T, node->box.width, node->box.height, node->box.depth);
        else if (node->opcode == FREP_BOX_CHEAP) emit_box_cheap(b, R, T, node->box.width, node->box.height, node->box.depth);
        else if (node->opcode == FREP_SPHERE)    emit_sphere(b, R, T, node->sphere.radius);
        else if (node->opcode == FREP_CYLINDER)  emit_cylinder(b, R, T, node->cylinder.radius, node->cylinder.height);
        else if (node->opcode == FREP_PLANE)     emit_plane(b, R, T, node->plane.offset);
        assert(s->num_blocks <= MAX_INSTRUCTION_BLOCKS);
    }
    else
    {
        assert(false && "Unexpected node type");
    }
}

instruction_blocks_t generate_blocks(frep_t *node)
// This function generates a list of instruction blocks that evaluates the
// tree and stores the resulting distance value in register[0]. Each block
// is assigned registers during the recursive tree parsing.
{
    assert(node);

    static instruction_block_t _blocks[MAX_INSTRUCTION_BLOCKS];
    instruction_blocks_t s = {0};
    s.blocks = _blocks;
    s.num_blocks = 0;

    _generate_blocks(&s, node);

    return s;
}

}
