#pragma once
#include "ast.h"

/*
FRep primitives
*/
frep_t *fBox(float width, float height, float depth) {
    frep_t *f = frep_calloc();
    f->opcode = AST_BOX;
    f->box.width = width;
    f->box.height = height;
    f->box.depth = depth;
    return f;
}
frep_t *fBoxCheap(float width, float height, float depth) {
    frep_t *f = frep_calloc();
    f->opcode = AST_BOX_CHEAP;
    f->box.width = width;
    f->box.height = height;
    f->box.depth = depth;
    return f;
}
frep_t *fSphere(float radius) {
    frep_t *f = frep_calloc();
    f->opcode = AST_SPHERE;
    f->sphere.radius = radius;
    return f;
}
frep_t *fCylinder(float radius, float height) {
    frep_t *f = frep_calloc();
    f->opcode = AST_CYLINDER;
    f->cylinder.radius = radius;
    f->cylinder.height = height;
    return f;
}
frep_t *fPlane(float sign, float offset) {
    frep_t *f = frep_calloc();
    f->opcode = AST_PLANE;
    f->plane.sign = sign;
    f->plane.offset = offset;
    return f;
}

/*
Function operators
*/
frep_t *fOpUnion(frep_t *left, frep_t *right) {
    frep_t *f = frep_calloc();
    f->opcode = AST_UNION;
    f->left = left;
    f->right = right;
    return f;
}
frep_t *fOpSubtract(frep_t *left, frep_t *right) {
    frep_t *f = frep_calloc();
    f->opcode = AST_SUBTRACT;
    f->left = left;
    f->right = right;
    return f;
}
frep_t *fOpIntersect(frep_t *left, frep_t *right) {
    frep_t *f = frep_calloc();
    f->opcode = AST_INTERSECT;
    f->left = left;
    f->right = right;
    return f;
}

/*
Spatial operators
*/
frep_t *pOpRotate(frep_t *f, float rx, float ry, float rz) {
    f->rx = rx;
    f->ry = ry;
    f->rz = rz;
    return f;
}
frep_t *pOpTranslate(frep_t *f, float tx, float ty, float tz) {
    f->tx = tx;
    f->ty = ty;
    f->tz = tz;
    return f;
}
