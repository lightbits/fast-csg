#pragma once
#include "ast.h"

namespace ast_builder {

/*
FRep primitives
*/
ast_t *fBox(float width, float height, float depth) {
    ast_t *f = ast_calloc();
    f->opcode = AST_BOX;
    f->box.width = width;
    f->box.height = height;
    f->box.depth = depth;
    return f;
}
ast_t *fBoxCheap(float width, float height, float depth) {
    ast_t *f = ast_calloc();
    f->opcode = AST_BOX_CHEAP;
    f->box.width = width;
    f->box.height = height;
    f->box.depth = depth;
    return f;
}
ast_t *fSphere(float radius) {
    ast_t *f = ast_calloc();
    f->opcode = AST_SPHERE;
    f->sphere.radius = radius;
    return f;
}
ast_t *fCylinder(float radius, float height) {
    ast_t *f = ast_calloc();
    f->opcode = AST_CYLINDER;
    f->cylinder.radius = radius;
    f->cylinder.height = height;
    return f;
}
ast_t *fPlane(float sign, float offset) {
    ast_t *f = ast_calloc();
    f->opcode = AST_PLANE;
    f->plane.sign = sign;
    f->plane.offset = offset;
    return f;
}

/*
Function operators
*/
ast_t *fOpUnion(ast_t *left, ast_t *right) {
    ast_t *f = ast_calloc();
    f->opcode = AST_UNION;
    f->left = left;
    f->right = right;
    return f;
}
ast_t *fOpSubtract(ast_t *left, ast_t *right) {
    ast_t *f = ast_calloc();
    f->opcode = AST_SUBTRACT;
    f->left = left;
    f->right = right;
    return f;
}
ast_t *fOpIntersect(ast_t *left, ast_t *right) {
    ast_t *f = ast_calloc();
    f->opcode = AST_INTERSECT;
    f->left = left;
    f->right = right;
    return f;
}

/*
Spatial operators
*/
ast_t *pOpRotate(ast_t *f, float rx, float ry, float rz) {
    f->rx = rx;
    f->ry = ry;
    f->rz = rz;
    return f;
}
ast_t *pOpTranslate(ast_t *f, float tx, float ty, float tz) {
    f->tx = tx;
    f->ty = ty;
    f->tz = tz;
    return f;
}
