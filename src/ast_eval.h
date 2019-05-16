#pragma once
#include "ast.h"
#include <assert.h>
#include <math.h>

float ast_eval(ast_t *ast, float x, float y, float z)
{
    assert(ast);

    x -= ast->tx;
    y -= ast->ty;
    z -= ast->tz;

    if (ast->rx != 0.0f)
    {
        float cx = cosf(-ast->rx);
        float sx = sinf(-ast->rx);
        float zz = cx*z + sx*y;
        y = cx*y - sx*z;
        z = zz;
    }
    if (ast->ry != 0.0f)
    {
        float cy = cosf(-ast->ry);
        float sy = sinf(-ast->ry);
        float xx = cy*x + sy*z;
        z = cy*z - sy*x;
        x = xx;
    }
    if (ast->rz != 0.0f)
    {
        float cz = cosf(-ast->rz);
        float sz = sinf(-ast->rz);
        float xx = cz*x - sz*y;
        y = cz*y + sz*x;
        x = xx;
    }

    switch (ast->opcode)
    {
        case AST_BOX:
        {
            float dx = fabsf(x) - ast->box.width;
            float dy = fabsf(y) - ast->box.height;
            float dz = fabsf(z) - ast->box.depth;
            float dbx = (dx < 0.0f) ? dx : 0.0f; float b = dbx;
            float dby = (dy < 0.0f) ? dy : 0.0f; if (dby > b) b = dby;
            float dbz = (dz < 0.0f) ? dz : 0.0f; if (dbz > b) b = dbz;
            if (dx < 0.0f) dx = 0.0f;
            if (dy < 0.0f) dy = 0.0f;
            if (dz < 0.0f) dz = 0.0f;
            return sqrtf(dx*dx + dy*dy + dz*dz) + b;
        }
        case AST_BOX_CHEAP:
        {
            float dx = fabsf(x) - ast->box.width;
            float dy = fabsf(y) - ast->box.height;
            float dz = fabsf(z) - ast->box.depth;
            float d = dx;
            if (dy > d) d = dy;
            if (dz > d) d = dz;
            return d;
        }
        case AST_SPHERE:
        {
            return sqrtf(x*x + y*y + z*z) - ast->sphere.radius;
        }
        case AST_CYLINDER:
        {
            float a = sqrtf(x*x + z*z) - ast->cylinder.radius;
            float b = fabsf(y) - ast->cylinder.height;
            return a > b ? a : b;
        }
        case AST_PLANE:
        {
            return ast->plane.sign*x - ast->plane.offset;
        }
        case AST_UNION:
        {
            float a = ast_eval(ast->left, x, y, z);
            float b = ast_eval(ast->right, x, y, z);
            return a < b ? a : b;
        }
        case AST_INTERSECT:
        {
            float a = ast_eval(ast->left, x, y, z);
            float b = ast_eval(ast->right, x, y, z);
            return a > b ? a : b;
        }
        case AST_SUBTRACT:
        {
            float a = ast_eval(ast->left, x, y, z);
            float b = -ast_eval(ast->right, x, y, z);
            return a > b ? a : b;
        }
        #if 0
        case AST_BLEND:
        {
            float a = ast_eval(ast->left, x, y, z);
            float b = ast_eval(ast->right, x, y, z);
            return ast->blend.alpha*a + (1.0f-ast->blend.alpha)*b;
        }
        #endif
        default:
        {
            assert(false && "invalid node type");
        }
    }
    return 0.0f;
}
