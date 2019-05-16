#pragma once
#include "frep.h"
#include <assert.h>
#include <math.h>

float frep_eval(frep_t *f, float x, float y, float z)
{
    assert(f);

    x -= f->tx;
    y -= f->ty;
    z -= f->tz;

    if (f->rx != 0.0f)
    {
        float cx = cosf(-f->rx);
        float sx = sinf(-f->rx);
        float zz = cx*z + sx*y;
        y = cx*y - sx*z;
        z = zz;
    }
    if (f->ry != 0.0f)
    {
        float cy = cosf(-f->ry);
        float sy = sinf(-f->ry);
        float xx = cy*x + sy*z;
        z = cy*z - sy*x;
        x = xx;
    }
    if (f->rz != 0.0f)
    {
        float cz = cosf(-f->rz);
        float sz = sinf(-f->rz);
        float xx = cz*x - sz*y;
        y = cz*y + sz*x;
        x = xx;
    }

    switch (f->opcode)
    {
        case FREP_BOX:
        {
            float dx = fabsf(x) - f->box.width;
            float dy = fabsf(y) - f->box.height;
            float dz = fabsf(z) - f->box.depth;
            float dbx = (dx < 0.0f) ? dx : 0.0f; float b = dbx;
            float dby = (dy < 0.0f) ? dy : 0.0f; if (dby > b) b = dby;
            float dbz = (dz < 0.0f) ? dz : 0.0f; if (dbz > b) b = dbz;
            if (dx < 0.0f) dx = 0.0f;
            if (dy < 0.0f) dy = 0.0f;
            if (dz < 0.0f) dz = 0.0f;
            return sqrtf(dx*dx + dy*dy + dz*dz) + b;
        }
        case FREP_BOX_CHEAP:
        {
            float dx = fabsf(x) - f->box.width;
            float dy = fabsf(y) - f->box.height;
            float dz = fabsf(z) - f->box.depth;
            float d = dx;
            if (dy > d) d = dy;
            if (dz > d) d = dz;
            return d;
        }
        case FREP_SPHERE:
        {
            return sqrtf(x*x + y*y + z*z) - f->sphere.radius;
        }
        case FREP_CYLINDER:
        {
            float a = sqrtf(x*x + z*z) - f->cylinder.radius;
            float b = fabsf(y) - f->cylinder.height;
            return a > b ? a : b;
        }
        case FREP_PLANE:
        {
            return f->plane.sign*x - f->plane.offset;
        }
        case FREP_UNION:
        {
            float f1 = frep_eval(f->left, x, y, z);
            float f2 = frep_eval(f->right, x, y, z);
            return f1 < f2 ? f1 : f2;
        }
        case FREP_INTERSECT:
        {
            float f1 = frep_eval(f->left, x, y, z);
            float f2 = frep_eval(f->right, x, y, z);
            return f1 > f2 ? f1 : f2;
        }
        case FREP_SUBTRACT:
        {
            float f1 = frep_eval(f->left, x, y, z);
            float f2 = -frep_eval(f->right, x, y, z);
            return f1 > f2 ? f1 : f2;
        }
        #if 0
        case FREP_BLEND:
        {
            float f1 = frep_eval(f->left, x, y, z);
            float f2 = frep_eval(f->right, x, y, z);
            return f->blend.alpha*f1 + (1.0f - f->blend.alpha)*f2;
        }
        #endif
        default:
        {
            assert(false && "invalid node type");
        }
    }
    return 0.0f;
}
