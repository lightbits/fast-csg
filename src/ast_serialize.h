#pragma once
#include "ast.h"
#include <stdio.h>

#ifdef _MSC_VER
// Note: MSVC version returns -1 on overflow, but glibc returns total count (which may be >= buf_size)
#define snprintf _snprintf
#endif

static char *ast__to_string(ast_t *a, char *stream, size_t sizeof_buffer)
{
    if (!a) return stream;
    if      (a->type == AST_BOX)       stream += snprintf(stream, sizeof_buffer, "b[%g,%g,%g]", a->box.w, a->box.h, a->box.d);
    else if (a->type == AST_SPHERE)    stream += snprintf(stream, sizeof_buffer, "s[%g]", a->sphere.r);
    else if (a->type == AST_CYLINDER)  stream += snprintf(stream, sizeof_buffer, "c[%g,%g]", a->cylinder.r, a->cylinder.h);
    else if (a->type == AST_PLANE)     stream += snprintf(stream, sizeof_buffer, "p[%g]", a->plane.x);
    else if (a->type == AST_UNION)     stream += snprintf(stream, sizeof_buffer, "U");
    else if (a->type == AST_INTERSECT) stream += snprintf(stream, sizeof_buffer, "I");
    else if (a->type == AST_SUBTRACT)  stream += snprintf(stream, sizeof_buffer, "S");
    else if (a->type == AST_BLEND)     stream += snprintf(stream, sizeof_buffer, "B[%g]", a->blend.alpha);
    stream += snprintf(stream, sizeof_buffer, "[%g,%g,%g]", a->rx, a->ry, a->rz);
    stream += snprintf(stream, sizeof_buffer, "[%g,%g,%g]", a->tx, a->ty, a->tz);
    stream = ast__to_string(a->left, stream, sizeof_buffer);
    stream = ast__to_string(a->right, stream, sizeof_buffer);
    return stream;
}

static ast_t *ast__from_string(char **inout_stream)
{
    char *stream = *inout_stream;
    if (!stream) return NULL;
    if (*stream == '\0') return NULL;

    ast_t *a = ast_new();

    #define next_bracket() { while (*stream && *stream != '[') stream++; assert(*stream); stream++; assert(*stream); }
    if      (*stream == 'b') { a->type = AST_BOX;       next_bracket(); assert(3 == sscanf(stream, "%f,%f,%f", &a->box.w,      &a->box.h, &a->box.d)); next_bracket(); }
    else if (*stream == 's') { a->type = AST_SPHERE;    next_bracket(); assert(1 == sscanf(stream, "%f",       &a->sphere.r                        )); next_bracket(); }
    else if (*stream == 'c') { a->type = AST_CYLINDER;  next_bracket(); assert(2 == sscanf(stream, "%f,%f",    &a->cylinder.r, &a->cylinder.h      )); next_bracket(); }
    else if (*stream == 'p') { a->type = AST_PLANE;     next_bracket(); assert(1 == sscanf(stream, "%f",       &a->plane.x                         )); next_bracket(); }
    else if (*stream == 'U') { a->type = AST_UNION;     next_bracket(); }
    else if (*stream == 'I') { a->type = AST_INTERSECT; next_bracket(); }
    else if (*stream == 'S') { a->type = AST_SUBTRACT;  next_bracket(); }
    else if (*stream == 'B') { a->type = AST_BLEND;     next_bracket(); assert(1 == sscanf(stream, "%f",       &a->blend.alpha                     )); next_bracket(); }
    else assert(false && "invalid node type");
    assert(3 == sscanf(stream, "%f,%f,%f", &a->rx, &a->ry, &a->rz));
    next_bracket();
    assert(3 == sscanf(stream, "%f,%f,%f", &a->tx, &a->ty, &a->tz));
    while (*stream && *stream != ']') stream++;
    assert(*stream);
    stream++;
    #undef next_bracket

    a->left = ast__from_string(&stream);
    a->right = ast__from_string(&stream);
    *inout_stream = stream;
    return a;
}

char *ast_to_string(ast_t *a)
{
    static char buffer[1024*1024];
    ast__to_string(a, buffer, sizeof(buffer));
    return buffer;
}

ast_t *ast_from_string(char *stream)
{
    return ast__from_string(&stream);
}

#ifdef _MSC_VER
#undef snprintf
#endif
