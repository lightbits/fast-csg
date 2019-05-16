#pragma once
#include "sdf_builder.h"

sdf_node_t *model_simple01() { return sdf_box(1.0f, 0.5f, 0.25f); }
sdf_node_t *model_simple02() { return sdf_cylinder(1.0f, 0.5f); }
sdf_node_t *model_simple03() { return sdf_sphere(0.98f); }
sdf_node_t *model_simple04() { return sdf_plane(0.98f); }
sdf_node_t *model_simple05() { return sdf_rotate(sdf_translate(sdf_box(0.98f, 0.63f, 0.33f), 0.1f,-0.2f,0.3f), 0.1f,0.2f,-0.3f); }
sdf_node_t *model_simple06() { return sdf_rotate(sdf_translate(sdf_sphere(0.98f),            0.1f,-0.2f,0.3f), 0.1f,0.2f,-0.3f); }
sdf_node_t *model_simple07() { return sdf_rotate(sdf_translate(sdf_cylinder(0.98f, 0.63f),   0.1f,-0.2f,0.3f), 0.1f,0.2f,-0.3f); }
sdf_node_t *model_simple08() { return sdf_rotate(sdf_translate(sdf_plane(0.98f),             0.1f,-0.2f,0.3f), 0.1f,0.2f,-0.3f); }
sdf_node_t *model_simple09() { return sdf_blend(0.4f, sdf_sphere(1.0f), sdf_cylinder(0.3f,1.0f)); }
sdf_node_t *model_simple10() {
    sdf_node_t *d1 = sdf_box(0.98f, 0.63f, 0.33f);
        sdf_rotate(d1, -0.3f, 0.2f, -0.1f);
        sdf_translate(d1, 0.3f, -0.5f, 0.3f);
    sdf_node_t *d2 = sdf_sphere(0.63f);
        sdf_rotate(d2, 0.7f, 0.8f, -0.3f);
        sdf_translate(d2, -0.6f, +0.5f, 0.2f);
    sdf_node_t *d = sdf_union(d1, d2);
    return d;
}
sdf_node_t *model_simple11() { return sdf_subtract(sdf_box(1.0f,1.0f,1.0f), sdf_translate(sdf_sphere(0.5f), 0,1.0f,0)); }
sdf_node_t *model_simple12() { return sdf_subtract(sdf_rotate(sdf_box(1.0f,1.0f,1.0f), 0.77f,0.77f,0), sdf_sphere(0.5f)); }
sdf_node_t *model_simple13() { return sdf_subtract(sdf_box(1.0f,1.0f,1.0f), sdf_cylinder(0.5f,2.0f)); }
sdf_node_t *model_simple14() { return sdf_union(sdf_box(0.5f,0.5f,0.5f), sdf_translate(sdf_sphere(0.25f),0.5f,0,0)); }
sdf_node_t *model_simple15() { return sdf_intersect(sdf_box(0.5f,0.5f,0.5f), sdf_translate(sdf_sphere(0.25f),0.5f,0,0)); }
sdf_node_t *model_simple16() { return sdf_subtract(sdf_box(0.5f,0.5f,0.5f), sdf_translate(sdf_sphere(0.25f),0.5f,0,0)); }

sdf_node_t *model_complex_2d_1()
{
    auto *d1 = sdf_translate(sdf_box(0.9f, 0.1f, 0.5f), 0.0f, 0.5f, 0.0f);
    auto *d2 = sdf_translate(sdf_box(0.8f, 0.05f, 0.5f), 0.0f, -0.5f, 0.0f);
    auto *d3 = sdf_sphere(0.5f);
    auto *d4 = sdf_box(1.0f, 0.2f, 0.5f);
    return sdf_rotate(sdf_translate(sdf_union(sdf_union(d1, d2), sdf_subtract(d3, d4)), 0.1f, -0.2f, 0.0f), 0.0f, 0.0f, 0.2f);
}

sdf_node_t *model_complex02()
{
    sdf_node_t *a1 = sdf_plane(0.3f);
    sdf_node_t *a2 = sdf_cylinder(0.2f, 0.3f);
    sdf_node_t *a3 = sdf_box(0.3f,0.3f,0.3f);
    sdf_node_t *a4 = sdf_sphere(0.5f);
    sdf_node_t *a5 = sdf_union(a1,a2);
    sdf_node_t *a6 = sdf_subtract(a3,a4);
    sdf_node_t *a7 = sdf_union(a5,a6);
    sdf_node_t *b1 = sdf_plane(0.3f);
    sdf_node_t *b2 = sdf_cylinder(0.2f, 0.3f);
    sdf_node_t *b3 = sdf_box(0.3f,0.3f,0.3f);
    sdf_node_t *b4 = sdf_sphere(0.5f);
    sdf_node_t *b5 = sdf_union(b1,b2);
    sdf_node_t *b6 = sdf_subtract(b3,b4);
    sdf_node_t *b7 = sdf_union(b5,b6);
    sdf_node_t *d = sdf_union(a7,b7);
    return d;
}

sdf_node_t *model_complex03()
{
    float s = 0.3f;
    sdf_node_t *d1 = sdf_sphere(1.0f*s);
    sdf_node_t *c1 = sdf_cylinder(0.54f*s,1.2f*s); sdf_rotate(c1, 0,0,0);
    sdf_node_t *c2 = sdf_cylinder(0.54f*s,1.2f*s); sdf_rotate(c2, 1.54f,0,0);
    sdf_node_t *c3 = sdf_cylinder(0.54f*s,1.2f*s); sdf_rotate(c3, 0,0,1.54f);
    sdf_node_t *c12 = sdf_union(c1,c2);
    sdf_node_t *c123 = sdf_union(c12,c3);
    sdf_node_t *d2 = sdf_subtract(d1,c123);

    sdf_node_t *b1 = sdf_box(0.74f*s,0.74f*s,0.74f*s);
    sdf_node_t *d3 = sdf_intersect(d2,b1);

    sdf_node_t *s2 = sdf_sphere(0.3f*s);
    sdf_node_t *c5 = sdf_cylinder(0.1f*s, 0.8f*s); sdf_rotate(c5, 1.54f,0,0);
    sdf_node_t *c6 = sdf_cylinder(0.1f*s, 0.8f*s); sdf_rotate(c6, 0,0,0);
    sdf_node_t *c56 = sdf_union(c5,c6); sdf_rotate(c56, 0.7f, 0.0f, 0.0f); sdf_translate(c56, 1.0f*s, 0.0f, 0.0f);
    sdf_node_t *s2c56 = sdf_union(s2,c56);
    sdf_node_t *d4 = sdf_union(d3, s2c56);

    sdf_node_t *b2 = sdf_box(0.2f*s,0.2f*s,0.2f*s); sdf_translate(b2,-1.0f*s,0,0); sdf_rotate(b2,0.77f,0.77f,0);
    sdf_node_t *d5 = sdf_union(d4,b2);

    sdf_node_t *d = d5;
    return d;
}

sdf_node_t *model_complex04()
{
    float s = 1.5f;
    sdf_node_t *d1 = sdf_sphere(1.0f*s);
    sdf_node_t *d2 = sdf_cylinder(0.54f*s,1.2f*s); sdf_rotate(d2, 0,0,0);
    sdf_node_t *d3 = sdf_cylinder(0.54f*s,1.2f*s); sdf_rotate(d3, 1.54f,0,0);
    sdf_node_t *d4 = sdf_cylinder(0.54f*s,1.2f*s); sdf_rotate(d4, 0,0,1.54f);
    sdf_node_t *d5 = sdf_subtract(d1,d2);
    sdf_node_t *d6 = sdf_subtract(d5,d3);
    sdf_node_t *d7 = sdf_subtract(d6,d4);
    sdf_node_t *d8 = sdf_plane(0.74f*s); sdf_rotate(d8, 0,0,1.54f);
    sdf_node_t *d9 = sdf_plane(0.74f*s); sdf_rotate(d9, 0,0,-1.54f);
    sdf_node_t *d10 = sdf_plane(0.74f*s); sdf_rotate(d10, 0,1.54f,0);
    sdf_node_t *d11 = sdf_plane(0.74f*s); sdf_rotate(d11, 0,-1.54f,0);
    sdf_node_t *d12 = sdf_plane(0.74f*s); sdf_rotate(d12, 0,0,0);
    sdf_node_t *d13 = sdf_plane(0.74f*s); sdf_rotate(d13, 0,0,3.14f);
    sdf_node_t *d14 = sdf_intersect(d7,d8);
    sdf_node_t *d15 = sdf_intersect(d14,d9);
    sdf_node_t *d16 = sdf_intersect(d15,d10);
    sdf_node_t *d17 = sdf_intersect(d16,d11);
    sdf_node_t *d18 = sdf_intersect(d17,d12);
    sdf_node_t *d19 = sdf_intersect(d18,d13);
    sdf_node_t *d20 = sdf_sphere(0.3f*s);
    sdf_node_t *d21 = sdf_union(d19, d20);
    sdf_node_t *d22 = sdf_cylinder(0.1f*s, 0.8f*s); sdf_rotate(d22, 1.54f,0,0);
    sdf_node_t *d23 = sdf_cylinder(0.1f*s, 0.8f*s); sdf_rotate(d23, 0,0,0);
    sdf_node_t *d24 = sdf_union(d21, d22);
    sdf_node_t *d25 = sdf_union(d24, d23);
    return d25;
}

sdf_node_t *model_complex05()
{
    float s = 1.5f;
    sdf_node_t *d1 = sdf_sphere(1.0f*s);
    sdf_node_t *d2 = sdf_cylinder(0.54f*s,1.2f*s); sdf_rotate(d2, 0,0,0);
    sdf_node_t *d3 = sdf_subtract(d1,d2);
    sdf_node_t *d4 = sdf_plane(0.44f*s); sdf_rotate(d4, 0,0,1.54f);
    sdf_node_t *d5 = sdf_plane(0.44f*s); sdf_rotate(d5, 0,0,-1.54f);
    sdf_node_t *d6 = sdf_intersect(d3,d4);
    sdf_node_t *d7 = sdf_intersect(d6,d5);
    return d7;
}

sdf_node_t *model_chair1_2d()
{
    float k = 0.5f;
    sdf_node_t *seat = sdf_box(1.0f*k, 0.1f*k, 1.0f);
    sdf_node_t *leg1 = sdf_rotate(sdf_translate(sdf_box(0.1f*k, 1.0f*k, 1.0f), -1.0f*k,0,0), 0,0,-0.2f);
    sdf_node_t *leg2 = sdf_rotate(sdf_translate(sdf_box(0.1f*k, 1.0f*k, 1.0f), +1.0f*k,0,0), 0,0,+0.1f);
    sdf_node_t *legs = sdf_translate(sdf_union(leg1, leg2), 0,-1.0f*k,0);
    sdf_node_t *back = sdf_rotate(sdf_translate(sdf_box(0.1f*k, 1.0f*k, 1.0f), 1.0f*k,1.0f*k,0), 0,0,-0.1f);
    sdf_node_t *seat_and_legs = sdf_union(seat, legs);
    sdf_node_t *chair = sdf_union(seat_and_legs, back);
    return chair;
}

sdf_node_t *model_chair2_2d()
{
    float k = 0.5f;
    sdf_node_t *seat = sdf_rotate(sdf_box(0.8f*k, 0.15f*k, 1.0f), 0,0,0.2f);
    sdf_node_t *leg1 = sdf_rotate(sdf_translate(sdf_box(0.1f*k, 1.0f*k, 1.0f), -0.75f*k,0,0), 0,0,-0.05f);
    sdf_node_t *leg2 = sdf_rotate(sdf_translate(sdf_box(0.1f*k, 1.0f*k, 1.0f), +0.8f*k,0.05f*k,0), 0,0,0.1f);
    sdf_node_t *mid = sdf_translate(sdf_box(0.8f*k, 0.05f*k, 1.0f), 0,-1.0f*k,0);
    sdf_node_t *legs = sdf_intersect(sdf_translate(sdf_union(leg1, leg2), 0,-1.0f*k,0),
                                     sdf_rotate(sdf_plane(1.9f*k), 0,0,-3.14f/2.0f));
    sdf_node_t *seat_and_legs = sdf_union(seat, legs);
    sdf_node_t *chair = sdf_union(seat_and_legs, mid);
    return chair;
}

sdf_node_t *model_translated_sphere()
{
    return
    sdf_translate(sdf_sphere(1.0f), -0.5f,0.0f,0.0f);
}

sdf_node_t *model_intersection()
{
    return
    sdf_intersect(sdf_translate(sdf_sphere(0.5f), -0.2f,0.0f,0.0f),
                  sdf_translate(sdf_sphere(0.5f), +0.2f,0.0f,0.0f));
}

sdf_node_t *model_two_spheres()
{
    return
    sdf_union(sdf_translate(sdf_sphere(0.1f), -0.5f,0.0f,0.0f),
              sdf_translate(sdf_sphere(0.5f), +0.3f,0.0f,0.0f));
}

sdf_node_t *model_two_spheres_equal()
{
    return
    sdf_union(sdf_translate(sdf_sphere(0.3f), -0.4f,0.0f,0.0f),
              sdf_translate(sdf_sphere(0.3f), +0.4f,0.0f,0.0f));
}

sdf_node_t *model_four_spheres()
{
    return
    sdf_union(
              sdf_union(
                        sdf_translate(sdf_sphere(0.2f), 0.0f,0.7f,0.0f),
                        sdf_translate(sdf_sphere(0.2f), 0.0f,-0.7f,0.0f)),
              sdf_union(
                        sdf_translate(sdf_sphere(0.4f), -0.5f,0.0f,0.0f),
                        sdf_translate(sdf_sphere(0.4f), +0.5f,0.0f,0.0f)));
}

sdf_node_t *model_scissor()
{
    return
    sdf_union(
              sdf_translate(sdf_sphere(0.4f), 0.0f,0.6f,0.0f),
              sdf_intersect(
                        sdf_translate(sdf_sphere(0.8f), -0.5f,0.0f,0.0f),
                        sdf_translate(sdf_sphere(0.8f), +0.5f,0.0f,0.0f)));
}

sdf_node_t *model_fillet()
{
    return
    sdf_union
    (
        sdf_translate(sdf_sphere(0.25f), 0.25f,0.25f,0.0f),
        sdf_intersect
        (
            sdf_rotate(sdf_plane(0.53f), 0.0f,0.0f,3.1415f/4.0f),
            sdf_box(0.5f, 0.5f, 0.5f)
        )
    );
}

sdf_node_t *model_two_box()
{
    return
    sdf_union
    (
        sdf_translate(sdf_box(0.55f,0.05f,1.0f), 0.25f,0.5f,0.0f),
        sdf_translate(sdf_box(0.05f,0.55f,1.0f), -0.25f,0.0f,0.0f)
    );
}

sdf_node_t *model_two_box_unequal()
{
    return
    sdf_union
    (
        sdf_translate(sdf_box(0.35f,0.05f,1.0f), 0.15f,0.5f,0.0f),
        sdf_translate(sdf_box(0.05f,0.55f,1.0f), -0.25f,0.0f,0.0f)
    );
}

sdf_node_t *model_offset_box()
{
    return sdf_rotate(sdf_translate(sdf_box(0.5f,0.5f,0.5f), 0.2f, -0.2f, 0.0f), 0.0f, 0.0f, -0.5f);
}

sdf_node_t *model_motion0(int which)
{
    if (which == 0) {
        auto *d1 = sdf_box(0.3f, 0.3f, 0.3f);
        auto *d2 = sdf_box(0.2f, 0.2f, 0.2f);
        d2 = sdf_rotate(sdf_translate(d2, +0.3f, +0.2f, 0.0f), 0.0f, 0.0f, 0.3f);
        auto *d5 = sdf_union(d1, d2);
        d5 = sdf_rotate(d5, 0.0f, 0.0f, 0.2f);
        d5 = sdf_translate(d5, 0.45f, -0.5f, 0.0f);
        auto *d6 = sdf_sphere(0.3f);
        d6 = sdf_translate(d6, -0.4f, +0.2f, 0.0);
        return sdf_union(d5, d6);
    } else {
        auto *d1 = sdf_box(0.3f, 0.3f, 0.3f);
        auto *d2 = sdf_box(0.2f, 0.2f, 0.2f);
        d2 = sdf_rotate(sdf_translate(d2, +0.3f, +0.2f, 0.0f), 0.0f, 0.0f, 0.3f);
        auto *d5 = sdf_union(d1, d2);
        d5 = sdf_rotate(d5, 0.0f, 0.0f, 0.2f);
        d5 = sdf_rotate(sdf_translate(d5, 0.45f, -0.1f, 0.0f), 0.0f, 0.0f, -0.3f);
        auto *d6 = sdf_sphere(0.3f);
        d6 = sdf_translate(d6, -0.4f, +0.2f, 0.0);
        return sdf_union(d5, d6);
    }
}

sdf_node_t *model_motion1(int which)
{
    if (which == 0) {
        auto *d1 = sdf_box(0.3f, 0.3f, 0.3f);
        auto *d2 = sdf_box(0.2f, 0.2f, 0.2f);
        auto *d3 = sdf_box(0.2f, 0.2f, 0.2f);
        d2 = sdf_rotate(sdf_translate(d2, +0.3f, +0.2f, 0.0f), 0.0f, 0.0f, 0.3f);
        d3 = sdf_rotate(sdf_translate(d3, -0.3f, -0.2f, 0.0f), 0.0f, 0.0f, 0.7f);
        auto *d4 = sdf_union(d2, d3);
        auto *d5 = sdf_union(d1, d4);
        d5 = sdf_rotate(d5, 0.0f, 0.0f, 0.2f);
        d5 = sdf_translate(d5, 0.4f, -0.2f, 0.0f);
        auto *d6 = sdf_sphere(0.3f);
        d6 = sdf_translate(d6, -0.3f, +0.2f, 0.0);
        return sdf_union(d5, d6);
    } else {
        auto *d1 = sdf_box(0.3f, 0.3f, 0.3f);
        auto *d2 = sdf_box(0.2f, 0.2f, 0.2f);
        auto *d3 = sdf_box(0.2f, 0.2f, 0.2f);
        d2 = sdf_rotate(sdf_translate(d2, +0.3f, +0.2f, 0.0f), 0.0f, 0.0f, 0.3f);
        d3 = sdf_rotate(sdf_translate(d3, -0.3f, -0.2f, 0.0f), 0.0f, 0.0f, 0.7f);
        auto *d4 = sdf_union(d2, d3);
        auto *d5 = sdf_union(d1, d4);
        d5 = sdf_rotate(d5, 0.0f, 0.0f, 0.2f);
        d5 = sdf_translate(d5, 0.45f, -0.1f, 0.0f);
        auto *d6 = sdf_sphere(0.3f);
        d6 = sdf_translate(d6, -0.3f, +0.2f, 0.0);
        return sdf_union(d5, d6);
    }
}
