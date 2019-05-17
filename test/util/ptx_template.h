#define STR(X) #X
const char *ptx_template = STR(
.version 6.0
.target sm_60
.address_size 64
.func (.reg.f32 f%d) tree(.reg.f32 x0, .reg.f32 y0, .reg.f32 z0) {
    .reg.f32 f<%d>;
    %s
    ret.uni;
}
.visible.entry main(.param.u64 param0, .param.u64 param1) {
    .reg.f32 x0;
    .reg.f32 y0;
    .reg.f32 z0;
    .reg.f32 w0;
    .reg.b32 r<5>;
    .reg.b64 rd<9>;
    .reg.f32 d;
    ld.param.u64 rd1, [param0];
    ld.param.u64 rd2, [param1];
    cvta.to.global.u64 rd3, rd2;
    cvta.to.global.u64 rd4, rd1;
    mov.u32 r1, tid.x;         // threadIdx.x
    mov.u32 r2, ctaid.x;       // blockIdx.x
    mov.u32 r3, ntid.x;        // blockDim.x
    mad.lo.s32 r4, r3, r2, r1; // blockDim.x*blockIdx.x + threadIdx.x
    mul.wide.s32 rd5, r4, 16;  // sizeof(vec4)*(blockDim.x*blockIdx.x + threadIdx.x)
    add.s64 rd6, rd4, rd5;     // param0 + sizeof(vec4)*(blockDim.x*blockIdx.x + threadIdx.x)
    ld.global.v4.f32 {x0, y0, z0, w0}, [rd6];
    mul.wide.s32 rd7, r4, 4;   // sizeof(float)*(blockDim.x*blockIdx.x + threadIdx.x)
    add.s64 rd8, rd3, rd7;     // param1 + sizeof(float)*(blockDim.x*blockIdx.x + threadIdx.x)
    call.uni (d), tree, (x0,y0,z0);
    st.global.f32 [rd8], d;
    ret;
}
);
#undef STR
