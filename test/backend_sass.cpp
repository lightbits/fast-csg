#define COMPUTE_CAPABILITY_6_X
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "../src/frep.h"
#include "../src/frep_builder.h"
#include "../src/frep_eval.h"
#include "../src/backend_sass.h"
#include "util/sass_simulator.h"

float frep_eval_sass(float x0, float y0, float z0, instruction_t *instructions, int num_instructions, bool debug=false)
{
    static sass_simulator_t sim = {0};
    sim.init(debug);
    sim.reg[REGISTER_X0] = x0;
    sim.reg[REGISTER_Y0] = y0;
    sim.reg[REGISTER_Z0] = z0;
    for (int i = 0; i < num_instructions; i++)
        sim.execute(instructions[i]);
    return sim.reg[REGISTER_D+0];
}

void run_test(int test_number, frep_t *tree)
{
    instruction_blocks_t blocks = generate_sass_blocks(tree);

    int num_instructions;
    instruction_t *instructions = schedule_blocks(blocks, &num_instructions);

    printf("///////////////////////////////////////////////////\n");
    printf("                 test number %d\n", test_number);

    frep_eval_sass(0.0f,0.0f,0.0f, instructions, num_instructions, true);

    for (int i = -4; i <= 4; i++)
    for (int j = -4; j <= 4; j++)
    for (int k = -4; k <= 4; k++)
    {
        float x0 = i/4.0f;
        float y0 = j/4.0f;
        float z0 = k/4.0f;
        float f_sass = frep_eval_sass(x0,y0,z0, instructions, num_instructions);
        float f_true = frep_eval(tree, x0,y0,z0);
        if (fabsf(f_sass - f_true) > 0.00001f)
        {
            printf("\nEvaluation mismatch!\n");
            printf("true: f(%.2f,%.2f,%.2f) = %f\n", x0,y0,z0,f_true);
            printf("sass: f(%.2f,%.2f,%.2f) = %f\n", x0,y0,z0,f_sass);
            exit(1);
        }
    }
    printf("ok!\n");
}

int main()
{
    frep_t *tree;

    tree = fBoxCheap(0.9f,0.6f,0.3f);
    run_test(0, tree);

    tree = fSphere(0.3f);
    run_test(1, tree);

    tree = fCylinder(0.6f,0.3f);
    run_test(2, tree);

    tree = fPlane(1.0f, 0.3f);
    pOpRotate(tree, 0.3f,0.5f,0.4f);
    pOpTranslate(tree, 0.2f,0.5f,0.4f);
    run_test(3, tree);

    frep_t *d1 = fBoxCheap(1.0f,0.5f,0.25f);
    pOpRotate(d1, 0.1f,0.4f,0.3f);
    pOpTranslate(d1, 0.5f,0.25f,0.25f);
    frep_t *d2 = fSphere(0.8f);
    pOpTranslate(d2, 1.0f,0,0);
    frep_t *d3 = fCylinder(0.4f, 0.2f);
    pOpTranslate(d3, 1.0f, 1.0f, 0.3f);
    tree = fOpUnion(fOpUnion(d1, d2), d3);
    run_test(4, tree);
}
