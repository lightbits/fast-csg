#include "../src/backend_glsl.h"
#include "../src/frep_builder.h"

int main() {
    frep_t *f = fBoxCheap(1.0f, 0.5f, 0.25f);
    f = fOpUnion(f, fBox(2.0f, 1.0f, 1.0f));
    char *s = generate_glsl(f);
    printf("%s\n", s);
}
