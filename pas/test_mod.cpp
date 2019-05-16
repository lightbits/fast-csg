// modf implementation

// #include <math_functions.h> and see how fmodf(x,y) is implemented in ptx and cubin
// It's very ugly

// I'm not sure what the CRT is doing here, as fmodf(-0.1f, 1.3f) = -0.1f. In GLSL,
// they define mod(x,y) = x - y*floor(x/y). I think floor(x/y) can be implemented in
// PTX as div.rnd.rm???
#include <math.h>
int main()
{
    /*
    x = i*m + r, 0 <= r < m
    x mod m = r

    x / m = i + r/m
    floor(x/m) = i // want to preserve this calculation to do count-limited repetition
    r = x - floor(x/m)*m
    */

    float m = 1.3f;
    float x = -1.4f;
    printf("%f %f\n", x - floorf(x/m)*m, fmodf(x, m));
}
