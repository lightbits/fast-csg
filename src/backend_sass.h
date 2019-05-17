#if defined(COMPUTE_CAPABILITY_3_X)
// Kepler
#error "Target devices of compute capability 3.x are not supported by the SASS backend."

#elif defined(COMPUTE_CAPABILITY_5_X) || defined(COMPUTE_CAPABILITY_6_X)
// Maxwell, Pascal (e.g. GTX 1080, Titan X)
#include "sass_6_x/backend.h"

#elif defined(COMPUTE_CAPABILITY_7_X)
// Volta, Turing (e.g. RTX Titan, 2080)
#error "Target devices of compute capability 7.x are not supported by the SASS backend."

#else
#error "Missing #define. Specify the compute capability target for the SASS backend."
#endif
