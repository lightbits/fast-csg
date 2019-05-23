#pragma once

namespace backend_sass {

enum named_register_t
{
    // This is used to indicate immediate values
    // Note: this enum must be 0 because we use memset to clear instructions
    NO_REGISTER=0,

    // Input position coordinates
    REGISTER_X0,
    REGISTER_Y0,
    REGISTER_Z0,

    // Temporary calculations
    REGISTER_X,
    REGISTER_Y,
    REGISTER_Z,
    REGISTER_W,

    // Result registers (e.g. f(p))
    REGISTER_D,       // result is to be stored here
    REGISTER_D_LEFT,  // result from left child in tree is stored here
    REGISTER_D_RIGHT, // result from right child in tree is stored here

    // constant zero
    REGISTER_RZ,
    NUM_NAMED_REGISTERS
};

}
