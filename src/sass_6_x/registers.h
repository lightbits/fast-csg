#pragma once

namespace backend_sass {

enum named_register_t
{
    // used to indicate immediate values
    NO_REGISTER=0,

    // input position is stored in these
    REGISTER_X0=0,
    REGISTER_Y0,
    REGISTER_Z0,

    // an instruction block is allocated these by the scheduler
    REGISTER_X,
    REGISTER_Y,
    REGISTER_Z,
    REGISTER_W,
    REGISTER_D,       // result is to be stored here
    REGISTER_D_LEFT,  // result from left child in SDF tree is stored here
    REGISTER_D_RIGHT, // result from right child in SDF tree is stored here

    // constant zero
    REGISTER_RZ=0xff
};

}
