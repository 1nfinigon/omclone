#pragma endian little

import std.sys;
import type.LEB128;

struct LengthDelimited {
    type::LEB128 length;
    char data[length];
};

struct S32Pos {
    s32 x;
    s32 y;
};

enum Instruction : u8 {
    ROTATE_CW = 'R',
    ROTATE_CCW = 'r',
    EXTEND = 'E',
    RETRACT = 'e',
    GRAB = 'G',
    DROP = 'g',
    PIVOT_CW = 'P',
    PIVOT_CCW = 'p',
    FORWARD = 'A',
    BACKWARD = 'a',
    REPEAT = 'C',
    RESET = 'X',
    NOOP = 'O',
    EMPTY = ' ',
};

struct PartInstruction {
    s32 index;
    Instruction instruction;
};

struct Part {
    LengthDelimited name;
    u8 const1;
    std::assert_warn(const1 == 1, "unexpected const");
    S32Pos pos;
    u32 size;
    s32 rotation;
    u32 which_input_or_output;
    
    u32 numInstructions;
    PartInstruction instructions[numInstructions];
    
    if (name.data == "track") {
        u32 numTrackHexes;
        S32Pos trackHexes[numTrackHexes];
    }
    u32 armNumber;
    if (name.data == "pipe") {
        u32 conduitId;
        u32 numConduitHexes;
        S32Pos conduitHexes[numConduitHexes];
    }
};

struct Solution {
    u32 format;
    std::assert_warn(format == 7, "Unsupported format");
    LengthDelimited puzzleName;
    LengthDelimited solutionName;
    u32 isSolved;
    if (isSolved) {
        u32 const0;
        u32 cycles;
        u32 const1;
        u32 cost;
        u32 const2;
        u32 area;
        u32 const3;
        u32 instructions;
        std::assert_warn(const0 == 0 && const1 == 1 && const2 == 2 && const3 == 3, "unexpected constant fields");
    }
    
    u32 numParts;
    Part parts[numParts];
};

Solution solution @ 0x00;