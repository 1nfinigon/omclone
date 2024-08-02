#pragma endian little

import std.sys;
import type.LEB128;

struct LengthDelimited {
    type::LEB128 length;
    char data[length];
};

bitfield AllowedParts {
    ALLOW_ARM                 : 1; // 0
    ALLOW_MULTIARM_EQUILIBRIUM: 1; // 1
    ALLOW_PISTON              : 1; // 2
    ALLOW_TRACK               : 1; // 3
    padding                   : 4; // 4-7
    ALLOW_BONDER              : 1; // 8
    ALLOW_UNBONDER            : 1; // 9
    ALLOW_MULTIBONDER         : 1; // 10
    ALLOW_TRIPLEX_BONDER      : 1; // 11
    ALLOW_CALCIFICATION       : 1; // 12
    ALLOW_DUPLICATION         : 1; // 13
    ALLOW_PROJECTION          : 1; // 14
    ALLOW_PURIFICATION        : 1; // 15
    ALLOW_ANIMISMUS           : 1; // 16
    ALLOW_DISPOSAL            : 1; // 17
    ALLOW_QUINTESSENCE_GLYPHS : 1; // 18
    padding                   : 3; // 19-21
    ALLOW_DROP_AND_ROTATION   : 1; // 22
    ALLOW_GRAB                : 1; // 23
    ALLOW_RESET               : 1; // 24
    ALLOW_REPEAT_NOOP         : 1; // 25
    ALLOW_PIVOT               : 1; // 26
    padding                   : 1;
    ALLOW_VAN_BERLO           : 1; // 28
    padding                   : 35; // 29-63
};

enum AtomType : u8 {
    ConduitSpace = 0,
    Salt = 1,
    Air = 2,
    Earth = 3,
    Fire = 4,
    Water = 5,
    Quicksilver = 6,
    Gold = 7,
    Silver = 8,
    Copper = 9,
    Iron = 10,
    Tin = 11,
    Lead = 12,
    Vitae = 13,
    Mors = 14,
    RepeatingOutputMarker = 15,
    Quintessence = 16,
};

struct S8Pos {
    s8 x;
    s8 y;
};

struct Atom {
    AtomType atomType;
    S8Pos pos;
};

bitfield BondType {
    normal : 1;
    triplex_r : 1;
    triplex_k : 1;
    triplex_y : 1;
    conduit : 1;
};

struct Bond {
    BondType type;
    S8Pos from;
    S8Pos to;
};

struct Molecule {
    u32 numAtoms;
    Atom atoms[numAtoms];
    
    u32 numBonds;
    Bond bonds[numBonds];
};

struct Puzzle {
    u32 format;
    std::assert_warn(format == 3, "Unsupported format");
    LengthDelimited puzzleName;
    u64 creatorID;
    AllowedParts allowedParts;
    
    u32 numInputs;
    Molecule inputs[numInputs];
    
    u32 numOutputs;
    Molecule outputs[numOutputs];
    
    u32 outputMultiplier;
    bool isProduction;
};

Puzzle puzzle @ 0x00;