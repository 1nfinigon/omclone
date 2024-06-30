#pragma endian little

import type.LEB128;

struct LengthDelimited {
    type::LEB128 length;
    char data[length];
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
    LengthDelimited puzzleName;
    u64 creatorID;
    u64 allowedParts;
    
    u32 numInputs;
    Molecule inputs[numInputs];
    
    u32 numOutputs;
    Molecule outputs[numOutputs];
    
    u32 outputMultiplier;
    bool isProduction;
};

Puzzle puzzle @ 0x00;