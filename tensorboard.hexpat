#pragma author WerWolv
#pragma description Google Protobuf encoding 

import std.core;
import std.mem;

import type.leb128;

struct ZigZag32 {
	u32 value;
} [[sealed, format("format_zigzag32")]];

fn format_zigzag32(ZigZag32 zigzag) {
	return s32((s32(zigzag.value) << 1) ^ (s32(zigzag.value) >> 31));
};

struct ZigZag64 {
	u64 value;
} [[sealed, format("format_zigzag64")]];

fn format_zigzag64(ZigZag64 zigzag) {
	return s64((s64(zigzag.value) << 1) ^ (s64(zigzag.value) >> 63));
};

enum WireType : u8 {
	Varint 			= 0,
	_64Bit 			= 1,
	LengthDelimited = 2,
	StartGroup 		= 3,
	EndGroup 		= 4,
	_32Bit 			= 5
};

bitfield Key {
	field_number : 5;
	wire_type    : 3;
} [[bitfield_order(std::core::BitfieldOrder::MostToLeastSignificant, 8)]];

union _64Bit {
	u64 fixed64;
	ZigZag64 sfixed64;
	double dbl;
};

union _32Bit {
	u32 fixed32;
	ZigZag32 sfixed32;
	float flt;
};

struct LengthDelimited {
	type::LEB128 length;
	char data[length];
};


struct Entry {
	Key key;
	
	if (key.wire_type == WireType::Varint)
		type::LEB128 value;
	else if (key.wire_type == WireType::_64Bit)
		_64Bit value;
	else if (key.wire_type == WireType::LengthDelimited)
		LengthDelimited value;
	else if (key.wire_type == WireType::_32Bit)
		_32Bit value;
};

struct TBEntry {
    u64 length;
    u32 crc_length;
    u64 end = $ + length;
    Entry entry[while ($ < end)];
    u32 crc_data;
};

TBEntry entries[while(!std::mem::eof())] @ 0x00;