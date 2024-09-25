use crate::search;
use crate::sim::BasicInstr;
use eyre::Result;
use std::io::{Read, Write};

pub struct Item {
    pub playouts: [u32; BasicInstr::N_TYPES],
}

pub struct History(Vec<Item>);

fn read_u32_le<R: Read>(reader: &mut R) -> Result<u32> {
    let mut buffer = [0u8; std::mem::size_of::<u32>()];
    reader.read_exact(&mut buffer)?;
    Ok(u32::from_le_bytes(buffer))
}

fn write_u32_le<W: Write>(writer: &mut W, value: u32) -> Result<()> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

impl History {
    pub fn new() -> Self {
        Self(Vec::new())
    }

    pub fn read<R: Read>(r: &mut R) -> Result<Self> {
        let version = read_u32_le(r)?;
        match version {
            0 => {
                let len = read_u32_le(r)? as usize;
                assert!(len < 100000, "unreasonable len");
                let mut history = Vec::with_capacity(len);
                for i in 0..len {
                    let mut playouts = [0u32; BasicInstr::N_TYPES];
                    for instr in 0..BasicInstr::N_TYPES {
                        playouts[instr] = read_u32_le(r)?;
                    }
                    history.push(Item { playouts });
                }
                Ok(History(history))
            }
            _ => panic!("version number {} unsupported", version),
        }
    }

    pub fn write<W: Write>(&self, w: &mut W) -> Result<()> {
        write_u32_le(w, 0)?;
        write_u32_le(w, self.0.len().try_into().unwrap())?;
        for item in self.0.iter() {
            for &playout in item.playouts.iter() {
                write_u32_le(w, playout)?;
            }
        }
        Ok(())
    }

    pub fn append(&mut self, stats: &search::NextUpdatesWithStats) {
        let playouts: Vec<u32> = stats.0.iter().map(|s| s.visits).collect();
        let item = Item {
            playouts: playouts.try_into().unwrap(),
        };
        self.0.push(item);
    }
}
