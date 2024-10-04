use crate::search;
use crate::sim::BasicInstr;
use enum_primitive_derive::Primitive;
use eyre::Result;
use num_traits::{FromPrimitive, ToPrimitive};
use std::io::{Read, Write};

#[derive(Primitive)]
pub enum Kind {
    Mcts = 0,
    FromOptimalSolution = 1,
}

pub struct Item {
    pub kind: Kind,
    pub playouts: [u32; BasicInstr::N_TYPES],
}

pub struct History(pub Vec<Item>);

fn read_f32_le<R: Read>(reader: &mut R) -> Result<f32> {
    let mut buffer = [0u8; std::mem::size_of::<f32>()];
    reader.read_exact(&mut buffer)?;
    Ok(f32::from_le_bytes(buffer))
}

fn write_f32_le<W: Write>(writer: &mut W, value: f32) -> Result<()> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

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

    pub fn append_mcts(&mut self, stats: &search::NextUpdatesWithStats) {
        let playouts: Vec<u32> = stats.updates_with_stats.iter().map(|s| s.visits).collect();
        let item = Item {
            kind: Kind::Mcts,
            playouts: playouts.try_into().unwrap(),
        };
        self.0.push(item);
    }

    pub fn append_from_optimal_solution(&mut self, instr: BasicInstr) {
        let mut playouts = [0u32; BasicInstr::N_TYPES];
        playouts[instr.to_usize().unwrap()] = 1;
        let item = Item {
            kind: Kind::FromOptimalSolution,
            playouts,
        };
        self.0.push(item);
    }
}

pub struct HistoryFile {
    pub solution_name: String,
    pub history: History,
    pub timestep_limit: u32,
    pub final_outcome: f32,
}

impl HistoryFile {
    pub fn read<R: Read>(r: &mut R) -> Result<Self> {
        let version = read_u32_le(r)?;
        match version {
            2 => {
                let solution_name = {
                    let length = read_u32_le(r)?;
                    let mut dat = vec![0u8; length as usize];
                    r.read_exact(&mut dat)?;
                    String::from_utf8(dat)?
                };
                let timestep_limit = read_u32_le(r)?;
                let final_outcome = read_f32_le(r)?;
                let len = read_u32_le(r)? as usize;
                assert!(len < 100000, "unreasonable len");
                let mut history = Vec::with_capacity(len);
                for _ in 0..len {
                    let mut playouts = [0u32; BasicInstr::N_TYPES];
                    let kind = Kind::from_u32(read_u32_le(r)?).unwrap();
                    for playout_elt in playouts.iter_mut() {
                        *playout_elt = read_u32_le(r)?;
                    }
                    history.push(Item { kind, playouts });
                }
                Ok(Self {
                    solution_name,
                    history: History(history),
                    timestep_limit,
                    final_outcome,
                })
            }
            _ => panic!("version number {} unsupported", version),
        }
    }

    pub fn write<W: Write>(&self, w: &mut W) -> Result<()> {
        let HistoryFile {
            solution_name,
            history: History(history),
            timestep_limit,
            final_outcome,
        } = self;
        write_u32_le(w, 2)?;
        {
            let dat = solution_name.as_bytes();
            write_u32_le(w, dat.len().try_into()?)?;
            w.write_all(dat)?;
        }
        write_u32_le(w, *timestep_limit)?;
        write_f32_le(w, *final_outcome)?;
        write_u32_le(w, history.len().try_into().unwrap())?;
        for item in history.iter() {
            write_u32_le(w, item.kind.to_u32().unwrap())?;
            for &playout in item.playouts.iter() {
                write_u32_le(w, playout)?;
            }
        }
        Ok(())
    }
}
