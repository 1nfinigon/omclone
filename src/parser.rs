use crate::sim::*;

use eyre::{bail, ensure, eyre, Result};
use num_traits::FromPrimitive;
use std::collections::HashMap;
use std::io::{Read, Write};

use bitflags::bitflags;
bitflags! {
    #[repr(transparent)]
    pub struct AllowedParts: u64 {
        const ALLOW_ARM                  = 1<<0 ;
        const ALLOW_MULTIARM_EQUILIBRIUM = 1<<1 ;
        const ALLOW_PISTON               = 1<<2 ;
        const ALLOW_TRACK                = 1<<3 ;
        const ALLOW_BONDER               = 1<<8 ;
        const ALLOW_UNBONDER             = 1<<9 ;
        const ALLOW_MULTIBONDER          = 1<<10;
        const ALLOW_TRIPLEX_BONDER       = 1<<11;
        const ALLOW_CALCIFICATION        = 1<<12;
        const ALLOW_DUPLICATION          = 1<<13;
        const ALLOW_PROJECTION           = 1<<14;
        const ALLOW_PURIFICATION         = 1<<15;
        const ALLOW_ANIMISMUS            = 1<<16;
        const ALLOW_DISPOSAL             = 1<<17;
        const ALLOW_QUINTESSENCE_GLYPHS  = 1<<18;
        const ALLOW_DROP_AND_ROTATION    = 1<<22;
        const ALLOW_GRAB                 = 1<<23;
        const ALLOW_RESET                = 1<<24;
        const ALLOW_REPEAT_NOOP          = 1<<25;
        const ALLOW_PIVOT                = 1<<26;
        const ALLOW_BERLOS_WHEEL         = 1<<28;
    }
}

fn expect_arr<const N: usize>(f: &mut impl Read, expected: [u8; N]) -> Result<()> {
    let mut dat = [0u8; N];
    f.read_exact(&mut dat)?;
    ensure!(
        dat == expected,
        "Expected value {:?} but read {:?}",
        expected,
        dat
    );
    Ok(())
}
fn expect_u8(f: &mut impl Read, expected: u8) -> Result<()> {
    expect_arr(f, expected.to_le_bytes())
}
fn expect_i32(f: &mut impl Read, expected: i32) -> Result<()> {
    expect_arr(f, expected.to_le_bytes())
}

fn parse_u8(f: &mut impl Read) -> Result<u8> {
    let mut dat = [0u8; 1];
    f.read_exact(&mut dat)?;
    Ok(u8::from_le_bytes(dat))
}
fn parse_i8(f: &mut impl Read) -> Result<i8> {
    let mut dat = [0u8; 1];
    f.read_exact(&mut dat)?;
    Ok(i8::from_le_bytes(dat))
}
fn parse_i32(f: &mut impl Read) -> Result<i32> {
    let mut dat = [0u8; 4];
    f.read_exact(&mut dat)?;
    Ok(i32::from_le_bytes(dat))
}
fn parse_u64(f: &mut impl Read) -> Result<u64> {
    let mut dat = [0u8; 8];
    f.read_exact(&mut dat)?;
    Ok(u64::from_le_bytes(dat))
}
fn parse_pos(f: &mut impl Read) -> Result<Pos> {
    let x = parse_i32(f)?;
    let y = parse_i32(f)?;
    Ok(Pos::new(x, y))
}
fn parse_bytepos(f: &mut impl Read) -> Result<Pos> {
    let x = parse_i8(f)? as i32;
    let y = parse_i8(f)? as i32;
    Ok(Pos::new(x, y))
}
fn parse_leb128(f: &mut impl Read) -> Result<usize> {
    let mut acc = 0;
    loop {
        let dat = parse_u8(f)?;
        acc *= 0x80;
        acc += (dat & 0x7F) as usize;
        if dat & 0x80 == 0 {
            return Ok(acc);
        }
    }
}
fn parse_str(f: &mut impl Read) -> Result<String> {
    let length = parse_leb128(f)?;
    let mut dat = vec![0; length as usize];
    f.read_exact(&mut dat)?;
    //Have to re-wrap to convert to anyhow error
    Ok(String::from_utf8(dat)?)
}

fn write_u8(f: &mut impl Write, n: u8) -> Result<()> {
    f.write_all(&n.to_le_bytes())?;
    Ok(())
}
fn write_i8(f: &mut impl Write, n: i8) -> Result<()> {
    f.write_all(&n.to_le_bytes())?;
    Ok(())
}
fn write_i32(f: &mut impl Write, n: i32) -> Result<()> {
    f.write_all(&n.to_le_bytes())?;
    Ok(())
}
fn write_u64(f: &mut impl Write, n: u64) -> Result<()> {
    f.write_all(&n.to_le_bytes())?;
    Ok(())
}
fn write_pos(f: &mut impl Write, p: Pos) -> Result<()> {
    write_i32(f, p.x)?;
    write_i32(f, p.y)?;
    Ok(())
}
fn write_bytepos(f: &mut impl Write, p: Pos) -> Result<()> {
    write_i8(f, p.x.try_into()?)?;
    write_i8(f, p.y.try_into()?)?;
    Ok(())
}
fn write_leb128(f: &mut impl Write, mut n: usize) -> Result<()> {
    loop {
        let mut this = n % 0x80;
        n /= 0x80;
        if n != 0 {
            this |= 0x80;
        }
        write_u8(f, this.try_into()?)?;
        if n == 0 {
            return Ok(());
        }
    }
}
fn write_str(f: &mut impl Write, s: &str) -> Result<()> {
    let dat = s.as_bytes();
    write_leb128(f, dat.len())?;
    f.write_all(dat)?;
    Ok(())
}

pub struct FullSolution {
    pub puzzle_name: String,
    pub solution_name: String,
    pub stats: Option<SolutionStats>,
    part_list: Vec<Part>,
}

fn parse_stats(f: &mut impl Read) -> Result<Option<SolutionStats>> {
    let finished = parse_i32(f)?;
    if finished == 0 {
        Ok(None)
    } else {
        expect_i32(f, 0)?;
        let cycles = parse_i32(f)?;
        expect_i32(f, 1)?;
        let cost = parse_i32(f)?;
        expect_i32(f, 2)?;
        let area = parse_i32(f)?;
        expect_i32(f, 3)?;
        let instructions = parse_i32(f)?;
        Ok(Some(SolutionStats {
            cycles,
            cost,
            area,
            instructions,
        }))
    }
}

fn write_stats(f: &mut impl Write, stats: &Option<SolutionStats>) -> Result<()> {
    match stats {
        None => write_i32(f, 0)?,
        Some(stats) => {
            write_i32(f, 4)?;
            write_i32(f, 0)?;
            write_i32(f, stats.cycles)?;
            write_i32(f, 1)?;
            write_i32(f, stats.cost)?;
            write_i32(f, 2)?;
            write_i32(f, stats.area)?;
            write_i32(f, 3)?;
            write_i32(f, stats.instructions)?;
        }
    }
    Ok(())
}

#[derive(Debug, PartialEq, Eq)]
enum PartType {
    TGlyph(BasicGlyphType),
    TArm(ArmType),
    TTrack,
    TInput,
    TOutput,
    TOutputRep,
    TConduit,
}
use PartType::*;
struct Part {
    part_type: PartType,
    pos: Pos,
    arm_size: i32,
    rot: i32,
    input_output_index: i32,
    instructions: Vec<(i32, Instr)>,
    tracks: Option<Vec<Pos>>,
    arm_index: i32,
    conduit: Option<(i32, Vec<Pos>)>,
}
fn name_to_part_type(name: String) -> Result<PartType> {
    use ArmType::*;
    use BasicGlyphType::*;
    Ok(match name.as_str() {
        "glyph-calcification" => TGlyph(Calcification),
        "glyph-life-and-death" => TGlyph(Animismus),
        "glyph-projection" => TGlyph(Projection),
        "glyph-dispersion" => TGlyph(Dispersion),
        "glyph-purification" => TGlyph(Purification),
        "glyph-duplication" => TGlyph(Duplication),
        "glyph-unification" => TGlyph(Unification),
        "bonder" => TGlyph(Bonding),
        "unbonder" => TGlyph(Unbonding),
        "bonder-prisma" => TGlyph(TriplexBond),
        "bonder-speed" => TGlyph(MultiBond),
        "glyph-disposal" => TGlyph(Disposal),
        "glyph-marker" => TGlyph(Equilibrium),
        "arm1" => TArm(PlainArm),
        "arm2" => TArm(DoubleArm),
        "arm3" => TArm(TripleArm),
        "arm6" => TArm(HexArm),
        "piston" => TArm(Piston),
        "baron" => TArm(VanBerlo),
        "track" => TTrack,
        "input" => TInput,
        "out-std" => TOutput,
        "out-rep" => TOutputRep,
        "pipe" => TConduit,
        _ => bail!("Arm/Glyph {:?} not recognized", name),
    })
}
fn part_type_to_name(part: &PartType) -> Result<&'static str> {
    use ArmType::*;
    use BasicGlyphType::*;
    Ok(match part {
        TGlyph(Calcification) => "glyph-calcification",
        TGlyph(Animismus) => "glyph-life-and-death",
        TGlyph(Projection) => "glyph-projection",
        TGlyph(Dispersion) => "glyph-dispersion",
        TGlyph(Purification) => "glyph-purification",
        TGlyph(Duplication) => "glyph-duplication",
        TGlyph(Unification) => "glyph-unification",
        TGlyph(Bonding) => "bonder",
        TGlyph(Unbonding) => "unbonder",
        TGlyph(TriplexBond) => "bonder-prisma",
        TGlyph(MultiBond) => "bonder-speed",
        TGlyph(Disposal) => "glyph-disposal",
        TGlyph(Equilibrium) => "glyph-marker",
        TArm(PlainArm) => "arm1",
        TArm(DoubleArm) => "arm2",
        TArm(TripleArm) => "arm3",
        TArm(HexArm) => "arm6",
        TArm(Piston) => "piston",
        TArm(VanBerlo) => "baron",
        TTrack => "track",
        TInput => "input",
        TOutput => "out-std",
        TOutputRep => "out-rep",
        TConduit => "pipe",
    })
}
pub fn parse_solution(f: &mut impl Read) -> Result<FullSolution> {
    expect_i32(f, 7)?;
    let puzzle_name = parse_str(f)?;
    let solution_name = parse_str(f)?;
    let stats = parse_stats(f)?;
    let mut solution_output = FullSolution {
        puzzle_name,
        solution_name,
        stats,
        part_list: Vec::new(),
    };
    for _ in 0..parse_i32(f)? {
        let part_type = name_to_part_type(parse_str(f)?)?;
        expect_u8(f, 1)?;
        let pos = parse_pos(f)?;
        let arm_size = parse_i32(f)?;
        let rot = parse_i32(f)?;
        let input_output_index = parse_i32(f)?;
        let instruction_count = parse_i32(f)?;
        let mut instructions = Vec::new();
        for _ in 0..instruction_count {
            let instr_pos = parse_i32(f)? as i32;
            let action = parse_u8(f)?;
            instructions.push((
                instr_pos,
                Instr::from_byte(action).ok_or(eyre!("invalid instruction"))?,
            ));
        }
        let tracks = if part_type == PartType::TTrack {
            let track_count = parse_i32(f)?;
            let mut inner_tracks = Vec::new();
            for _ in 0..track_count {
                inner_tracks.push(parse_pos(f)?);
            }
            Some(inner_tracks)
        } else {
            None
        };
        let arm_index = parse_i32(f)?;
        let conduit = if part_type == PartType::TConduit {
            let conduit_id = parse_i32(f)?;
            let conduit_count = parse_i32(f)?;
            let mut inner_conduits = Vec::new();
            for _ in 0..conduit_count {
                inner_conduits.push(parse_pos(f)?);
            }
            Some((conduit_id, inner_conduits))
        } else {
            None
        };
        solution_output.part_list.push(Part {
            part_type,
            pos,
            arm_size,
            rot,
            input_output_index,
            instructions,
            tracks,
            arm_index,
            conduit,
        });
    }
    Ok(solution_output)
}
pub fn create_solution(
    world: &WorldWithTapes,
    puzzle_name: String,
    solution_name: String,
    stats: Option<SolutionStats>,
) -> FullSolution {
    let mut solution_output = FullSolution {
        puzzle_name,
        solution_name,
        stats,
        part_list: Vec::new(),
    };
    let part_list = &mut solution_output.part_list;
    for glyph in &world.world.glyphs {
        let rot = glyph.rot;
        let pos = glyph.pos;
        let arm_size = 0;
        let arm_index = 0;
        let instructions = Vec::new();
        //Clone and return points to relative position (absolute position done in setup_sim/reposition_glyph)
        let (part_type, input_output_index, tracks, conduit) = match &glyph.glyph_type {
            GlyphType::Track(track) => (
                TTrack,
                0,
                Some(track.iter().map(|x| rotate(x - pos, -rot)).collect()),
                None,
            ),
            GlyphType::Input(_, id) => (TInput, *id, None, None),
            GlyphType::Output(_, _, id) => (TOutput, *id, None, None),
            GlyphType::OutputRepeating(_, _, id) => (TOutputRep, *id, None, None),
            GlyphType::Conduit(conduit, id) => (
                TConduit,
                *id,
                None,
                Some((*id, conduit.iter().map(|x| rotate(x - pos, -rot)).collect())),
            ),
            basic_glyph_type => (TGlyph(basic_glyph_type.try_into().unwrap()), 0, None, None),
        };
        part_list.push(Part {
            part_type,
            pos,
            arm_size,
            rot,
            input_output_index,
            instructions,
            tracks,
            arm_index,
            conduit,
        });
    }
    let mut found_first = false;
    let loop_len = world.repeat_length;
    for (id, (arm, tape)) in world.world.arms.iter().zip(world.tapes.iter()).enumerate() {
        let mut tape = tape.clone();
        tape.clear_leading_emptys();

        let part_type = TArm(arm.arm_type);
        let pos = arm.pos;
        let arm_size = arm.len;
        let rot = arm.rot;
        let input_output_index = 0;

        let mut instructions = Vec::new();
        let mut step = tape.first as i32;
        for &instr in &tape.instructions {
            if instr != BasicInstr::Empty {
                instructions.push((step, instr.into()));
            }
            step += 1;
        }
        let target_loop = (loop_len + tape.first) as i32;
        if !found_first && step < target_loop {
            instructions.push((target_loop - 1, Instr::Noop));
            found_first = true;
        }
        let tracks = None;
        let arm_index = id as _;
        let conduit = None;
        part_list.push(Part {
            part_type,
            pos,
            arm_size,
            rot,
            input_output_index,
            instructions,
            tracks,
            arm_index,
            conduit,
        });
    }
    solution_output
}
pub fn write_solution(f: &mut impl Write, sol: &FullSolution) -> Result<()> {
    write_i32(f, 7)?;
    write_str(f, &sol.puzzle_name)?;
    write_str(f, &sol.solution_name)?;
    write_stats(f, &sol.stats)?;
    write_i32(f, sol.part_list.len() as i32)?;
    for part in &sol.part_list {
        write_str(f, part_type_to_name(&part.part_type)?)?;
        write_u8(f, 1)?;
        write_pos(f, part.pos)?;
        write_i32(f, part.arm_size)?;
        write_i32(f, part.rot)?;
        write_i32(f, part.input_output_index)?;
        write_i32(f, part.instructions.len() as i32)?;
        for (instr_pos, instr) in &part.instructions {
            write_i32(f, *instr_pos)?;
            write_u8(f, instr.to_byte())?;
        }
        if let Some(tracks) = &part.tracks {
            ensure!(
                part.part_type == PartType::TTrack,
                "Tracks on non-track piece"
            );
            write_i32(f, tracks.len() as i32)?;
            for t in tracks {
                write_pos(f, *t)?;
            }
        }
        write_i32(f, part.arm_index)?;
        if let Some((id, conduits)) = &part.conduit {
            ensure!(
                part.part_type == PartType::TConduit,
                "Conduits on non-conduit piece"
            );
            write_i32(f, *id)?;
            write_i32(f, conduits.len() as i32)?;
            for c in conduits {
                write_pos(f, *c)?;
            }
        }
    }
    Ok(())
}

pub struct FullPuzzle {
    pub puzzle_name: String,
    pub creator_id: u64,
    pub allowed_parts: AllowedParts,
    pub inputs: Vec<AtomPattern>,
    pub outputs: Vec<AtomPattern>,
    pub output_multiplier: i32,
    pub production: bool,
    //TODO: production data here
}
fn parse_molecule(f: &mut impl Read) -> Result<AtomPattern> {
    let mut pattern = Vec::new();
    let mut atom_locs: HashMap<Pos, usize> = HashMap::new();
    for i in 0..(parse_i32(f)? as usize) {
        let atom_type = AtomType::from_u8(parse_u8(f)?).ok_or(eyre!("Illegal atom type"))?;
        let pos = parse_bytepos(f)?;
        pattern.push(Atom::new(pos, atom_type));
        let check = atom_locs.insert(pos, i);
        ensure!(check.is_none(), "Multiple atoms in same location!");
    }
    //bonds
    for _ in 0..parse_i32(f)? {
        let bond_type = Bonds::from_bits(parse_u8(f)?).ok_or(eyre!("Illegal bond type"))?;
        let from_pos = parse_bytepos(f)?;
        let to_pos = parse_bytepos(f)?;

        let atom1 = *atom_locs
            .get(&from_pos)
            .ok_or(eyre!("bond1 to nonatom position"))?;
        let atom2 = *atom_locs
            .get(&to_pos)
            .ok_or(eyre!("bond2 to nonatom position"))?;
        let rot =
            pos_to_rot(pattern[atom2].pos - pattern[atom1].pos).ok_or(eyre!("nonadjacent bond"))?;
        pattern[atom1].connections[rot as usize] = bond_type;
        pattern[atom2].connections[normalize_dir(rot + 3) as usize] = bond_type;
    }
    Ok(pattern)
}
fn write_molecule(f: &mut impl Write, pattern: &AtomPattern) -> Result<()> {
    let mut atom_locs: HashMap<Pos, usize> = HashMap::new();
    write_i32(f, pattern.len().try_into()?)?;
    for (i, atom) in pattern.iter().enumerate() {
        use num_traits::ToPrimitive;
        write_u8(f, atom.atom_type.to_u8().ok_or(eyre!("invalid atom type"))?)?;
        write_bytepos(f, atom.pos)?;
        atom_locs.insert(atom.pos, i);
    }
    let mut bonds: HashMap<(Pos, Pos), Bonds> = HashMap::new();
    for atom1_idx in 0..pattern.len() {
        for angle in 0..6 {
            let bond_type = pattern[atom1_idx].connections[angle as usize];
            if !bond_type.is_empty() {
                let mut from_pos = pattern[atom1_idx].pos;
                let mut to_pos = pattern[atom1_idx].pos + rot_to_pos(angle);
                let atom2_idx = *atom_locs
                    .get(&to_pos)
                    .ok_or(eyre!("bond to nonatom position"))?;
                ensure!(
                    pattern[atom2_idx].connections[normalize_dir(angle + 3) as usize] == bond_type
                );
                if (from_pos.x, from_pos.y) > (to_pos.x, to_pos.y) {
                    std::mem::swap(&mut from_pos, &mut to_pos);
                }
                ensure!(from_pos != to_pos, "bond to same location");
                if let Some(old_value) = bonds.insert((from_pos, to_pos), bond_type) {
                    ensure!(old_value == bond_type, "inconsistent bond types");
                }
            }
        }
    }
    write_i32(f, bonds.len().try_into()?)?;
    for ((from_pos, to_pos), bond_type) in bonds {
        write_u8(f, bond_type.bits())?;
        write_bytepos(f, from_pos)?;
        write_bytepos(f, to_pos)?;
    }
    Ok(())
}

/// `input` needs to be in relative space (i.e. call this before
/// `reposition_pattern`).
fn process_repeats(input: &Vec<Atom>, reps: i32) -> Result<AtomPattern> {
    let mut rep_offset = None;
    for atom in input {
        if atom.atom_type == AtomType::RepeatingOutputMarker {
            ensure!(rep_offset == None, "Multiple repeating atom markers!");
            rep_offset = Some(atom.pos);
        }
    }
    let rep_offset = rep_offset.ok_or(eyre!("Repeating atom marker expected but not found!"))?;
    let mut output = Vec::with_capacity(input.len() * (reps as usize));
    for atom in input {
        if atom.atom_type == AtomType::RepeatingOutputMarker {
            output.push(Atom {
                pos: atom.pos + ((reps - 1) * rep_offset),
                ..*atom
            });
        } else {
            for i in 0..reps {
                output.push(Atom {
                    pos: atom.pos + (i * rep_offset),
                    ..*atom
                });
            }
        }
    }
    Ok(output)
}

pub fn parse_puzzle(f: &mut impl Read) -> Result<FullPuzzle> {
    expect_i32(f, 3)?;
    let puzzle_name = parse_str(f)?;
    let creator_id = parse_u64(f)?;
    let allowed_parts = parse_u64(f)?;
    let allowed_parts = AllowedParts::from_bits(allowed_parts)
        .ok_or(eyre!("allowed parts bitfield error: {:b}", allowed_parts))?;
    let mut inputs = Vec::new();
    for _ in 0..parse_i32(f)? {
        inputs.push(parse_molecule(f)?)
    }
    let mut outputs = Vec::new();
    for _ in 0..parse_i32(f)? {
        outputs.push(parse_molecule(f)?)
    }
    let output_multiplier = parse_i32(f)?;
    let production = parse_u8(f)? != 0;
    //TODO: if production is true, read their stuff
    let puzzle_output = FullPuzzle {
        puzzle_name,
        creator_id,
        allowed_parts,
        inputs,
        outputs,
        output_multiplier,
        production,
    };
    Ok(puzzle_output)
}

#[allow(dead_code)]
pub fn write_puzzle(f: &mut impl Write, puzzle: &FullPuzzle) -> Result<()> {
    let FullPuzzle {
        puzzle_name,
        creator_id,
        allowed_parts,
        inputs,
        outputs,
        output_multiplier,
        production,
    } = puzzle;

    write_i32(f, 3)?;
    write_str(f, puzzle_name)?;
    write_u64(f, *creator_id)?;
    write_u64(f, allowed_parts.bits())?;

    write_i32(f, inputs.len().try_into()?)?;
    for input in inputs {
        write_molecule(f, input)?;
    }

    write_i32(f, outputs.len().try_into()?)?;
    for output in outputs {
        write_molecule(f, output)?;
    }

    write_i32(f, *output_multiplier)?;
    write_u8(f, if *production { 1 } else { 0 })?;
    //TODO: if production is true, write their stuff
    Ok(())
}

fn process_instructions(input: &[(i32, Instr)]) -> Result<Tape<Instr>> {
    let mut first = usize::MAX; //A giant value
    let mut instructions = Vec::new();
    for &(pos, instr) in input {
        ensure!(instr != Instr::Empty, "Empty instruction found in file!");
        let pos = pos as usize;
        if first > pos {
            if first != usize::MAX {
                let splice_source = std::iter::repeat(Instr::Empty).take(first - pos);
                instructions.splice(..0, splice_source);
            }
            first = pos
        }
        if pos >= first + instructions.len() {
            instructions.resize(pos - first + 1, Instr::Empty);
        }
        instructions[pos - first] = instr;
    }
    if first == usize::MAX {
        first = 0;
    }
    Ok(Tape {
        first,
        instructions,
    })
}

pub fn puzzle_prep(puzzle: &FullPuzzle, soln: &FullSolution) -> Result<InitialWorld> {
    //Convert from generic parts to glyphs, arms, etc.
    let mut glyphs = Vec::new();
    let mut arms_and_tapes = Vec::new();
    for p in &soln.part_list {
        match &p.part_type {
            TGlyph(gtype) => {
                glyphs.push(Glyph {
                    glyph_type: (*gtype).into(),
                    pos: p.pos,
                    rot: p.rot,
                });
            }
            TArm(atype) => {
                let instr = process_instructions(&p.instructions)?;
                arms_and_tapes.push((
                    Arm::new(p.pos, p.rot, p.arm_size, *atype),
                    instr,
                    p.arm_index,
                ));
            }
            TInput => {
                let id = p.input_output_index;
                let pattern = puzzle.inputs.get(id as usize).ok_or(eyre!(
                    "Input ID {} not found (max {})",
                    id,
                    puzzle.inputs.len()
                ))?;
                let input_glyph = GlyphType::Input(pattern.clone(), p.input_output_index);
                let mut glyph = Glyph {
                    glyph_type: input_glyph,
                    pos: Pos::zeros(),
                    rot: 0,
                };
                glyph.rot_by(p.rot);
                glyph.move_by(p.pos);
                glyphs.push(glyph);
            }
            TOutput => {
                let id = p.input_output_index;
                let pattern = puzzle.outputs.get(id as usize).ok_or(eyre!(
                    "Output ID {} not found (max {})",
                    id,
                    puzzle.outputs.len()
                ))?;
                let output_glyph = GlyphType::Output(
                    pattern.clone(),
                    6 * puzzle.output_multiplier,
                    p.input_output_index,
                );
                let mut glyph = Glyph {
                    glyph_type: output_glyph,
                    pos: Pos::zeros(),
                    rot: 0,
                };
                glyph.rot_by(p.rot);
                glyph.move_by(p.pos);
                glyphs.push(glyph);
            }
            TOutputRep => {
                let id = p.input_output_index;
                let pattern = puzzle.outputs.get(id as usize).ok_or(eyre!(
                    "Output(rep) ID {} not found (max {})",
                    id,
                    puzzle.outputs.len()
                ))?;
                let pattern = process_repeats(pattern, 6)?;
                let output_glyph = GlyphType::OutputRepeating(
                    pattern,
                    6 * puzzle.output_multiplier,
                    p.input_output_index,
                );
                let mut glyph = Glyph {
                    glyph_type: output_glyph,
                    pos: Pos::zeros(),
                    rot: 0,
                };
                glyph.rot_by(p.rot);
                glyph.move_by(p.pos);
                glyphs.push(glyph);
            }
            TTrack => {
                let track_locs = p
                    .tracks
                    .clone()
                    .ok_or(eyre!("Track data not found on track glyph"))?;
                let mut glyph = Glyph {
                    glyph_type: GlyphType::Track(track_locs),
                    pos: Pos::zeros(),
                    rot: 0,
                };
                glyph.rot_by(p.rot);
                glyph.move_by(p.pos);
                glyphs.push(glyph);
            }
            TConduit => {
                let (conduit_id, conduit_locs) = p
                    .conduit
                    .clone()
                    .ok_or(eyre!("Conduit data not found on conduit glyph"))?;
                let mut glyph = Glyph {
                    glyph_type: GlyphType::Conduit(conduit_locs, conduit_id),
                    pos: Pos::zeros(),
                    rot: 0,
                };
                glyph.rot_by(p.rot);
                glyph.move_by(p.pos);
                glyphs.push(glyph);
            }
        }
    }
    arms_and_tapes.sort_by(|a, b| a.2.cmp(&b.2));
    let (arms, tapes) = arms_and_tapes
        .into_iter()
        .map(|(arm, tape, _)| (arm, tape))
        .unzip();
    Ok(InitialWorld {
        glyphs,
        arms,
        tapes,
    })
}
