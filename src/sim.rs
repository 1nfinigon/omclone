#![allow(unused_parens)]
use bitflags::bitflags;
#[cfg(feature = "color_eyre")]
use color_eyre::{
    eyre::{bail, ensure, eyre},
    Result,
};
#[cfg(not(feature = "color_eyre"))]
use simple_eyre::{
    eyre::{bail, ensure, eyre},
    Result,
};
pub use nalgebra::Vector2;
use slotmap::{new_key_type, Key, SecondaryMap, SlotMap};
use std::collections::VecDeque;
use rustc_hash::{FxHashMap, FxHashSet};

pub type Rot = i32;
pub type Pos = Vector2<i32>;

#[derive(Copy, Clone, Debug, PartialEq, Eq, num_derive::FromPrimitive, num_derive::ToPrimitive)]
#[repr(u8)]
pub enum Instr {
    RotateClockwise,
    RotateCounterClockwise,
    Extend,
    Retract,
    Grab,
    Drop,
    PivotClockwise,
    PivotCounterClockwise,
    Forward,
    Back,
    Repeat,
    Reset,
    Noop,
    Empty,
}
impl Instr {
    pub fn to_byte(&self) -> u8 {
        use Instr::*;
        match self {
            RotateClockwise         => b'R',
            RotateCounterClockwise  => b'r',
            Extend                  => b'E',
            Retract                 => b'e',
            Grab                    => b'G',
            Drop                    => b'g',
            PivotClockwise          => b'P',
            PivotCounterClockwise   => b'p',
            Forward                 => b'A',
            Back                    => b'a',
            Repeat                  => b'C',
            Reset                   => b'X',
            Noop                    => b'O',
            Empty                   => b' ',
        }
    }
    pub fn from_byte(input: u8) -> Result<Self> {
        use Instr::*;
        match input {
            b'R' => Ok(RotateClockwise),
            b'r' => Ok(RotateCounterClockwise),
            b'E' => Ok(Extend),
            b'e' => Ok(Retract),
            b'G' => Ok(Grab),
            b'g' => Ok(Drop),
            b'P' => Ok(PivotClockwise),
            b'p' => Ok(PivotCounterClockwise),
            b'A' => Ok(Forward),
            b'a' => Ok(Back),
            b'C' => Ok(Repeat),
            b'X' => Ok(Reset),
            b'O' => Ok(Noop),
            b' ' => Ok(Empty),
            _ => Err(eyre!("Illegal instruction byte")),
        }
    }
    pub fn to_char(&self) -> char {
        use Instr::*;
        match self {
            RotateClockwise         => 'd',
            RotateCounterClockwise  => 'a',
            Extend                  => 'w',
            Retract                 => 's',
            Grab                    => 'f',
            Drop                    => 'r',
            PivotClockwise          => 'e',
            PivotCounterClockwise   => 'q',
            Forward                 => 'g',
            Back                    => 't',
            Repeat                  => 'C',
            Reset                   => 'X',
            Noop                    => 'O',
            Empty                   => ' ',
        }
    }
    pub fn from_char(input: char) -> Option<Self> {
        use Instr::*;
        match input {
             'd' =>  Some(RotateClockwise       ),
             'a' =>  Some(RotateCounterClockwise),
             'w' =>  Some(Extend                ),
             's' =>  Some(Retract               ),
             'f' =>  Some(Grab                  ),
             'r' =>  Some(Drop                  ),
             'e' =>  Some(PivotClockwise        ),
             'q' =>  Some(PivotCounterClockwise ),
             'g' =>  Some(Forward               ),
             't' =>  Some(Back                  ),
             //'C' =>  Some(Repeat                ),
             //'X' =>  Some(Reset                 ),
             //'O' =>  Some(Noop                  ),
             ' ' =>  Some(Empty                 ),
             _ => None
        }
    }
}

#[derive(Debug, Clone)]
pub struct Tape {
    pub first: usize,
    pub instructions: Vec<Instr>,
}
impl Tape {
    pub fn get(&self, timestep: usize, loop_len: usize) -> Instr {
        use Instr::Empty;
        if timestep >= self.first {
            let after_first = timestep - self.first;
            *self
                .instructions
                .get(after_first % loop_len)
                .unwrap_or(&Empty)
        } else {
            Empty
        }
    }
    pub fn to_string(&self) -> String{
        let mut output = " ".repeat(self.first);
        for i in &self.instructions{
            output.push(i.to_char());
        }
        output
    }
    pub fn modify_and_string(&mut self) -> String{
        while self.instructions.get(0).unwrap_or(&Instr::Noop) == &Instr::Empty
        {
            self.first += 1;
            self.instructions.remove(0);
        }
        self.to_string()
    }
}

pub fn normalize_dir(r: Rot) -> Rot {
    return r.rem_euclid(6);
}

pub fn pos_to_rot(input: Pos) -> Result<Rot> {
    match (input.x, input.y) {
        ( 1, 0) => Ok(0),
        ( 0, 1) => Ok(1),
        (-1, 1) => Ok(2),
        (-1, 0) => Ok(3),
        ( 0,-1) => Ok(4),
        ( 1,-1) => Ok(5),
        _ => Err(eyre!("Invalid position converted to rotation: {}", input)),
    }
}
pub fn offset(n: i32, angle: Rot) -> Pos {
    match normalize_dir(angle) {
        0 => Pos::new( n, 0),
        1 => Pos::new( 0, n),
        2 => Pos::new(-n, n),
        3 => Pos::new(-n, 0),
        4 => Pos::new( 0,-n),
        5 => Pos::new( n,-n),
        _ => panic!("Invalid Rotation"),
    }
}
pub fn rot_to_pos(angle: Rot) -> Pos {
    offset(1, angle)
}
pub fn rotate(pos: Pos, angle: Rot) -> Pos {
    match normalize_dir(angle) {
        0 => pos, //  ( pos.x      ,       pos.y)
        1 => Pos::new(      -pos.y , pos.x+pos.y),
        2 => Pos::new(-pos.x-pos.y , pos.x      ),
        3 => Pos::new(-pos.x       ,      -pos.y),
        4 => Pos::new(       pos.y ,-pos.x-pos.y),
        5 => Pos::new( pos.x+pos.y ,-pos.x      ),
        _ => panic!("Invalid Rotation"),
    }
}

pub fn rotate_around(pos: Pos, angle: Rot, pivot: Pos) -> Pos {
    rotate(pos - pivot, angle) + pivot
}

bitflags! {
    pub struct Bonds: u8 {
        const NO_BOND     =0b0000_0000;
        const NORMAL      =0b0000_0001;
        const TRIPLEX_R   =0b0000_0010;
        const TRIPLEX_K   =0b0000_0100;
        const TRIPLEX_Y   =0b0000_1000;
        const TRIPLEX     =0b0000_1110;
        const DYNAMIC_BOND=0b0000_1111;
        const CONDUIT_BOND=0b0001_0000;
        const BERLO_TYPE  =0b0010_0000;
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, num_derive::FromPrimitive, num_derive::ToPrimitive)]
#[repr(u8)]
pub enum AtomType {
    ConduitSpace = 0,
    Salt, Air, Earth, Fire, Water,
    Quicksilver, Gold, Silver, Copper, Iron, Tin, Lead,
    Vitae, Mors, RepeatingOutputMarker, Quintessence,
}
impl AtomType {
    pub fn is_element(&self) -> bool {
        use AtomType::*;
        match &self {
            Air | Earth | Fire | Water => true,
            _ => false,
        }
    }
    pub fn promotable_metal(&self) -> Option<AtomType> {
        use AtomType::*;
        match &self {
            Lead => Some(Tin),
            Tin => Some(Iron),
            Iron => Some(Copper),
            Copper => Some(Silver),
            Silver => Some(Gold),
            _ => None,
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Atom {
    pub pos: Pos,
    pub atom_type: AtomType,
    pub connections: [Bonds; 6],
}
impl Atom {
    pub fn new(pos: Pos, atom_type: AtomType) -> Atom {
        Atom {
            pos,
            atom_type,
            connections: [Bonds::NO_BOND; 6],
        }
    }
    pub fn rotate_connections(&mut self, r: Rot) {
        let mut new_connections = [Bonds::NO_BOND; 6];
        for i in 0..6 {
            new_connections[normalize_dir(i + r) as usize] = self.connections[i as usize];
        }
        self.connections = new_connections;
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Movement {
    HeldStill,
    Linear(Pos),
    Rotation(Rot, Pos),
    Pivot(Rot),
}

pub type AtomPattern = Vec<Atom>;

//Note: pre-setup, AtomPatterns are local and must be offset/rotated. After, they are global.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GlyphType {
    Calcification, Animismus, Projection,
    Dispersion, Purification, Duplication, Unification,
    Bonding, Unbonding, TriplexBond, MultiBond,
    Disposal, Equilibrium,
    Track(Vec<Pos>), Conduit(AtomPattern),
    Input(AtomPattern), Output(AtomPattern, i32),
}
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ArmType {
    PlainArm, DoubleArm, TripleArm, HexArm,
    Piston, VanBerlo,
}
#[derive(Debug, Clone)]
pub struct Glyph {
    pub pos: Pos,
    pub rot: Rot,
    pub glyph_type: GlyphType,
}
impl Glyph {
    pub fn reposition(&self, rel_pos: Pos) -> Pos {
        self.pos + rotate(rel_pos, self.rot)
    }
    //returns true if it was a glyph that can be repositioned
    pub fn reposition_pattern(&mut self) -> bool {
        use GlyphType::*;
        match &mut self.glyph_type {
            Input(pat) | Output(pat, _) | Conduit(pat) => {
                for mut a in pat {
                    a.pos = self.pos + rotate(a.pos, self.rot);
                    a.rotate_connections(self.rot);
                }
                true
            }
            _ => false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Arm {
    pub pos: Pos,
    pub rot: Rot,
    pub len: i32,
    pub arm_type: ArmType,
    pub instruction_tape: Tape,
    pub grabbing: bool,
    pub atoms_grabbed: [AtomKey; 6],
}

impl Arm {
    pub fn new(pos: Pos, rot: Rot, len: i32, arm_type: ArmType, instruction_tape: Tape) -> Self {
        Arm { pos, rot, len, arm_type, instruction_tape,
            grabbing: false,
            atoms_grabbed: [AtomKey::null(); 6],
        }
    }
    pub fn angles_between_arm(&self) -> Rot {
        use ArmType::*;
        match self.arm_type {
            PlainArm => 6,
            DoubleArm => 3,
            TripleArm => 2,
            HexArm => 1,
            Piston => 6,
            VanBerlo => 1,
        }
    }
}

#[derive(Debug)]
pub struct InitialWorld {
    pub glyphs: Vec<Glyph>,
    pub arms: Vec<Arm>,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct TrackMapData {
    pub minus: Option<Pos>,
    pub plus: Option<Pos>,
}
pub type TrackMap = FxHashMap<Pos, TrackMapData>;
#[derive(Debug, Clone)]
pub struct World {
    pub timestep: u64,
    pub atoms: WorldAtoms,
    pub area_touched: FxHashSet<Pos>,
    pub glyphs: Vec<Glyph>,
    pub arms: Vec<Arm>,
    pub repeat_length: usize,
    pub track_map: TrackMap,
    pub cost: i32,
    pub instruction_count: i32,
}
/*impl std::fmt::Display for World {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let arm_data: Vec<(Pos, Rot, bool)> = self.arms.iter().map(|a| (a.pos, a.rot, a.grabbing)).collect();
        let atom_data: Vec<(Pos, AtomType)> = self.atoms.atom_map.iter().map(|(_, a)| (a.pos, a.atom_type)).collect();
        write!(f, "{:?} || {:?}", arm_data, atom_data)
    }
}*/

new_key_type! { pub struct AtomKey; }
#[derive(Debug, Clone)]
pub struct WorldAtoms {
    pub atom_map: SlotMap<AtomKey, Atom>,
    pub locs: FxHashMap<Pos, AtomKey>,
    pub moves: SecondaryMap<AtomKey, Movement>,
}
impl WorldAtoms {
    fn new() -> WorldAtoms {
        WorldAtoms {
            atom_map: SlotMap::with_key(),
            locs: Default::default(),
            moves: SecondaryMap::new(),
        }
    }
    fn get_atom_mut(&mut self, loc: Pos) -> Option<&mut Atom> {
        let &key = self.locs.get(&loc)?;
        Some(&mut self.atom_map[key])
    }
    fn get_type(&self, loc: Pos) -> Option<AtomType> {
        let &key = self.locs.get(&loc)?;
        Some(self.atom_map.get(key)?.atom_type)
    }
    fn get_consumable_type(&self, loc: Pos) -> Option<AtomType> {
        let &key = self.locs.get(&loc)?;
        if self.moves.get(key) != None {
            return None;
        }
        Some(self.atom_map.get(key)?.atom_type)
    }

    fn destroy_atom_at(&mut self, loc: Pos) {
        let atom_key = self.locs.remove(&loc).expect("Destroying nonatom!");
        if self.moves.remove(atom_key) != None {
            panic!("Destroying atom that is still held!");
        }
        let atom = self
            .atom_map
            .remove(atom_key)
            .expect("Inconsistent atoms (destroy)!");
        if atom.pos != loc {
            panic!("Inconsistent atoms (destroy loc)!");
        }
    }
    fn create_atom(&mut self, atom: Atom) -> AtomKey {
        let key = self.atom_map.insert(atom);
        let atom_ref = &self.atom_map[key];
        self.locs.insert(atom_ref.pos, key);
        key
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct SolutionStats {
    pub cycles: i32,
    pub cost: i32,
    pub area: i32,
    pub instructions: i32,
}

//Running the world
impl World {
    //sets up a move for the specified atom and all atoms connected to it
    fn premove_atoms(&mut self, atom_key: AtomKey, movement: Movement) -> Result<()> {
        let mut moving_atoms = VecDeque::<AtomKey>::new();
        moving_atoms.push_back(atom_key);
        let normalized_movement = match movement{
            Movement::Pivot(r) => Movement::Rotation(r, self.atoms.atom_map[atom_key].pos),
            _ => movement
        };

        while let Some(this_key) = moving_atoms.pop_front() {
            let maybe_move = self.atoms.moves.get(this_key);
            if let Some(curr_move) = maybe_move {
                if curr_move != &normalized_movement {
                    return Err(eyre!("Atom moved in multiple directions!"));
                }
            } else {
                self.atoms.moves.insert(this_key, normalized_movement);
                let atom = &self.atoms.atom_map[this_key];
                for dir in 0..6 {
                    if !(atom.connections[dir as usize] & Bonds::DYNAMIC_BOND).is_empty() {
                        let newpos = atom.pos + rot_to_pos(dir);
                        let newkey = *self.atoms.locs.get(&newpos).expect("Inconsistent atoms (movement prep)");
                        moving_atoms.push_back(newkey);
                    }
                }
            }
        }
        Ok(())
    }

    //Finalize atom movement
    fn move_atoms(&mut self) -> Result<()> {
        //remove all grabbed atoms from the grid
        for (atom_key, _action) in self.atoms.moves.iter() {
            let atom = &mut self.atoms.atom_map[atom_key];
            let loc_check = self.atoms.locs.remove(&atom.pos);
            if loc_check != Some(atom_key) {
                panic!("Inconsistent atoms (move loc)!");
            }
        }
        //add them back in at their new locations
        for (atom_key, action) in self.atoms.moves.iter() {
            let atom = &mut self.atoms.atom_map[atom_key];
            match action {
                Movement::Linear(p) => atom.pos += p,
                Movement::Rotation(r, p) => {
                    atom.pos = rotate_around(atom.pos, *r, *p);
                    atom.rotate_connections(*r);
                }
                Movement::HeldStill => (),
                Movement::Pivot(_) => bail!("Atom movement is pivot post-normalization!"),
            }
            let current = self.atoms.locs.insert(atom.pos, atom_key);
            if current != None {
                bail!("Atom moved to position with another atom!");
            }
        }
        Ok(())
    }

    fn do_instruction(&mut self, arm_id: usize, timestep: u64) -> Result<()> {
        let arm = self.arms.get_mut(arm_id).unwrap();
        use ArmType::*;
        use Instr::*;
        use Movement::*;
        let arm_type = arm.arm_type;
        let tape = &arm.instruction_tape;
        let timestep = timestep as usize;
        let instruction = tape.get(timestep, self.repeat_length);
        let action = match instruction {
            Extend => {
                if arm_type == Piston && arm.len < 3 {
                    arm.len += 1;
                    Linear(rot_to_pos(arm.rot))
                } else { HeldStill }
            }
            Retract => {
                if arm_type == Piston && arm.len > 1 {
                    arm.len -= 1;
                    Linear(-rot_to_pos(arm.rot))
                } else { HeldStill }
            }
            RotateCounterClockwise => {
                arm.rot += 1;
                Rotation(1, arm.pos)
            }
            RotateClockwise => {
                arm.rot -= 1;
                Rotation(-1, arm.pos)
            }
            PivotCounterClockwise => Pivot(1),
            PivotClockwise => Pivot(-1),
            Drop => {
                if arm_type != VanBerlo {
                    arm.atoms_grabbed = [AtomKey::null(); 6];
                    arm.grabbing = false;
                }
                HeldStill
            }
            Grab => {
                if !arm.grabbing && arm_type != VanBerlo {
                    arm.grabbing = true;
                    for r in (0..6).step_by(arm.angles_between_arm() as usize) {
                        let grab_pos = arm.pos + (rot_to_pos(arm.rot + r) * arm.len);
                        let null_key = AtomKey::null();
                        let current = self.atoms.locs.get(&grab_pos).unwrap_or(&null_key);
                        arm.atoms_grabbed[r as usize] = *current;
                    }
                }
                HeldStill
            }
            Forward => {
                let track_data = self.track_map.get(&arm.pos)
                    .ok_or(eyre!("Forward movement not on track"))?;
                track_data.plus.map_or(HeldStill, |x| {
                    arm.pos += x;
                    Linear(x)
                })
            }
            Back => {
                let track_data = self.track_map.get(&arm.pos)
                    .ok_or(eyre!("Backward movement not on track"))?;
                track_data.minus.map_or(HeldStill, |x| {
                    arm.pos += x;
                    Linear(x)
                })
            }
            Repeat | Reset | Noop => {
                bail!("Unprocessed instruction!");
            }
            Empty => (HeldStill),
        };
        for atom in arm.atoms_grabbed {
            if !atom.is_null() {
                self.premove_atoms(atom, action)?;
            }
        }
        Ok(())
    }

    fn process_glyphs(&mut self) {
        fn try_bond(atoms: &mut WorldAtoms, loc1: &Pos, loc2: &Pos, bond_type: Bonds, bond_mask: Bonds ) {
            let rot = pos_to_rot(loc2 - loc1).unwrap() as usize;
            if let (Some(&key1), Some(&key2)) = (atoms.locs.get(loc1), atoms.locs.get(loc2)) {
                let bond1 = atoms.atom_map[key1].connections[rot];
                assert_eq!(atoms.atom_map[key2].connections[(rot + 3) % 6], bond1, "Inconsistent bonds");
                if (bond1 & !bond_mask).is_empty() {
                    atoms.atom_map[key1].connections[rot] |= bond_type;
                    atoms.atom_map[key2].connections[(rot + 3) % 6] |= bond_type;
                }
            }
        }
        use AtomType::*;
        use GlyphType::*;
        for glyph in &mut self.glyphs {
            let atoms = &mut self.atoms;
            let rot = normalize_dir(glyph.rot) as usize;

            let pos = glyph.pos; //primary position
            let pos_bi = glyph.reposition(Pos::new(1, 0)); //position for all 2-sized glyphs
            let pos_tri = glyph.reposition(Pos::new(0, 1)); //third (closely packed) position

            let pos_ani = glyph.reposition(Pos::new(1, -1));

            let pos_disp2 = glyph.reposition(Pos::new(1, -1));
            let pos_disp3 = glyph.reposition(Pos::new(0, -1));
            let pos_disp4 = glyph.reposition(Pos::new(-1, 0));

            let pos_multi2 = glyph.reposition(Pos::new(0, -1));
            let pos_multi3 = glyph.reposition(Pos::new(-1, 1));

            let pos_unif2 = glyph.reposition(Pos::new(-1, 1));
            let pos_unif3 = glyph.reposition(Pos::new(0, -1));
            let pos_unif4 = glyph.reposition(Pos::new(1, -1));
            match &mut glyph.glyph_type{
                Calcification => {
                    if let Some(atom) = atoms.get_atom_mut(pos) {
                        if atom.atom_type.is_element(){
                            atom.atom_type = AtomType::Salt;
                        }
                    }
                }
                Animismus => {
                    let a1 = atoms.get_consumable_type(pos);
                    let a2 = atoms.get_consumable_type(pos_bi);
                    let o1 = atoms.get_type(pos_tri);
                    let o2 = atoms.get_type(pos_ani);
                    if (Some(Salt),Some(Salt),None,None) == (a1,a2,o1,o2){
                        atoms.destroy_atom_at(pos);
                        atoms.destroy_atom_at(pos_bi);
                        atoms.create_atom(Atom::new(pos_tri, Vitae));
                        atoms.create_atom(Atom::new(pos_ani, Mors));
                    }
                }
                Projection => {
                    let qs = atoms.get_consumable_type(pos);
                    let metal = atoms.get_type(pos_bi);
                    if let (Some(Quicksilver), Some(metal)) = (qs, metal){
                        if let Some(newtype) = metal.promotable_metal(){
                            atoms.destroy_atom_at(pos);
                            atoms.get_atom_mut(pos_bi).unwrap().atom_type = newtype;
                        }
                    }
                }
                Dispersion => {
                    let q = atoms.get_consumable_type(pos);
                    let o1 = atoms.get_type(pos_bi);
                    let o2 = atoms.get_type(pos_disp2);
                    let o3 = atoms.get_type(pos_disp3);
                    let o4 = atoms.get_type(pos_disp4);
                    if (Some(Quintessence),None,None,None,None) == (q,o1,o2,o3,o4){
                        atoms.destroy_atom_at(pos);
                        atoms.create_atom(Atom::new(pos_bi, Earth));
                        atoms.create_atom(Atom::new(pos_disp2, Water));
                        atoms.create_atom(Atom::new(pos_disp3, Fire));
                        atoms.create_atom(Atom::new(pos_disp4, Air));
                    }
                }
                Unification => {
                    let output = atoms.get_type(pos);
                    let a1 = atoms.get_consumable_type(pos_tri);
                    let a2 = atoms.get_consumable_type(pos_unif2);
                    let a3 = atoms.get_consumable_type(pos_unif3);
                    let a4 = atoms.get_consumable_type(pos_unif4);
                    if let (None,Some(a),Some(b),Some(c),Some(d)) = (output, a1, a2, a3, a4){
                        let set = [a,b,c,d];
                        if set.contains(&Earth) && set.contains(&Water) && set.contains(&Fire) && set.contains(&Air){
                            atoms.destroy_atom_at(pos_tri);
                            atoms.destroy_atom_at(pos_unif2);
                            atoms.destroy_atom_at(pos_unif3);
                            atoms.destroy_atom_at(pos_unif4);
                            atoms.create_atom(Atom::new(pos, Quintessence));
                        }
                    }
                }
                Purification => {
                    let a1 = atoms.get_consumable_type(pos);
                    let a2 = atoms.get_consumable_type(pos_bi);
                    let o = atoms.get_type(pos_tri);
                    match (a1,a2,o){
                        (Some(a1),Some(a2),None) if a1 == a2 => {
                            let next = a1.promotable_metal();
                            if let Some(newtype) = next {
                                atoms.destroy_atom_at(pos);
                                atoms.destroy_atom_at(pos_bi);
                                atoms.create_atom(Atom::new(pos_tri, newtype));
                            }
                        }
                        _ => ()
                    }
                }
                Duplication => {
                    let source = atoms.get_type(pos);
                    let salt = atoms.get_type(pos_bi);
                    if let (Some(elem), Some(Salt)) = (source, salt){
                        if elem.is_element(){
                            atoms.get_atom_mut(pos_bi).unwrap().atom_type = elem;
                        }
                    }
                }
                Bonding => {
                    try_bond(atoms, &pos, &pos_bi, Bonds::NORMAL, Bonds::NORMAL);
                }
                MultiBond => {
                    try_bond(atoms, &pos, &pos_bi, Bonds::NORMAL, Bonds::NORMAL);
                    try_bond(atoms, &pos, &pos_multi2, Bonds::NORMAL, Bonds::NORMAL);
                    try_bond(atoms, &pos, &pos_multi3, Bonds::NORMAL, Bonds::NORMAL);
                }
                TriplexBond => {
                    try_bond(atoms, &pos, &pos_bi, Bonds::TRIPLEX_K, Bonds::TRIPLEX);
                    try_bond(atoms, &pos, &pos_tri, Bonds::TRIPLEX_R, Bonds::TRIPLEX);
                    try_bond(atoms, &pos_bi, &pos_tri, Bonds::TRIPLEX_Y, Bonds::TRIPLEX);
                }
                Unbonding => {
                    if let (Some(&key1), Some(&key2)) = (atoms.locs.get(&pos), atoms.locs.get(&pos_bi)){
                        let bond1 = atoms.atom_map[key1].connections[rot];
                        assert_eq!(atoms.atom_map[key2].connections[(rot+3)%6], bond1, "Inconsistent bonds");
                        if !(bond1 & Bonds::DYNAMIC_BOND).is_empty() {
                            atoms.atom_map[key1].connections[rot] = Bonds::NO_BOND;
                            atoms.atom_map[key2].connections[(rot+3)%6] = Bonds::NO_BOND;
                        }
                    }
                }
                Input(atom_spawn_points) => {
                    if atom_spawn_points.iter().all(|a| atoms.locs.get(&a.pos) == None){
                        for a in atom_spawn_points{
                            atoms.create_atom(a.clone());
                        }
                    }
                }
                Output(atom_drop_points, output_count) => {
                    let full_match = atom_drop_points.iter().all( |a|-> bool {
                            let try_key = atoms.locs.get(&a.pos);
                            if let Some(&atom_key) = try_key{
                                &atoms.atom_map[atom_key] == a && atoms.moves.get(atom_key) == None
                            } else {false}
                        });
                    if full_match {
                        for a in atom_drop_points{
                            atoms.destroy_atom_at(a.pos);
                        }
                        if *output_count > 0{
                            *output_count -= 1;
                        }
                    }
                },
                Disposal => {
                    if let Some(&atom_key) = atoms.locs.get(&pos){
                        if atoms.atom_map[atom_key].connections == [Bonds::NO_BOND;6]
                        && !atoms.moves.contains_key(atom_key) {
                            atoms.destroy_atom_at(pos);
                        }
                    }
                },
                Conduit(_atom_teleport) => {
                    //TODO
                },
                Equilibrium | Track(_) => (),
            }
        }
    }

    //returns true if the solution is fully solved
    pub fn run_step(&mut self) -> Result<()> {
        for a in 0..self.arms.len() {
            self.do_instruction(a, self.timestep)?;
        }
        self.process_glyphs();

        self.move_atoms()?;
        self.process_glyphs();
        self.atoms.moves.clear();
        self.timestep += 1;
        Ok(())
    }

    pub fn is_complete(&self) -> bool{
        let mut all_outputs_full = true;
        for g in &self.glyphs {
            if let GlyphType::Output(_, i) = g.glyph_type {
                all_outputs_full &= (i == 0);
            }
        }
        all_outputs_full
    }
}

//Setup stuff
impl World {
    fn add_track(track_map: &mut TrackMap, g: &Glyph) -> Result<()> {
        let track_pos = {
            if let GlyphType::Track(t) = &g.glyph_type {
                t
            } else {
                bail!("Adding track from nontrack object");
            }
        };
        ensure!(track_pos.len() > 0, "Track of length 0!");
        track_map.insert(
            g.reposition(track_pos[0]),
            TrackMapData {
                minus: None,
                plus: None,
            },
        );
        //first do all except the possible loopback
        for track_pair in track_pos.windows(2) {
            let (t_prev, t_now) = (track_pair[0], track_pair[1]);
            let offset = t_now - t_prev;
            track_map.insert(
                g.reposition(t_now),
                TrackMapData {
                    minus: Some(-offset),
                    plus: None,
                },
            );
            track_map.get_mut(&g.reposition(t_prev)).unwrap().plus = Some(offset);
        }
        //check for looping on first/last
        if track_pos.len() > 2 {
            let first = *track_pos.first().unwrap();
            let last = *track_pos.last().unwrap();
            let offset = first - last;
            if pos_to_rot(offset).is_ok() {
                track_map.get_mut(&g.reposition(first)).unwrap().minus = Some(-offset);
                track_map.get_mut(&g.reposition(last)).unwrap().plus = Some(offset);
            }
        }
        Ok(())
    }

    //modifies the original arm to have the new instructions
    //returns the length of the tape (repetition size)
    fn normalize_instructions(
        original: &mut Arm,
        track_map: &TrackMap,
    ) -> Result<usize> {
        use ArmType::*;
        use Instr::*;
        let mut arm = original.clone();
        let arm_type = arm.arm_type;
        let old_instructions = &original.instruction_tape.instructions;
        let mut instructions = Vec::with_capacity(old_instructions.len() + 8);
        let mut last_repeat = 0;
        let mut curr = 0;
        let mut track_steps = 0;
        while curr < old_instructions.len() {
            let instr = old_instructions[curr];
            if !matches!(instr, Repeat | Reset | Noop) {
                instructions.push(instr);
                curr += 1;
            }
            let mut basic_move = |instr: Instr|->Result<()>{
                match instr{
                    Extend => {
                        if arm_type == Piston && arm.len < 3 {
                            arm.len += 1;
                        }
                    }
                    Retract => {
                        if arm_type == Piston && arm.len > 1 {
                            arm.len -= 1;
                        }
                    }
                    RotateCounterClockwise => {
                        arm.rot += 1;
                    }
                    RotateClockwise => {
                        arm.rot -= 1;
                    }
                    Grab => arm.grabbing = true,
                    Drop => arm.grabbing = false,
                    Forward => {
                        let track_data = track_map.get(&arm.pos)
                            .ok_or(eyre!("Forward movement not on track (preprocess)"))?;
                        arm.pos += track_data.plus.unwrap_or_default();
                        track_steps += 1;
                    }
                    Back => {
                        let track_data = track_map.get(&arm.pos)
                            .ok_or(eyre!("Backward movement not on track (preprocess)"))?;
                        arm.pos += track_data.minus.unwrap_or_default();
                        track_steps -= 1;
                    }
                    PivotCounterClockwise | PivotClockwise | Empty => {}
                    Reset | Repeat | Noop => {bail!("Instruction {:?} not basic move!",instr);}
                }
                Ok(())
            };
            match instr {
                Extend|Retract|
                RotateCounterClockwise|RotateClockwise|
                Grab|Drop|
                Forward|Back|
                PivotCounterClockwise|PivotClockwise|
                Empty => {
                    basic_move(instr)?;
                }

                Repeat => {
                    let rep_len = curr - last_repeat;
                    if rep_len == 0 {
                        instructions.push(Empty);
                        curr += 1;
                    } else {
                        for i in 0..rep_len {
                            ensure!( i == 0|| old_instructions.get(curr + i).unwrap_or(&Empty) == &Empty,
                                "Repeat instruction overlaps with {:?} on {}/{}/{}",
                                instructions[curr + i],curr,last_repeat,i
                            );
                            let copied_instr = instructions[last_repeat + i];
                            instructions.push(copied_instr);
                            basic_move(copied_instr)?;
                        }
                        last_repeat = curr;
                        curr += rep_len;
                    }
                }
                Reset => {
                    let mut reset_vec = Vec::new();
                    if arm.grabbing {
                        reset_vec.push(Drop)
                    };
                    while arm.len > original.len {
                        reset_vec.push(Retract);
                        arm.len -= 1;
                    }
                    let mut rot_tmp = normalize_dir(original.rot - arm.rot + 3) - 3;
                    while rot_tmp > 0 {
                        reset_vec.push(RotateCounterClockwise);
                        rot_tmp -= 1;
                    }
                    while rot_tmp < 0 {
                        reset_vec.push(RotateClockwise);
                        rot_tmp += 1;
                    }
                    arm.rot = original.rot;
                    // look for a path forward on the track that's shorter than
                    // the path backward.
                    if track_steps > 0 {
                        let mut tmp_pos = arm.pos;
                        'path_forward: for i in 1..track_steps {
                            let track_data = track_map
                                .get(&arm.pos)
                                .ok_or(eyre!("Reset track while not on track (Forward)"))?;
                            if let Some(offset) = track_data.plus {
                                tmp_pos += offset
                            } else {
                                break 'path_forward;
                            }
                            if tmp_pos == original.pos {
                                track_steps = -i;
                                break 'path_forward;
                            }
                        }
                    } else if track_steps < 0 {
                        let mut tmp_pos = arm.pos;
                        'path_backward: for i in 1..-track_steps {
                            let track_data = track_map
                                .get(&arm.pos)
                                .ok_or(eyre!("Reset track while not on track (Backward)"))?;
                            if let Some(offset) = track_data.minus {
                                tmp_pos += offset
                            } else {
                                break 'path_backward;
                            }
                            if tmp_pos == original.pos {
                                track_steps = i;
                                break 'path_backward;
                            }
                        }
                    }
                    while track_steps > 0 {
                        reset_vec.push(Back);
                        track_steps -= 1;
                    }
                    while track_steps < 0 {
                        reset_vec.push(Forward);
                        track_steps += 1;
                    }
                    while arm.len < original.len {
                        reset_vec.push(Extend);
                        arm.len += 1;
                    }

                    if reset_vec.len() == 0 {
                        reset_vec.push(Empty)
                    };
                    for i in 0..reset_vec.len() {
                        ensure!(
                            i == 0 || old_instructions.get(curr + i).unwrap_or(&Empty) == &Empty,
                            "Reset instruction overlaps with {:?} on {}/{}",
                            instructions[last_repeat + i],
                            curr,
                            i
                        );
                        instructions.push(reset_vec[i]);
                    }
                    curr += reset_vec.len();
                }
                Noop => {
                    instructions.push(Empty);
                    curr += 1;
                }
            };
        }
        instructions.shrink_to_fit();
        let len = instructions.len();
        original.instruction_tape = Tape {
            first: original.instruction_tape.first,
            instructions,
        };
        Ok(len)
    }

    pub fn setup_sim(init: &InitialWorld) -> Result<Self> {
        let mut world = World {
            timestep: 0,
            atoms: WorldAtoms::new(),
            area_touched: Default::default(),
            glyphs: init.glyphs.clone(),
            arms: init.arms.clone(),
            repeat_length: 0,
            track_map: Default::default(),
            cost: 0,
            instruction_count: 0,
        };
        for g in &mut world.glyphs {
            use GlyphType::*;
            world.cost += match &g.glyph_type {
                Calcification | Bonding | Unbonding => 10,
                Animismus | Projection | Dispersion | Purification => 20,
                Duplication | Unification | TriplexBond => 20,
                MultiBond => 30,
                Disposal | Equilibrium | Output(_, _) | Input(_) | Conduit(_) => 0,
                Track(v) => (v.len() as i32) * 5,
            };
            if let Track(_) = &g.glyph_type {
                World::add_track(&mut world.track_map, g)?;
            }
            g.reposition_pattern(); //reposition input/output/conduits
        }
        for a in &mut world.arms {
            use ArmType::*;
            world.cost += match a.arm_type {
                PlainArm => 20,
                DoubleArm | TripleArm | HexArm | VanBerlo => 30,
                Piston => 40,
            };
            let instr_len = World::normalize_instructions(a, &world.track_map)?;
            world.instruction_count += a.instruction_tape.instructions.iter()
                .filter(|&&a| a != Instr::Empty).count() as i32;
            if world.repeat_length < instr_len {
                world.repeat_length = instr_len;
            }
            if a.arm_type == VanBerlo {
                use AtomType::*;
                a.grabbing = true;
                const ATOM_SETUP: [AtomType; 6] = [Salt, Water, Air, Salt, Fire, Earth];
                for i in 0..6 {
                    let pos = a.pos + rot_to_pos((i as Rot) + a.rot);
                    let atom_type = ATOM_SETUP[i];
                    let key = world.atoms.create_atom(Atom {
                        pos, atom_type, connections: [Bonds::BERLO_TYPE; 6],
                    });
                    a.atoms_grabbed[i] = key;
                }
            }
        }
        world.process_glyphs();
        Ok(world)
    }
    pub fn get_stats(&self) -> SolutionStats {
        let cycles = (self.timestep as i32) + 1;
        let cost = self.cost;
        let area = self.area_touched.len() as i32;
        let instructions = self.instruction_count;
        SolutionStats {cycles,cost,area,instructions}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rotation_tests() {
        for r in 0..6 {
            assert_eq!(offset(1, r), rotate(Pos::new(1, 0), r));
            assert_eq!(offset(2, r), rotate(Pos::new(2, 0), r));
            assert_eq!(rotate(Pos::new(3, 1), r + 1), rotate(Pos::new(-1, 4), r));
        }
    }
}
