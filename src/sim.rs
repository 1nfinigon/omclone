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
use std::collections::{VecDeque, HashMap};
use rustc_hash::{FxHashMap, FxHashSet};
use std::{f32::consts::PI,fmt,error};

#[derive(Debug)]
pub struct SimError{
    pub error_str: &'static str,
    pub location: XYPos,
}
impl fmt::Display for SimError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let hex = xy_to_pos(self.location);
        let hex_pos = pos_to_xy(hex);
        if nalgebra::distance_squared(&hex_pos,&self.location) < 0.1{
            write!(f, "{} at {:?}", self.error_str,hex)
        } else {
            write!(f, "{} near {:?} (at {:?})", self.error_str,hex,self.location)
        }
    }
}
impl error::Error for SimError{}
pub type SimResult<T> = std::result::Result<T,SimError>;
fn sim_error_pos(error_str: &'static str, location: Pos) -> SimError{
    SimError{error_str, location: pos_to_xy(location)}
}
pub type Rot = i32;
pub type Pos = Vector2<i32>;

#[derive(Copy, Clone, Debug, PartialEq, Eq, enum_primitive_derive::Primitive)]
#[repr(u8)]
pub enum Instr {
    RotateClockwise        = 01,
    RotateCounterClockwise = 02,
    Extend                 = 03,
    Retract                = 04,
    Grab                   = 05,
    Drop                   = 06,
    PivotClockwise         = 07,
    PivotCounterClockwise  = 08,
    Forward                = 09,
    Back                   = 10,
    Repeat                 = 11,
    Reset                  = 12,
    Noop                   = 13,
    Empty                  = 14,
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
    pub fn from_byte(input: u8) -> Option<Self> {
        use Instr::*;
        match input {
            b'R' => Some(RotateClockwise),
            b'r' => Some(RotateCounterClockwise),
            b'E' => Some(Extend),
            b'e' => Some(Retract),
            b'G' => Some(Grab),
            b'g' => Some(Drop),
            b'P' => Some(PivotClockwise),
            b'p' => Some(PivotCounterClockwise),
            b'A' => Some(Forward),
            b'a' => Some(Back),
            b'C' => Some(Repeat),
            b'X' => Some(Reset),
            b'O' => Some(Noop),
            b' ' => Some(Empty),
            _ => None,
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
        if timestep >= self.first && loop_len > 0{
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

pub fn pos_to_rot(input: Pos) -> Option<Rot> {
    match (input.x, input.y) {
        ( 1, 0) => Some(0),
        ( 0, 1) => Some(1),
        (-1, 1) => Some(2),
        (-1, 0) => Some(3),
        ( 0,-1) => Some(4),
        ( 1,-1) => Some(5),
        _ => None,
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
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, enum_primitive_derive::Primitive)]
#[repr(u8)]
pub enum AtomType {
    ConduitSpace = 0,
    Salt=1, Air=2, Earth=3, Fire=4, Water=5,
    Quicksilver=6, Gold=7, Silver=8, Copper=9, Iron=10, Tin=11, Lead=12,
    Vitae=13, Mors=14, RepeatingOutputMarker=15, Quintessence=16,
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
    pub is_berlo: bool,
}
impl Atom {
    pub fn new(pos: Pos, atom_type: AtomType) -> Atom {
        Atom {
            pos,
            atom_type,
            connections: [Bonds::NO_BOND; 6],
            is_berlo: false,
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
}
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ArmMovement{
    Move(Movement),
    Pivot(Rot),
    LengthAdjust(i32),
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
    pub fn angles_between_arm(arm_type: ArmType) -> Rot {
        use ArmType::*;
        match arm_type {
            PlainArm => 6,
            DoubleArm => 3,
            TripleArm => 2,
            HexArm => 1,
            Piston => 6,
            VanBerlo => 1,
        }
    }
    fn do_motion(&mut self, action: ArmMovement) {
        use ArmMovement::*;
        use Movement::*;
        match action{
            Move(Linear(p)) => self.pos += p,
            Move(Rotation(r,_)) => self.rot += r,
            LengthAdjust(a) => self.len += a,
            Move(HeldStill) | Pivot(_) => {},
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
new_key_type! { pub struct AtomKey; }
#[derive(Debug, Clone)]
pub struct WorldAtoms {
    pub atom_map: SlotMap<AtomKey, Atom>,
    pub locs: FxHashMap<Pos, AtomKey>,
}
impl WorldAtoms {
    fn new() -> WorldAtoms {
        WorldAtoms {
            atom_map: SlotMap::with_key(),
            locs: Default::default(),
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
    fn get_consumable_type(&self, motion: &WorldStepInfo, loc: Pos) -> Option<AtomType> {
        let &key = self.locs.get(&loc)?;
        if motion.atoms.contains_key(key) {
            return None;
        }
        let atom = self.atom_map.get(key).expect("Inconsistent atoms (consume check)");
        if atom.connections.iter().any(|bonds| !bonds.is_empty()){
            return None
        }
        //Don't need to check is_berlo since the berlo arm will always be grabbing it (applying a motion)
        Some(atom.atom_type)
    }

    //Note: only use on non-grabbed atoms!
    fn destroy_atom_at(&mut self, loc: Pos) {
        let atom_key = self.locs.remove(&loc).expect("Destroying nonatom!");
        let atom = self.atom_map.remove(atom_key)
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

pub struct WorldStepInfo{
    pub atoms: SecondaryMap<AtomKey, Movement>,
    pub arms: Vec<ArmMovement>,
    pub spawning_atoms: Vec<Atom>,
}
impl WorldStepInfo{
    pub fn new() -> Self{
        WorldStepInfo{atoms: SecondaryMap::new(),arms: Vec::new(), spawning_atoms:Vec::new()}
    }
    pub fn clear(&mut self) {
        self.atoms.clear();
        self.arms.clear();
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct SolutionStats {
    pub cycles: i32,
    pub cost: i32,
    pub area: i32,
    pub instructions: i32,
}

pub type XYPos = nalgebra::Point2<f32>;
pub type XYVec = nalgebra::Vector2<f32>;
pub struct FloatAtom{
    pub pos: XYPos,
    pub rot: f32,
    pub atom_type: AtomType,
    pub connections: [Bonds; 6],
}
pub struct FloatArm{
    pub pos: XYPos,
    pub rot: f32,
    pub len: f32,
    pub arm_type: ArmType,
    pub grabbing: bool,
}
pub struct FloatWorld{//Might add arms, maybe some other animation stuff too
    pub portion: f32,
    pub atoms_xy: Vec<FloatAtom>,
    pub arms_xy: Vec<FloatArm>
}
fn pos_to_xy(input: Pos) -> XYPos{
    let a = input.x as f32;
    let b = input.y as f32;
    XYPos::new(a*2.+b,b*f32::sqrt(3.))
}
fn rot_to_angle(r: Rot) -> f32{
    (-r as f32)*PI/3.
}
fn xy_to_simple_pos(input: XYPos) -> Pos{
    let b = input.y/f32::sqrt(3.);
    let a = (input.x-b)/2.;
    Pos::new(a.round() as i32, b.round() as i32)
}
//https://stackoverflow.com/questions/7705228/hexagonal-grids-how-do-you-find-which-hexagon-a-point-is-in/23370350#23370350
fn xy_to_pos(input: XYPos) -> Pos{
    //x=a*2.+b, y=b*f32::sqrt(3.)
    let b = input.y/f32::sqrt(3.);
    let a = (input.x-b)/2.;
    // check closest to (p+1,q), (p,q+1), (p-1,q) or (p,q-1)
    let (p, q) = (a.round(), b.round());
    let (lambda, mu) = (a-p, b-q);
    // opposite signs, so we are guaranteed to be inside hexagon (p,q)
    let base_hex = Pos::new(p as i32, q as i32);
    if lambda * mu < 0.0 {return base_hex;}
    let distance = nalgebra::distance_squared(&pos_to_xy(base_hex),&input);
    // inside circle, so guaranteed inside hexagon (p,q)
    if distance < 1. {return base_hex;}

    // same sign, but which end of the parallelogram are we?
    let sign = lambda.signum();
    let candidate = if ( lambda.abs() > mu.abs() ) {(p+sign,q)} else {(p,q+sign)};
    let candidate_hex = Pos::new(candidate.0 as i32, candidate.1 as i32);
    let distance2 = nalgebra::distance_squared(&pos_to_xy(candidate_hex),&input);
    if distance < distance2 {base_hex} else {candidate_hex}
}
impl FloatWorld{
    pub fn new() -> Self{
        FloatWorld{
            portion: 0.,
            atoms_xy: Vec::new(),
            arms_xy: Vec::new(),
        }
    }
    pub fn regenerate(&mut self, world: &World, motion: &WorldStepInfo, portion: f32) {
        self.arms_xy.clear();
        self.atoms_xy.clear();
        self.portion = portion;
        use Movement::*;
        use ArmMovement::*;
        fn apply_movement(base: Pos, movement: Movement, amount: f32) -> (XYPos, f32){
            let xy = pos_to_xy(base);
            match movement{
                Linear(offset) => {
                    let offset_xy = pos_to_xy(offset)-XYPos::origin();
                    return (xy+(offset_xy*amount), 0.)
                },
                Rotation(r, pivot) => {
                    let pivot_xy = pos_to_xy(pivot);
                    let offset = xy-pivot_xy;
                    let rot = rot_to_angle(r)*amount;
                    return (pivot_xy+(nalgebra::Rotation2::new(-rot)*offset), rot)
                }
                HeldStill => {
                    return (xy, 0.)
                }
            }
        }
        for (atom_key, atom) in &world.atoms.atom_map{
            let movement = motion.atoms.get(atom_key).unwrap_or(&HeldStill);
            let (xy, angle) = apply_movement(atom.pos, *movement, portion);
            let f_atom = FloatAtom{
                pos: xy,
                rot: angle,
                atom_type: atom.atom_type,
                connections: atom.connections,
            };
            self.atoms_xy.push(f_atom);
        }
        assert_eq!(motion.arms.len(), world.arms.len());
        for a_index in 0..world.arms.len(){
            let arm = &world.arms[a_index];
            let mut arm_len = arm.len as f32;
            let motion = motion.arms[a_index];
            let new_motion = match motion{
                Move(m) => m,
                LengthAdjust(l) => {
                    arm_len += (l as f32)*portion;
                    HeldStill
                },
                Pivot(_) => HeldStill,
            };
            let (xy, angle) = apply_movement(arm.pos, new_motion, portion);
            let new_arm = FloatArm{
                pos: xy,
                rot: angle+rot_to_angle(arm.rot),
                len: arm_len*2.,
                arm_type: arm.arm_type,
                grabbing: arm.grabbing,
            };
            self.arms_xy.push(new_arm);
        }
    }
    pub fn generate_static(&mut self, world: &World) {
        self.arms_xy.clear();
        self.atoms_xy.clear();
        for (_atom_key, atom) in &world.atoms.atom_map{
            let xy = pos_to_xy(atom.pos);
            let f_atom = FloatAtom{
                pos: xy,
                rot: 0.,
                atom_type: atom.atom_type,
                connections: atom.connections,
            };
            self.atoms_xy.push(f_atom);
        }
        for a_index in 0..world.arms.len(){
            let arm = &world.arms[a_index];
            let arm_len = arm.len as f32;
            let new_arm = FloatArm{
                pos: pos_to_xy(arm.pos),
                rot: rot_to_angle(arm.rot),
                len: arm_len*2.,
                arm_type: arm.arm_type,
                grabbing: arm.grabbing,
            };
            self.arms_xy.push(new_arm);
        }
    }
}

//Running the world
impl World {
    //sets up a move for the specified atom and all atoms connected to it
    fn premove_atoms(&self, motion: &mut WorldStepInfo, atom_key: AtomKey, movement: Movement) -> SimResult<()> {
        let mut moving_atoms = VecDeque::<AtomKey>::new();
        moving_atoms.push_back(atom_key);

        while let Some(this_key) = moving_atoms.pop_front() {
            let maybe_move = motion.atoms.get(this_key);
            if let Some(curr_move) = maybe_move {
                if curr_move != &movement {
                    let error_str = &"Atom moved in multiple directions!";
                    return Err(sim_error_pos(error_str,self.atoms.atom_map[this_key].pos));
                }
            } else {
                motion.atoms.insert(this_key, movement);
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
    fn apply_motion(&mut self, motion: &WorldStepInfo,) -> SimResult<()> {
        for i in 0..self.arms.len() {
            self.arms[i].do_motion(motion.arms[i]);
        }
        //remove all grabbed atoms from the grid
        for (atom_key, _action) in motion.atoms.iter() {
            let atom = &mut self.atoms.atom_map[atom_key];
            let loc_check = self.atoms.locs.remove(&atom.pos);
            if loc_check != Some(atom_key) {
                panic!("Inconsistent atoms (move loc)!");
            }
        }
        //add them back in at their new locations
        for (atom_key, action) in motion.atoms.iter() {
            let atom = &mut self.atoms.atom_map[atom_key];
            match action {
                Movement::Linear(p) => atom.pos += p,
                Movement::Rotation(r, p) => {
                    atom.pos = rotate_around(atom.pos, *r, *p);
                    atom.rotate_connections(*r);
                }
                Movement::HeldStill => (),
            }
            let current = self.atoms.locs.insert(atom.pos, atom_key);
            if current != None {
                let error_str = &"Atom moved in multiple directions!";
                return Err(sim_error_pos(error_str,atom.pos));
            }
        }
        Ok(())
    }

    //Returns the movement the arm is going to do. Returns size of the largest molecule rotated
    fn do_instruction(&mut self, motion: &mut WorldStepInfo, arm_id: usize, timestep: u64) -> SimResult<()>{
        assert_eq!(motion.arms.len(), arm_id, "Arm motion array not same as arm ID");
        let arm = &mut self.arms[arm_id];
        use ArmType::*;
        use Instr::*;
        use Movement::*;
        use ArmMovement::*;
        let arm_type = arm.arm_type;
        let tape = &arm.instruction_tape;
        let timestep = timestep as usize;
        let instruction = tape.get(timestep, self.repeat_length);
        const STILL: ArmMovement = Move(HeldStill);
        let action = match instruction {
            Extend => {
                if arm_type == Piston && arm.len < 3 {
                    LengthAdjust(1)
                } else { STILL }
            }
            Retract => {
                if arm_type == Piston && arm.len > 1 {
                    LengthAdjust(-1)
                } else { STILL }
            }
            RotateCounterClockwise => {
                Move(Rotation(1, arm.pos))
            }
            RotateClockwise => {
                Move(Rotation(-1, arm.pos))
            }
            PivotCounterClockwise => Pivot(1),
            PivotClockwise => Pivot(-1),
            Drop => {
                if arm_type != VanBerlo {
                    arm.atoms_grabbed = [AtomKey::null(); 6];
                    arm.grabbing = false;
                }
                STILL
            }
            Grab => {
                if !arm.grabbing && arm_type != VanBerlo {
                    arm.grabbing = true;
                    for r in (0..6).step_by(Arm::angles_between_arm(arm.arm_type) as usize) {
                        let grab_pos = arm.pos + (rot_to_pos(arm.rot + r) * arm.len);
                        let null_key = AtomKey::null();
                        let mut current = self.atoms.locs.get(&grab_pos).unwrap_or(&null_key);
                        if current != &null_key{
                            if self.atoms.atom_map[*current].is_berlo{
                                current = &null_key;
                            }
                        }
                        arm.atoms_grabbed[r as usize] = *current;
                    }
                }
                STILL
            }
            Forward => {
                let track_data = self.track_map.get(&arm.pos)
                    .ok_or(sim_error_pos("Forward movement not on track", arm.pos))?;
                track_data.plus.map_or(STILL, |x| {
                    Move(Linear(x))
                })
            }
            Back => {
                let track_data = self.track_map.get(&arm.pos)
                    .ok_or(sim_error_pos("Backward movement not on track", arm.pos))?;
                track_data.minus.map_or(STILL, |x| {
                    Move(Linear(x))
                })
            }
            Repeat | Reset | Noop => {
                panic!("Unprocessed instruction!");
            }
            Empty => (STILL),
        };
        let rotation_store = arm.rot;
        for atom in arm.atoms_grabbed {
            if !atom.is_null() {
                let atom_movement = match action{
                    ArmMovement::Pivot(r) => Movement::Rotation(r, self.atoms.atom_map[atom].pos),
                    ArmMovement::Move(m) => m,
                    ArmMovement::LengthAdjust(a) => Linear(rot_to_pos(rotation_store)*a)
                };
                self.premove_atoms(motion, atom, atom_movement)?;
            }
        }
        motion.arms.push(action);
        Ok(())
    }

    fn initial_area(&mut self) -> SimResult<()>{
        use GlyphType::*;
        for glyph in &mut self.glyphs {
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
            fn add_all(area_set: &mut FxHashSet<Pos>,input: impl IntoIterator<Item=Pos>) -> SimResult<()>{
                for p in input{
                    if !area_set.insert(p){
                        return Err(SimError{error_str:"Overlap detected!", location:pos_to_xy(p)});
                    }
                }
                Ok(())
            }
            match &glyph.glyph_type{
                Calcification => {
                    add_all(&mut self.area_touched,[pos])
                }
                Animismus => {
                    add_all(&mut self.area_touched,[pos, pos_bi, pos_tri, pos_ani])
                }
                Projection => {
                    add_all(&mut self.area_touched,[pos, pos_bi])
                }
                Dispersion => {
                    add_all(&mut self.area_touched,[pos, pos_bi,pos_disp2,pos_disp3,pos_disp4])
                }
                Unification => {
                    add_all(&mut self.area_touched,[pos, pos_tri,pos_unif2,pos_unif3,pos_unif4])
                }
                Purification => {
                    add_all(&mut self.area_touched,[pos, pos_bi, pos_tri])
                }
                Duplication => {
                    add_all(&mut self.area_touched,[pos, pos_bi])
                }
                Bonding => {
                    add_all(&mut self.area_touched,[pos, pos_bi])
                }
                MultiBond => {
                    add_all(&mut self.area_touched,[pos, pos_bi,pos_multi2,pos_multi3])
                }
                TriplexBond => {
                    add_all(&mut self.area_touched,[pos, pos_bi, pos_tri])
                }
                Unbonding => {
                    add_all(&mut self.area_touched,[pos, pos_bi])
                }
                Input(atom_points) | Output(atom_points, _)=> {
                    add_all(&mut self.area_touched,atom_points.iter().map(|a|a.pos))
                }
                Track(pos_list) => {
                    add_all(&mut self.area_touched,pos_list.iter().map(|a|glyph.reposition(*a)))
                },
                Disposal => {
                    add_all(&mut self.area_touched,[pos, pos_bi, pos_tri, pos_ani, pos_disp2,pos_disp3,pos_disp4,pos_unif2])
                },
                Conduit(_atom_teleport) => {
                    Ok(())//TODO
                },
                Equilibrium => {
                    add_all(&mut self.area_touched,[pos])
                }
            }?;
        }
        Ok(())
    }
    fn process_glyphs(&mut self, motion: &mut WorldStepInfo) {
        fn try_bond(atoms: &mut WorldAtoms, loc1: &Pos, loc2: &Pos) {
            let rot = pos_to_rot(loc2 - loc1).unwrap() as usize;
            if let (Some(&key1), Some(&key2)) = (atoms.locs.get(loc1), atoms.locs.get(loc2)) {
                let [atom1,atom2] = atoms.atom_map.get_disjoint_mut([key1,key2]).expect("Inconsistent atoms!");
                if (atom1.is_berlo || atom2.is_berlo) {return;}
                let bond1 = atom1.connections[rot];
                assert_eq!(atom2.connections[(rot + 3) % 6], bond1, "Inconsistent bonds");
                if bond1 == Bonds::NO_BOND {
                    atom1.connections[rot] = Bonds::NORMAL;
                    atom2.connections[(rot + 3) % 6] = Bonds::NORMAL;
                }
            }
        }
        fn try_triplex_bond(atoms: &mut WorldAtoms, loc1: &Pos, loc2: &Pos, bond_type: Bonds ) {
            let rot = pos_to_rot(loc2 - loc1).unwrap() as usize;
            if let (Some(&key1), Some(&key2)) = (atoms.locs.get(loc1), atoms.locs.get(loc2)) {
                let [atom1,atom2] = atoms.atom_map.get_disjoint_mut([key1,key2]).expect("Inconsistent atoms!");
                if (atom1.is_berlo || atom2.is_berlo) {return;}
                if (atom1.atom_type != AtomType::Fire || atom2.atom_type != AtomType::Fire) {return;}
                let bond1 = atom1.connections[rot];
                assert_eq!(atom2.connections[(rot + 3) % 6], bond1, "Inconsistent bonds");
                if (bond1 & !Bonds::TRIPLEX).is_empty() {
                    atom1.connections[rot] |= bond_type;
                    atom2.connections[(rot + 3) % 6] |= bond_type;
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
                        if atom.atom_type.is_element() && !atom.is_berlo{
                            atom.atom_type = AtomType::Salt;
                        }
                    }
                }
                Animismus => {
                    let a1 = atoms.get_consumable_type(motion, pos);
                    let a2 = atoms.get_consumable_type(motion, pos_bi);
                    let o1 = atoms.get_type(pos_tri);
                    let o2 = atoms.get_type(pos_ani);
                    if (Some(Salt),Some(Salt),None,None) == (a1,a2,o1,o2){
                        atoms.destroy_atom_at(pos);
                        atoms.destroy_atom_at(pos_bi);
                        motion.spawning_atoms.push(Atom::new(pos_tri, Vitae));
                        motion.spawning_atoms.push(Atom::new(pos_ani, Mors));
                    }
                }
                Projection => {
                    let qs = atoms.get_consumable_type(motion, pos);
                    let metal = atoms.get_type(pos_bi);
                    if let (Some(Quicksilver), Some(metal)) = (qs, metal){
                        if let Some(newtype) = metal.promotable_metal(){
                            atoms.destroy_atom_at(pos);
                            atoms.get_atom_mut(pos_bi).unwrap().atom_type = newtype;
                        }
                    }
                }
                Dispersion => {
                    let q = atoms.get_consumable_type(motion, pos);
                    let o1 = atoms.get_type(pos_bi);
                    let o2 = atoms.get_type(pos_disp2);
                    let o3 = atoms.get_type(pos_disp3);
                    let o4 = atoms.get_type(pos_disp4);
                    if (Some(Quintessence),None,None,None,None) == (q,o1,o2,o3,o4){
                        atoms.destroy_atom_at(pos);
                        motion.spawning_atoms.push(Atom::new(pos_bi, Earth));
                        motion.spawning_atoms.push(Atom::new(pos_disp2, Water));
                        motion.spawning_atoms.push(Atom::new(pos_disp3, Fire));
                        motion.spawning_atoms.push(Atom::new(pos_disp4, Air));
                    }
                }
                Unification => {
                    let output = atoms.get_type(pos);
                    let a1 = atoms.get_consumable_type(motion, pos_tri);
                    let a2 = atoms.get_consumable_type(motion, pos_unif2);
                    let a3 = atoms.get_consumable_type(motion, pos_unif3);
                    let a4 = atoms.get_consumable_type(motion, pos_unif4);
                    if let (None,Some(a),Some(b),Some(c),Some(d)) = (output, a1, a2, a3, a4){
                        let set = [a,b,c,d];
                        if set.contains(&Earth) && set.contains(&Water) && set.contains(&Fire) && set.contains(&Air){
                            atoms.destroy_atom_at(pos_tri);
                            atoms.destroy_atom_at(pos_unif2);
                            atoms.destroy_atom_at(pos_unif3);
                            atoms.destroy_atom_at(pos_unif4);
                            motion.spawning_atoms.push(Atom::new(pos, Quintessence));
                        }
                    }
                }
                Purification => {
                    let a1 = atoms.get_consumable_type(motion, pos);
                    let a2 = atoms.get_consumable_type(motion, pos_bi);
                    let o = atoms.get_type(pos_tri);
                    match (a1,a2,o){
                        (Some(a1),Some(a2),None) if a1 == a2 => {
                            let next = a1.promotable_metal();
                            if let Some(newtype) = next {
                                atoms.destroy_atom_at(pos);
                                atoms.destroy_atom_at(pos_bi);
                                motion.spawning_atoms.push(Atom::new(pos_tri, newtype));
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
                    try_bond(atoms, &pos, &pos_bi);
                }
                MultiBond => {
                    try_bond(atoms, &pos, &pos_bi);
                    try_bond(atoms, &pos, &pos_multi2);
                    try_bond(atoms, &pos, &pos_multi3);
                }
                TriplexBond => {
                    try_triplex_bond(atoms, &pos, &pos_bi, Bonds::TRIPLEX_K);
                    try_triplex_bond(atoms, &pos, &pos_tri, Bonds::TRIPLEX_R);
                    try_triplex_bond(atoms, &pos_bi, &pos_tri, Bonds::TRIPLEX_Y);
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
                                &atoms.atom_map[atom_key] == a && !motion.atoms.contains_key(atom_key)
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
                        && !motion.atoms.contains_key(atom_key) {
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


    pub fn mark_area_and_collide(&mut self, float_world: &FloatWorld, spawning_atoms: &[Atom]) -> SimResult<()>{
        //atom radius = 29/41 or 1/sqrt(2)
        //spawning atom = 15/41
        //arm base = 20/41
        //grabber, cabinet = 24/41, 20/41
        use smallvec::SmallVec;
        const ATOM_RADIUS: f32 = 29./41.;
        const ARM_RADIUS: f32 = 20./41.;
        const SPAWNING_ATOM_RADIUS: f32 = 15./41.;
        const ATOM_ATOM_RADIUS_SQUARED: f32 = (ATOM_RADIUS*2.)*(ATOM_RADIUS*2.);
        const ATOM_ARM_RADIUS_SQUARED: f32 = (ATOM_RADIUS+ARM_RADIUS)*(ATOM_RADIUS+ARM_RADIUS);
        const ATOM_SPAWN_RADIUS_SQUARED: f32 = (ATOM_RADIUS+SPAWNING_ATOM_RADIUS)*(ATOM_RADIUS+SPAWNING_ATOM_RADIUS);
        fn make_candidates(primary: Pos) -> [Pos;7]{
            [
                Pos::new(primary.x, primary.y),
                Pos::new(primary.x+1, primary.y),
                Pos::new(primary.x-1, primary.y),
                Pos::new(primary.x, primary.y+1),
                Pos::new(primary.x, primary.y-1),
                Pos::new(primary.x-1, primary.y+1),
                Pos::new(primary.x+1, primary.y-1),
            ]
        }
        fn mark_point(area_set: &mut FxHashSet<Pos>, point: XYPos){
            let primary = xy_to_simple_pos(point);
            let candidates = make_candidates(primary);
            for hex in candidates{
                let distance = nalgebra::distance_squared(&pos_to_xy(hex),&point);
                if distance < ATOM_ATOM_RADIUS_SQUARED{
                    area_set.insert(hex);
                }
            }
        }
        fn collide(atom_collisions: &mut FxHashMap<Pos, SmallVec<[XYPos;2]>>, point: XYPos, check_dist:f32, error_str: &'static str) -> SimResult<()>{
            let primary = xy_to_simple_pos(point);
            let candidates = make_candidates(primary);
            for check in candidates{
                let possible_vec = atom_collisions.get(&check);
                if let Some(atoms) = possible_vec{
                    for atom_point in atoms{
                        if nalgebra::distance_squared(atom_point,&point) < check_dist {
                            let error_point = nalgebra::center(atom_point, &point);
                            return Err(SimError{error_str, location:error_point});
                        }
                    }
                }
            }
            Ok(())
        }
        let mut atom_collisions = FxHashMap::with_capacity_and_hasher(float_world.atoms_xy.len(),Default::default());
        for atom in &float_world.atoms_xy{
            let primary = xy_to_simple_pos(atom.pos);
            mark_point(&mut self.area_touched, atom.pos);
            collide(&mut atom_collisions, atom.pos, ATOM_ATOM_RADIUS_SQUARED, "atom/atom collision!")?;
            atom_collisions.entry(primary).or_insert_with(SmallVec::new).push(atom.pos);
        }
        for arm in &float_world.arms_xy{
            mark_point(&mut self.area_touched, arm.pos);
            collide(&mut atom_collisions, arm.pos, ATOM_ARM_RADIUS_SQUARED,"atom/arm collision!")?;
            for r in (0..6).step_by(Arm::angles_between_arm(arm.arm_type) as usize) {
                let angle = arm.rot+rot_to_angle(r);
                let offset = nalgebra::Rotation2::new(-angle)*XYVec::new(arm.len, 0.);
                mark_point(&mut self.area_touched, arm.pos+offset);
                //arm lengths are doubled in floatworld
                if arm.len > 1.0{
                    let offset = nalgebra::Rotation2::new(-angle)*XYVec::new(2., 0.);
                    mark_point(&mut self.area_touched, arm.pos+offset);
                }
                if arm.len > 3.0{
                    let offset = nalgebra::Rotation2::new(-angle)*XYVec::new(4., 0.);
                    mark_point(&mut self.area_touched, arm.pos+offset);
                }
            }
        }
        for spawning_atom in spawning_atoms{
            collide(&mut atom_collisions, pos_to_xy(spawning_atom.pos), ATOM_SPAWN_RADIUS_SQUARED, "atom/spawning atom collision!")?;
        }
        Ok(())
    }

    pub fn run_step(&mut self, mark_area: bool, motion: &mut WorldStepInfo, float_world: &mut FloatWorld) -> SimResult<()> {
        self.prepare_step(motion)?;
        let substep_count = self.substep_count(motion);
        if mark_area{
            for substep in 0..substep_count{
                let portion = substep as f32 / substep_count as f32;
                float_world.regenerate(self, motion, portion);
                self.mark_area_and_collide(float_world, &motion.spawning_atoms)?;
            }
        }
        self.finalize_step(motion)?;
        Ok(())
    }
    //returns the step size to be used for this timestep
    pub fn prepare_step(&mut self, motion: &mut WorldStepInfo) -> SimResult<()> {
        motion.clear();
        for i in 0..self.arms.len() {
            self.do_instruction(motion, i, self.timestep)?;
        }
        self.process_glyphs(motion);
        Ok(())
    }
    pub fn substep_count(&self, motion: &WorldStepInfo) -> usize{
        let mut max_radius:f64 = 8.; //This corresponds to minimum step count of 8
        for (atom_key, movement) in &motion.atoms{
            if let Movement::Rotation(_rot, center) = movement{
                let atom_pos = self.atoms.atom_map[atom_key].pos;
                let distance = nalgebra::distance(&pos_to_xy(atom_pos),&pos_to_xy(*center)) as f64;
                if max_radius < distance {
                    max_radius = distance;
                }
            }
        }
        let fixed_radius = f64::powf(2.,max_radius.log2().round());
        fixed_radius as usize
    }
    pub fn finalize_step(&mut self, motion: &mut WorldStepInfo) -> SimResult<()> {
        self.apply_motion(motion)?;
        self.process_glyphs(motion);
        for atom in motion.spawning_atoms.drain(..){
            self.atoms.create_atom(atom);
        }
        motion.clear();
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
            if pos_to_rot(offset).is_some() {
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
        let mut repeat_source = 0;
        let mut repeat_ending = 0;
        let mut any_nonrepeat = false;
        let mut curr = 0;
        let mut track_steps = 0;
        while curr < old_instructions.len() {
            let instr = old_instructions[curr];
            if !any_nonrepeat && !matches!(instr, Repeat | Empty | Noop){
                any_nonrepeat = true;
                repeat_source = curr;
            }
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
                        if arm.pos == original.pos{
                            track_steps = 0;
                        }
                    }
                    Back => {
                        let track_data = track_map.get(&arm.pos)
                            .ok_or(eyre!("Backward movement not on track (preprocess)"))?;
                        arm.pos += track_data.minus.unwrap_or_default();
                        track_steps -= 1;
                        if arm.pos == original.pos{
                            track_steps = 0;
                        }
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
                PivotCounterClockwise|PivotClockwise => {
                    basic_move(instr)?;
                    repeat_ending = curr;
                }
                Empty => {}

                Repeat => {
                    let rep_len = repeat_ending - repeat_source;
                    if rep_len == 0 {
                        instructions.push(Empty);
                        curr += 1;
                    } else {
                        for i in 0..rep_len {
                            let copied_instr = instructions[repeat_source + i];
                            ensure!( i == 0|| old_instructions.get(curr + i).unwrap_or(&Empty) == &Empty,
                                "Repeat instruction {:?} overlaps with {:?} on {}/{}. Repeat {}-{}",
                                copied_instr,old_instructions.get(curr + i),
                                curr,i,repeat_source,repeat_ending
                            );
                            instructions.push(copied_instr);
                            basic_move(copied_instr)?;
                        }
                        curr += rep_len;
                        any_nonrepeat = false;
                    }
                }
                Reset => {
                    let mut reset_vec = Vec::new();
                    if arm.grabbing {
                        reset_vec.push(Drop);
                        arm.grabbing = false;
                    }
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
                    arm.pos = original.pos;
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
                            "Reset instruction {:?} overlaps with {:?} on {}/{} (curr {})",
                            reset_vec[i],
                            old_instructions.get(curr + i),
                            i, reset_vec.len(),
                            curr,
                        );
                        instructions.push(reset_vec[i]);
                    }
                    curr += reset_vec.len();
                    repeat_ending = curr;
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
                        pos, atom_type, connections: [Bonds::NO_BOND; 6], is_berlo: true
                    });
                    a.atoms_grabbed[i] = key;
                }
            }
        }
        
        for glyph in &mut world.glyphs {
            if let GlyphType::Input(atom_spawn_points) = &glyph.glyph_type {
                if atom_spawn_points.iter().all(|a| world.atoms.locs.get(&a.pos) == None){
                    for a in atom_spawn_points{
                        world.atoms.create_atom(a.clone());
                    }
                }
            }
        }
        world.initial_area()?;
        world.timestep = world.get_first_timestep();
        Ok(world)
    }
    fn get_first_timestep(&self) -> u64{
        self.arms.iter().map(|a|a.instruction_tape.first as u64).fold(u64::MAX, std::cmp::min)
    }
    pub fn get_stats(&self) -> SolutionStats {
        let cycles = (self.timestep-self.get_first_timestep()) as i32;
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
