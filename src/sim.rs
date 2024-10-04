use bitflags::bitflags;
use enum_primitive_derive::Primitive;
use eyre::{ensure, Result};
pub use nalgebra::{Point2, Vector2};
use rustc_hash::{FxHashMap, FxHashSet};
use slotmap::{new_key_type, Key, SecondaryMap, SlotMap};
use smallvec::SmallVec;
use std::collections::VecDeque;
use std::fmt::{Display, Write};
use std::{error, f32::consts::PI, fmt};

#[derive(Debug)]
pub struct SimError {
    pub error_str: &'static str,
    pub location: XYPos,
}
impl fmt::Display for SimError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let hex = xy_to_pos(self.location);
        let hex_pos = pos_to_xy(hex);
        if nalgebra::distance_squared(&hex_pos, &self.location) < 0.1 {
            write!(f, "{} at {:?}", self.error_str, hex)
        } else {
            write!(
                f,
                "{} near {:?} (at {:?})",
                self.error_str, hex, self.location
            )
        }
    }
}
impl error::Error for SimError {}
pub type SimResult<T> = std::result::Result<T, SimError>;
fn sim_error_pos(error_str: &'static str, location: Pos) -> SimError {
    SimError {
        error_str,
        location: pos_to_xy(location),
    }
}

/// Convention: +x right, +y up
pub type Pos = Vector2<i32>;

/// A basic instruction that can be executed in one timestep (i.e. no Repeat,
/// Reset, Noop)
#[derive(Copy, Clone, Debug, PartialEq, Eq, Primitive)]
#[repr(u8)]
pub enum BasicInstr {
    Empty = 0,
    RotateClockwise = 1,
    RotateCounterClockwise = 2,
    Extend = 3,
    Retract = 4,
    Grab = 5,
    Drop = 6,
    PivotClockwise = 7,
    PivotCounterClockwise = 8,
    Forward = 9,
    Back = 10,
}

impl BasicInstr {
    pub const N_TYPES: usize = 11;
}

#[test]
fn basic_instr_len() {
    use num_traits::FromPrimitive;
    assert!(BasicInstr::from_u8(BasicInstr::N_TYPES.try_into().unwrap()).is_none());
    assert!(BasicInstr::from_u8((BasicInstr::N_TYPES - 1).try_into().unwrap()).is_some());
}

impl BasicInstr {
    pub fn from_char(input: char) -> Option<Self> {
        match input {
            'd' => Some(Self::RotateClockwise),
            'a' => Some(Self::RotateCounterClockwise),
            'w' => Some(Self::Extend),
            's' => Some(Self::Retract),
            'f' => Some(Self::Grab),
            'r' => Some(Self::Drop),
            'e' => Some(Self::PivotClockwise),
            'q' => Some(Self::PivotCounterClockwise),
            'g' => Some(Self::Forward),
            't' => Some(Self::Back),
            ' ' => Some(Self::Empty),
            _ => None,
        }
    }
    pub fn to_char(self) -> char {
        let instr: Instr = self.into();
        instr.to_char()
    }
}

/// An instruction that can be written into the saved solution tape (i.e.
/// includes Repeat, Reset, Noop)
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Instr {
    Empty,
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
}

impl From<BasicInstr> for Instr {
    fn from(value: BasicInstr) -> Self {
        match value {
            BasicInstr::Empty => Self::Empty,
            BasicInstr::RotateClockwise => Self::RotateClockwise,
            BasicInstr::RotateCounterClockwise => Self::RotateCounterClockwise,
            BasicInstr::Extend => Self::Extend,
            BasicInstr::Retract => Self::Retract,
            BasicInstr::Grab => Self::Grab,
            BasicInstr::Drop => Self::Drop,
            BasicInstr::PivotClockwise => Self::PivotClockwise,
            BasicInstr::PivotCounterClockwise => Self::PivotCounterClockwise,
            BasicInstr::Forward => Self::Forward,
            BasicInstr::Back => Self::Back,
        }
    }
}

impl TryFrom<Instr> for BasicInstr {
    type Error = ();
    fn try_from(value: Instr) -> std::result::Result<Self, Self::Error> {
        match value {
            Instr::Empty => Ok(Self::Empty),
            Instr::RotateClockwise => Ok(Self::RotateClockwise),
            Instr::RotateCounterClockwise => Ok(Self::RotateCounterClockwise),
            Instr::Extend => Ok(Self::Extend),
            Instr::Retract => Ok(Self::Retract),
            Instr::Grab => Ok(Self::Grab),
            Instr::Drop => Ok(Self::Drop),
            Instr::PivotClockwise => Ok(Self::PivotClockwise),
            Instr::PivotCounterClockwise => Ok(Self::PivotCounterClockwise),
            Instr::Forward => Ok(Self::Forward),
            Instr::Back => Ok(Self::Back),
            Instr::Repeat => Err(()),
            Instr::Reset => Err(()),
            Instr::Noop => Err(()),
        }
    }
}

impl Instr {
    pub fn to_byte(self) -> u8 {
        use Instr::*;
        match self {
            RotateClockwise => b'R',
            RotateCounterClockwise => b'r',
            Extend => b'E',
            Retract => b'e',
            Grab => b'G',
            Drop => b'g',
            PivotClockwise => b'P',
            PivotCounterClockwise => b'p',
            Forward => b'A',
            Back => b'a',
            Repeat => b'C',
            Reset => b'X',
            Noop => b'O',
            Empty => b' ',
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
    pub fn to_char(self) -> char {
        use Instr::*;
        match self {
            RotateClockwise => 'd',
            RotateCounterClockwise => 'a',
            Extend => 'w',
            Retract => 's',
            Grab => 'f',
            Drop => 'r',
            PivotClockwise => 'e',
            PivotCounterClockwise => 'q',
            Forward => 'g',
            Back => 't',
            Repeat => 'C',
            Reset => 'X',
            Noop => 'O',
            Empty => ' ',
        }
    }
}

pub trait InstrToChar: Copy {
    fn to_char(self) -> char;
}

impl InstrToChar for BasicInstr {
    fn to_char(self) -> char {
        self.to_char()
    }
}

impl InstrToChar for Instr {
    fn to_char(self) -> char {
        self.to_char()
    }
}

#[derive(Debug, Clone)]
pub struct Tape<InstrT> {
    pub first: usize,
    pub instructions: Vec<InstrT>,
}

pub fn compute_tape_repeat_length<T>(tapes: &[Tape<T>]) -> usize {
    tapes
        .iter()
        .map(|tape| tape.instructions.len())
        .max()
        .unwrap_or(0)
}

pub fn compute_tape_instruction_count(tapes: &[Tape<BasicInstr>]) -> usize {
    tapes
        .iter()
        .map(|tape| {
            tape.instructions
                .iter()
                .filter(|&&a| a != BasicInstr::Empty)
                .count()
        })
        .sum()
}

impl<InstrT: Copy + From<BasicInstr>> Tape<InstrT> {
    pub fn get(&self, timestep: usize, loop_len: usize) -> InstrT {
        use BasicInstr::Empty;
        if timestep >= self.first && loop_len > 0 {
            let after_first = timestep - self.first;
            *self
                .instructions
                .get(after_first % loop_len)
                .unwrap_or(&Empty.into())
        } else {
            Empty.into()
        }
    }
}

impl<InstrT: InstrToChar> Display for Tape<InstrT> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&" ".repeat(self.first))?;
        for i in &self.instructions {
            f.write_char(i.to_char())?;
        }
        Ok(())
    }
}

impl<InstrT: Eq + From<BasicInstr> + InstrToChar> Tape<InstrT> {
    pub fn clear_leading_emptys(&mut self) {
        while self.instructions.first() == Some(&BasicInstr::Empty.into()) {
            self.first += 1;
            self.instructions.remove(0);
        }
    }

    pub fn noop_clear_and_string(&mut self) -> String {
        self.clear_leading_emptys();
        self.to_string()
    }
}

/// A rotation. Positive is CCW. Not normalized.
#[derive(Copy, Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct RawRot(pub i32);

impl RawRot {
    pub fn normalize(self) -> Rot {
        Rot::from_i32(self.0)
    }
}

impl std::ops::Neg for RawRot {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

impl std::ops::Add for RawRot {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl std::ops::AddAssign for RawRot {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl std::ops::Sub for RawRot {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl std::ops::SubAssign for RawRot {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

impl std::ops::Add<i32> for RawRot {
    type Output = Self;
    fn add(self, rhs: i32) -> Self::Output {
        Self(self.0 + rhs)
    }
}

impl std::ops::AddAssign<i32> for RawRot {
    fn add_assign(&mut self, rhs: i32) {
        self.0 += rhs;
    }
}

/// +ve CCW. Guaranteed to be normalized to `0..6`
#[derive(Copy, Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct Rot(i8);

impl Rot {
    pub const ALL: [Rot; 6] = {
        let mut r = [Rot(0); 6];
        let mut i = 0;
        while i < 6 {
            r[i].0 = i as i8;
            i += 1;
        }
        r
    };

    /// Returns the diametrically opposing direction
    pub fn opp(self) -> Self {
        Self::from_i8(self.0 + 3)
    }

    pub fn step_by(step: usize) -> impl Iterator<Item = Self> {
        (0..6).step_by(step).map(Rot)
    }

    /// Returns a value in the range `0..6`
    pub fn raw_nonneg(self) -> RawRot {
        RawRot(self.0 as i32)
    }

    /// Returns a value in the range `-3..3`
    pub fn raw_minabs(self) -> RawRot {
        RawRot(self.to_i32_minabs())
    }

    pub fn from_i8(value: i8) -> Self {
        Self(value.rem_euclid(6))
    }

    pub fn from_i32(value: i32) -> Self {
        Self(value.rem_euclid(6) as i8)
    }

    /// Returns a value in the range `-3..3`
    pub fn to_i32_minabs(self) -> i32 {
        if self.0 >= 3 {
            (self.0 - 6).into()
        } else {
            self.0.into()
        }
    }

    pub fn to_usize(self) -> usize {
        self.0
            .try_into()
            .expect("Rot should have been normalized to range 0..6")
    }

    pub fn from_unit_pos(input: Pos) -> Option<Rot> {
        match (input.x, input.y) {
            (1, 0) => Some(Rot(0)),
            (0, 1) => Some(Rot(1)),
            (-1, 1) => Some(Rot(2)),
            (-1, 0) => Some(Rot(3)),
            (0, -1) => Some(Rot(4)),
            (1, -1) => Some(Rot(5)),
            _ => None,
        }
    }

    pub fn dist_to_pos(self, n: i32) -> Pos {
        match self.0 {
            0 => Pos::new(n, 0),
            1 => Pos::new(0, n),
            2 => Pos::new(-n, n),
            3 => Pos::new(-n, 0),
            4 => Pos::new(0, -n),
            5 => Pos::new(n, -n),
            _ => panic!("Invalid Rotation"),
        }
    }

    pub fn to_pos(self) -> Pos {
        self.dist_to_pos(1)
    }
}

impl std::ops::Neg for Rot {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self::from_i8(-self.0)
    }
}

impl std::ops::Add for Rot {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self::from_i8(self.0 + rhs.0)
    }
}

impl std::ops::AddAssign for Rot {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl std::ops::Sub for Rot {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self::from_i8(self.0 - rhs.0)
    }
}

impl std::ops::SubAssign for Rot {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl std::ops::Add<i8> for Rot {
    type Output = Self;
    fn add(self, rhs: i8) -> Self::Output {
        Self::from_i8(self.0 + rhs)
    }
}

impl std::ops::AddAssign<i8> for Rot {
    fn add_assign(&mut self, rhs: i8) {
        *self = *self + rhs;
    }
}

pub fn rotate(pos: Pos, angle: Rot) -> Pos {
    match angle.0 {
        0 => pos, //  ( pos.x      ,       pos.y)
        1 => Pos::new(-pos.y, pos.x + pos.y),
        2 => Pos::new(-pos.x - pos.y, pos.x),
        3 => Pos::new(-pos.x, -pos.y),
        4 => Pos::new(pos.y, -pos.x - pos.y),
        5 => Pos::new(pos.x + pos.y, -pos.x),
        _ => panic!("Invalid Rotation"),
    }
}

pub fn rotate_around(pos: Pos, angle: Rot, pivot: Pos) -> Pos {
    rotate(pos - pivot, angle) + pivot
}

/// aka "bestagon" metric
pub fn hex_distance(pos: Pos) -> u32 {
    let xy = pos.x + pos.y;
    [pos.x.abs(), pos.y.abs(), xy.abs()]
        .into_iter()
        .max()
        .unwrap()
        .try_into()
        .unwrap()
}

#[test]
fn test_hex_distance() {
    assert_eq!(hex_distance(Pos::new(0, 0)), 0);
    assert_eq!(hex_distance(Pos::new(0, 3)), 3);
    assert_eq!(hex_distance(Pos::new(0, -3)), 3);
    assert_eq!(hex_distance(Pos::new(3, 0)), 3);
    assert_eq!(hex_distance(Pos::new(-3, 0)), 3);
    assert_eq!(hex_distance(Pos::new(3, -3)), 3);
    assert_eq!(hex_distance(Pos::new(-3, 3)), 3);
    assert_eq!(hex_distance(Pos::new(-3, 2)), 3);
}

bitflags! {
    #[derive(PartialEq, Eq, Debug, Clone, Copy)]
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

#[derive(Copy, Clone, Debug, PartialEq, Eq, Primitive)]
#[repr(u8)]
pub enum AtomType {
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
}

impl AtomType {
    pub const ALL_DYNAMIC: [Self; 15] = [
        Self::Salt,
        Self::Air,
        Self::Earth,
        Self::Fire,
        Self::Water,
        Self::Quicksilver,
        Self::Gold,
        Self::Silver,
        Self::Copper,
        Self::Iron,
        Self::Tin,
        Self::Lead,
        Self::Vitae,
        Self::Mors,
        Self::Quintessence,
    ];

    pub fn is_element(&self) -> bool {
        use AtomType::*;
        matches!(&self, Air | Earth | Fire | Water)
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
        for i in Rot::ALL {
            new_connections[(i + r).to_usize()] = self.connections[i.to_usize()];
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
pub enum ArmMovement {
    Move(Movement),
    Pivot(Rot),
    LengthAdjust(i32),
}

pub type AtomPattern = Vec<Atom>;
pub type InOutId = i32;
pub type ConduitId = i32;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum BasicGlyphType {
    Calcification,
    Animismus,
    Projection,
    Dispersion,
    Purification,
    Duplication,
    Unification,
    Bonding,
    Unbonding,
    TriplexBond,
    MultiBond,
    Disposal,
    Equilibrium,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GlyphType {
    Calcification,
    Animismus,
    Projection,
    Dispersion,
    Purification,
    Duplication,
    Unification,
    Bonding,
    Unbonding,
    TriplexBond,
    MultiBond,
    Disposal,
    Equilibrium,
    Track {
        locs: Vec<Pos>,
    },
    Conduit {
        locs: Vec<Pos>,
        id: ConduitId,
    },
    Input {
        pattern: AtomPattern,
        id: InOutId,
    },
    Output {
        pattern: AtomPattern,
        count: i32,
        id: InOutId,
    },
    OutputRepeating {
        pattern: AtomPattern,
        count: i32,
        id: InOutId,
    },
}

impl GlyphType {
    pub fn move_by(&mut self, pos: Pos) {
        use GlyphType::*;
        match self {
            Calcification | Animismus | Projection | Dispersion | Purification | Duplication
            | Unification | Bonding | Unbonding | TriplexBond | MultiBond | Disposal
            | Equilibrium => (),
            Track { locs } | Conduit { locs, .. } => {
                for a in locs {
                    *a += pos;
                }
            }
            Input { pattern, .. } | Output { pattern, .. } | OutputRepeating { pattern, .. } => {
                for a in pattern {
                    a.pos += pos;
                }
            }
        }
    }
    pub fn rot_by(&mut self, rot: RawRot) {
        use GlyphType::*;
        match self {
            Calcification | Animismus | Projection | Dispersion | Purification | Duplication
            | Unification | Bonding | Unbonding | TriplexBond | MultiBond | Disposal
            | Equilibrium => (),
            Track { locs } | Conduit { locs, .. } => {
                for a in locs {
                    *a = rotate(*a, rot.normalize());
                }
            }
            Input { pattern, .. } | Output { pattern, .. } | OutputRepeating { pattern, .. } => {
                for a in pattern {
                    a.pos = rotate(a.pos, rot.normalize());
                    a.rotate_connections(rot.normalize());
                }
            }
        }
    }
}

impl From<BasicGlyphType> for GlyphType {
    fn from(value: BasicGlyphType) -> Self {
        match value {
            BasicGlyphType::Calcification => Self::Calcification,
            BasicGlyphType::Animismus => Self::Animismus,
            BasicGlyphType::Projection => Self::Projection,
            BasicGlyphType::Dispersion => Self::Dispersion,
            BasicGlyphType::Purification => Self::Purification,
            BasicGlyphType::Duplication => Self::Duplication,
            BasicGlyphType::Unification => Self::Unification,
            BasicGlyphType::Bonding => Self::Bonding,
            BasicGlyphType::Unbonding => Self::Unbonding,
            BasicGlyphType::TriplexBond => Self::TriplexBond,
            BasicGlyphType::MultiBond => Self::MultiBond,
            BasicGlyphType::Disposal => Self::Disposal,
            BasicGlyphType::Equilibrium => Self::Equilibrium,
        }
    }
}

impl TryFrom<&GlyphType> for BasicGlyphType {
    type Error = ();
    fn try_from(value: &GlyphType) -> std::result::Result<Self, Self::Error> {
        match value {
            GlyphType::Calcification => Ok(Self::Calcification),
            GlyphType::Animismus => Ok(Self::Animismus),
            GlyphType::Projection => Ok(Self::Projection),
            GlyphType::Dispersion => Ok(Self::Dispersion),
            GlyphType::Purification => Ok(Self::Purification),
            GlyphType::Duplication => Ok(Self::Duplication),
            GlyphType::Unification => Ok(Self::Unification),
            GlyphType::Bonding => Ok(Self::Bonding),
            GlyphType::Unbonding => Ok(Self::Unbonding),
            GlyphType::TriplexBond => Ok(Self::TriplexBond),
            GlyphType::MultiBond => Ok(Self::MultiBond),
            GlyphType::Disposal => Ok(Self::Disposal),
            GlyphType::Equilibrium => Ok(Self::Equilibrium),
            _ => Err(()),
        }
    }
}

impl TryFrom<GlyphType> for BasicGlyphType {
    type Error = ();
    fn try_from(value: GlyphType) -> std::result::Result<Self, Self::Error> {
        <BasicGlyphType as TryFrom<&GlyphType>>::try_from(&value)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ArmType {
    PlainArm,
    DoubleArm,
    TripleArm,
    HexArm,
    Piston,
    VanBerlo,
}

impl ArmType {
    pub fn angles_between_arm(self) -> usize {
        match self {
            Self::PlainArm => 6,
            Self::DoubleArm => 3,
            Self::TripleArm => 2,
            Self::HexArm => 1,
            Self::Piston => 6,
            Self::VanBerlo => 1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Glyph {
    pub pos: Pos,
    pub rot: RawRot,
    pub glyph_type: GlyphType,
}
impl Glyph {
    pub fn new(glyph_type: GlyphType) -> Self {
        Self {
            glyph_type,
            pos: Pos::zeros(),
            rot: RawRot(0),
        }
    }

    pub fn move_by(&mut self, pos: Pos) {
        self.pos += pos;
        self.glyph_type.move_by(pos);
    }
    pub fn rot_by(&mut self, rot: RawRot) {
        self.rot += rot;
        self.glyph_type.rot_by(rot);
    }
    pub fn absolute_position(&self, rel_pos: Pos) -> Pos {
        self.pos + rotate(rel_pos, self.rot.normalize())
    }
    pub fn positions(&self) -> SmallVec<[Pos; 7]> {
        let pos = self.pos; //primary position
        let pos_bi = self.absolute_position(Pos::new(1, 0)); //position for all 2-sized selfs
        let pos_tri = self.absolute_position(Pos::new(0, 1)); //third (closely packed) position

        let pos_ani = self.absolute_position(Pos::new(1, -1));

        let pos_disp2 = self.absolute_position(Pos::new(1, -1));
        let pos_disp3 = self.absolute_position(Pos::new(0, -1));
        let pos_disp4 = self.absolute_position(Pos::new(-1, 0));

        let pos_multi2 = self.absolute_position(Pos::new(0, -1));
        let pos_multi3 = self.absolute_position(Pos::new(-1, 1));

        let pos_unif2 = self.absolute_position(Pos::new(-1, 1));
        let pos_unif3 = self.absolute_position(Pos::new(0, -1));
        let pos_unif4 = self.absolute_position(Pos::new(1, -1));

        match &self.glyph_type {
            GlyphType::Calcification => smallvec::smallvec![pos],
            GlyphType::Animismus => smallvec::smallvec![pos, pos_bi, pos_tri, pos_ani],
            GlyphType::Projection => smallvec::smallvec![pos, pos_bi],
            GlyphType::Dispersion => {
                smallvec::smallvec![pos, pos_bi, pos_disp2, pos_disp3, pos_disp4]
            }
            GlyphType::Unification => {
                smallvec::smallvec![pos, pos_tri, pos_unif2, pos_unif3, pos_unif4]
            }
            GlyphType::Purification => smallvec::smallvec![pos, pos_bi, pos_tri],
            GlyphType::Duplication => smallvec::smallvec![pos, pos_bi],
            GlyphType::Bonding => smallvec::smallvec![pos, pos_bi],
            GlyphType::MultiBond => smallvec::smallvec![pos, pos_bi, pos_multi2, pos_multi3],
            GlyphType::TriplexBond => smallvec::smallvec![pos, pos_bi, pos_tri],
            GlyphType::Unbonding => smallvec::smallvec![pos, pos_bi],
            GlyphType::Input { pattern, .. }
            | GlyphType::Output { pattern, .. }
            | GlyphType::OutputRepeating { pattern, .. } => pattern.iter().map(|a| a.pos).collect(),
            GlyphType::Track { locs: pos_list } => pos_list.iter().copied().collect(),
            GlyphType::Disposal => {
                smallvec::smallvec![pos, pos_bi, pos_tri, pos_ani, pos_disp3, pos_disp4, pos_unif2,]
            }
            GlyphType::Conduit {
                locs: _atom_teleport,
                ..
            } => smallvec::smallvec![
                //TODO
            ],
            GlyphType::Equilibrium => smallvec::smallvec![pos],
        }
    }
}

#[derive(Debug, Clone)]
pub struct Arm {
    pub pos: Pos,
    pub rot: RawRot,
    pub len: i32,
    pub arm_type: ArmType,
    pub grabbing: bool,
    pub atoms_grabbed: [AtomKey; 6],
}

impl Arm {
    pub fn new(pos: Pos, rot: RawRot, len: i32, arm_type: ArmType) -> Self {
        Arm {
            pos,
            rot,
            len,
            arm_type,
            grabbing: false,
            atoms_grabbed: [AtomKey::null(); 6],
        }
    }
    fn do_motion(&mut self, action: ArmMovement) {
        use ArmMovement::*;
        use Movement::*;
        match action {
            Move(Linear(p)) => self.pos += p,
            Move(Rotation(r, _)) => self.rot += r.raw_minabs(),
            LengthAdjust(a) => self.len += a,
            Move(HeldStill) | Pivot(_) => {}
        }
    }
}

#[derive(Debug)]
pub struct InitialWorld {
    pub glyphs: Vec<Glyph>,
    pub arms: Vec<Arm>,

    /// Same length as `arms`
    pub tapes: Vec<Tape<Instr>>,
}

impl InitialWorld {
    /// Returns `true` if overlap is detected
    fn mark_initial_area_and_detect_overlap(&self) -> (FxHashSet<Pos>, bool) {
        let mut area_touched = FxHashSet::default();
        let mut has_overlap = false;

        // first, do all of the positions that cannot overlap with each other.

        for glyph in self.glyphs.iter() {
            for p in glyph.positions() {
                if !area_touched.insert(p) {
                    has_overlap = true;
                }
            }
        }
        for arm in self.arms.iter() {
            if !area_touched.insert(arm.pos) {
                has_overlap = true;
            }
        }

        // now, do everything that should be marked as area, but are irrelevant
        // to overlap.

        for arm in self.arms.iter() {
            for n in 1..=arm.len {
                for r in Rot::step_by(arm.arm_type.angles_between_arm()) {
                    area_touched.insert(arm.pos + (arm.rot.normalize() + r).dist_to_pos(n));
                }
            }
        }

        (area_touched, has_overlap)
    }

    pub fn has_overlap(&self) -> bool {
        let (_, overlap) = self.mark_initial_area_and_detect_overlap();
        overlap
    }

    pub fn move_by(&mut self, pos: Pos) {
        for glyph in self.glyphs.iter_mut() {
            glyph.move_by(pos);
        }
        for arm in self.arms.iter_mut() {
            arm.pos += pos;
        }
    }

    pub fn rot_by(&mut self, rot: RawRot) {
        for glyph in self.glyphs.iter_mut() {
            glyph.rot_by(rot);
        }
        for arm in self.arms.iter_mut() {
            arm.pos = rotate(arm.pos, rot.normalize());
            arm.rot += rot;
        }
    }

    pub fn bounding_box(&self) -> Option<(Pos, Pos)> {
        let (area_touched, _) = self.mark_initial_area_and_detect_overlap();
        if area_touched.is_empty() {
            return None;
        }
        let min_x = area_touched.iter().map(|p| p.x).min().unwrap();
        let max_x = area_touched.iter().map(|p| p.x).max().unwrap();
        let min_y = area_touched.iter().map(|p| p.y).min().unwrap();
        let max_y = area_touched.iter().map(|p| p.y).max().unwrap();
        Some((Pos::new(min_x, min_y), Pos::new(max_x, max_y)))
    }
}

pub type TrackMap = FxHashMap<Pos, Pos>;

#[derive(Debug, Clone, Default)]
pub struct TrackMaps {
    pub plus: TrackMap,
    pub minus: TrackMap,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConduitInfo {
    pub vecids: (usize, usize),
    pub offset_pos: Pos,
    pub offset_rot: RawRot,
}
#[derive(Debug, Clone)]
pub struct World {
    pub timestep: u64,
    pub atoms: WorldAtoms,
    pub area_touched: FxHashSet<Pos>,
    pub glyphs: Vec<Glyph>,
    pub arms: Vec<Arm>,
    pub track_maps: TrackMaps,
    pub cost: i32,
    pub conduit_pairs: ConduitPairMap,
}
pub type ConduitPairMap = FxHashMap<i32, ConduitInfo>;
new_key_type! { pub struct AtomKey; }
#[derive(Debug, Clone)]
pub struct WorldAtoms {
    pub atom_map: SlotMap<AtomKey, Atom>,
    pub locs: FxHashMap<Pos, AtomKey>,
}
impl WorldAtoms {
    pub fn new() -> WorldAtoms {
        WorldAtoms {
            atom_map: SlotMap::with_key(),
            locs: Default::default(),
        }
    }
    pub fn get_atom_mut(&mut self, loc: Pos) -> Option<&mut Atom> {
        let &key = self.locs.get(&loc)?;
        Some(&mut self.atom_map[key])
    }
    pub fn get_type(&self, loc: Pos) -> Option<AtomType> {
        let &key = self.locs.get(&loc)?;
        let atom = self
            .atom_map
            .get(key)
            .expect("Inconsistent atoms (type check)");
        Some(atom.atom_type)
    }
    fn get_nonberlo_type(&self, loc: Pos) -> Option<AtomType> {
        let &key = self.locs.get(&loc)?;
        let atom = self
            .atom_map
            .get(key)
            .expect("Inconsistent atoms (nonberlo type check)");
        if atom.is_berlo {
            return None;
        };
        Some(self.atom_map.get(key)?.atom_type)
    }
    fn get_consumable_type(&self, motion: &WorldStepInfo, loc: Pos) -> Option<AtomType> {
        let &key = self.locs.get(&loc)?;
        if motion.atoms.contains_key(key) || motion.recent_bonds.contains(&key) {
            return None;
        }
        let atom = self
            .atom_map
            .get(key)
            .expect("Inconsistent atoms (consume check)");
        if atom.connections != [Bonds::NO_BOND; 6] {
            return None;
        }
        //Don't need to check is_berlo since the berlo arm will always be grabbing it (applying a motion)
        Some(atom.atom_type)
    }
    //Same as above except doesn't have the unbond check
    fn get_consumable_type_immediate(&self, motion: &WorldStepInfo, loc: Pos) -> Option<AtomType> {
        let &key = self.locs.get(&loc)?;
        if motion.atoms.contains_key(key) {
            return None;
        }
        let atom = self
            .atom_map
            .get(key)
            .expect("Inconsistent atoms (consume check)");
        if atom.connections != [Bonds::NO_BOND; 6] {
            return None;
        }
        //Don't need to check is_berlo since the berlo arm will always be grabbing it (applying a motion)
        Some(atom.atom_type)
    }
    fn check_empty(&self, motion: &WorldStepInfo, loc: Pos) -> bool {
        if self.locs.contains_key(&loc) {
            return false;
        }
        return !motion.spawning_atoms.iter().any(|x| x.pos == loc);
    }

    //Note: only use on non-grabbed atoms!
    pub fn destroy_atom_at(&mut self, loc: Pos) {
        let atom_key = self.locs.remove(&loc).expect("Destroying nonatom!");
        let atom = self
            .atom_map
            .remove(atom_key)
            .expect("Inconsistent atoms (destroy)!");
        if atom.pos != loc {
            panic!("Inconsistent atoms (destroy loc)!");
        }
    }
    pub fn take_atom(&mut self, key: AtomKey) -> Atom {
        let atom = self
            .atom_map
            .remove(key)
            .expect("Inconsistent atoms (take)!");
        let loc = atom.pos;
        let atom_key = self.locs.remove(&loc).expect("Destroying nonatom!");
        if atom_key != key {
            panic!("Inconsistent atoms (take key)!");
        }
        atom
    }

    pub fn create_atom(&mut self, atom: Atom) -> SimResult<AtomKey> {
        let key = self.atom_map.insert(atom);
        let atom_ref = &self.atom_map[key];
        let output = self.locs.insert(atom_ref.pos, key);
        if let Some(_old_atom) = output {
            return Err(SimError {
                error_str: "Atom creation on hex with existing atom!",
                location: pos_to_xy(atom_ref.pos),
            });
        }
        Ok(key)
    }

    pub fn iter_molecule(&self, atom_key: AtomKey) -> WorldAtomMoleculeIterator {
        let mut iter = WorldAtomMoleculeIterator {
            atoms: self,
            stack: Vec::new(),
            seen: FxHashSet::default(),
        };
        iter.pend_visit(atom_key);
        iter
    }

    pub fn iter_molecules(&self) -> WorldAtomMoleculesIterator {
        WorldAtomMoleculesIterator {
            atoms: self,
            atom_iter: self.atom_map.iter(),
            seen: FxHashSet::default(),
        }
    }
}

pub struct WorldAtomMoleculeIterator<'a> {
    atoms: &'a WorldAtoms,
    stack: Vec<AtomKey>,
    seen: FxHashSet<AtomKey>,
}

impl<'a> WorldAtomMoleculeIterator<'a> {
    fn pend_visit(&mut self, key: AtomKey) {
        if self.seen.insert(key) {
            self.stack.push(key);
        }
    }
}

impl<'a> Iterator for WorldAtomMoleculeIterator<'a> {
    type Item = AtomKey;

    fn next(&mut self) -> Option<Self::Item> {
        self.stack.pop().map(|next_key| {
            self.pend_visit(next_key);
            next_key
        })
    }
}

pub struct WorldAtomMoleculesIterator<'a> {
    atoms: &'a WorldAtoms,
    atom_iter: slotmap::basic::Iter<'a, AtomKey, Atom>,
    seen: FxHashSet<AtomKey>,
}

impl<'a> Iterator for WorldAtomMoleculesIterator<'a> {
    type Item = Vec<AtomKey>;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((atom_key, _atom)) = self.atom_iter.next() {
            if self.seen.contains(&atom_key) {
                continue;
            }
            let mut molecule_iterator = self.atoms.iter_molecule(atom_key);
            let molecule: Vec<_> = molecule_iterator.by_ref().collect();
            self.seen.extend(molecule_iterator.seen);
            return Some(molecule);
        }
        None
    }
}

/// Contains information about what changes during the current timestep,
/// including motion information.
pub struct WorldStepInfo {
    pub atoms: SecondaryMap<AtomKey, Movement>,
    pub arms: Vec<ArmMovement>,
    pub spawning_atoms: VecDeque<Atom>,
    pub recent_bonds: FxHashSet<AtomKey>,
    /// Contains one entry per arm
    pub drop_conduit_check: FxHashSet<AtomKey>,
    /// When glyph is a conduit, repeat for number of atoms sent
    pub active_glyphs: VecDeque<usize>,
}

impl WorldStepInfo {
    pub fn new() -> Self {
        WorldStepInfo {
            atoms: SecondaryMap::new(),
            arms: Vec::new(),
            spawning_atoms: VecDeque::new(),
            recent_bonds: FxHashSet::default(),
            drop_conduit_check: FxHashSet::default(),
            active_glyphs: VecDeque::new(),
        }
    }
    pub fn clear(&mut self) {
        self.atoms.clear();
        self.arms.clear();
        self.spawning_atoms.clear();
        self.recent_bonds.clear();
        self.drop_conduit_check.clear();
        self.active_glyphs.clear();
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SolutionStats {
    pub cycles: i32,
    pub cost: i32,
    pub area: i32,
    pub instructions: i32,
}

/// Convention: +x right, +y up
pub type XYPos = Point2<f32>;
pub type XYVec = Vector2<f32>;

#[derive(Debug, Copy, Clone)]
pub struct FloatAtom {
    pub pos: XYPos,
    pub rot: f32,
    pub atom_type: AtomType,
    pub connections: [Bonds; 6],
}
impl From<&Atom> for FloatAtom {
    fn from(a: &Atom) -> FloatAtom {
        FloatAtom {
            pos: pos_to_xy(a.pos),
            rot: 0.,
            atom_type: a.atom_type,
            connections: a.connections,
        }
    }
}

/// Float view of arm
pub struct FloatArm {
    pub pos: XYPos,
    pub rot: f32,
    pub len: f32,
    pub arm_type: ArmType,
    pub grabbing: bool,
}

/// Float view of the world, required to simulate collisions
pub struct FloatWorld {
    //Might add arms, maybe some other animation stuff too
    pub portion: f32,
    pub atoms_xy: Vec<FloatAtom>,
    pub arms_xy: Vec<FloatArm>,
}

fn pos_to_xy(input: Pos) -> XYPos {
    let a = input.x as f32;
    let b = input.y as f32;
    XYPos::new(a * 2. + b, b * f32::sqrt(3.))
}

/// Taking a `RawRot` instead of a `Rot` is deliberate here; this is the source
/// of the "overclocking" bugs in the original name.
fn rot_to_angle(r: RawRot) -> f32 {
    (r.0 as f32) * PI / 3.
}

fn xy_to_simple_pos(input: XYPos) -> Pos {
    let b = input.y / f32::sqrt(3.);
    let a = (input.x - b) / 2.;
    Pos::new(a.round() as i32, b.round() as i32)
}
//https://stackoverflow.com/questions/7705228/hexagonal-grids-how-do-you-find-which-hexagon-a-point-is-in/23370350#23370350
fn xy_to_pos(input: XYPos) -> Pos {
    //x=a*2.+b, y=b*f32::sqrt(3.)
    let b = input.y / f32::sqrt(3.);
    let a = (input.x - b) / 2.;
    // check closest to (p+1,q), (p,q+1), (p-1,q) or (p,q-1)
    let (p, q) = (a.round(), b.round());
    let (lambda, mu) = (a - p, b - q);
    // opposite signs, so we are guaranteed to be inside hexagon (p,q)
    let base_hex = Pos::new(p as i32, q as i32);
    if lambda * mu < 0.0 {
        return base_hex;
    }
    let distance = nalgebra::distance_squared(&pos_to_xy(base_hex), &input);
    // inside circle, so guaranteed inside hexagon (p,q)
    if distance < 1. {
        return base_hex;
    }

    // same sign, but which end of the parallelogram are we?
    let sign = lambda.signum();
    let candidate = if lambda.abs() > mu.abs() {
        (p + sign, q)
    } else {
        (p, q + sign)
    };
    let candidate_hex = Pos::new(candidate.0 as i32, candidate.1 as i32);
    let distance2 = nalgebra::distance_squared(&pos_to_xy(candidate_hex), &input);
    if distance < distance2 {
        base_hex
    } else {
        candidate_hex
    }
}
impl FloatWorld {
    pub fn new() -> Self {
        FloatWorld {
            portion: 0.,
            atoms_xy: Vec::new(),
            arms_xy: Vec::new(),
        }
    }

    /// Sets `self` to the float view of `world`, with `motion` lerped by `portion`.
    pub fn regenerate(&mut self, world: &World, motion: &WorldStepInfo, portion: f32) {
        self.arms_xy.clear();
        self.atoms_xy.clear();
        self.portion = portion;
        use ArmMovement::*;
        use Movement::*;
        fn apply_movement(base: Pos, movement: Movement, amount: f32) -> (XYPos, f32) {
            let xy = pos_to_xy(base);
            match movement {
                Linear(offset) => {
                    let offset_xy = pos_to_xy(offset) - XYPos::origin();
                    (xy + (offset_xy * amount), 0.)
                }
                Rotation(r, pivot) => {
                    let pivot_xy = pos_to_xy(pivot);
                    let offset = xy - pivot_xy;
                    let rot = rot_to_angle(r.raw_minabs()) * amount;
                    (pivot_xy + (nalgebra::Rotation2::new(rot) * offset), rot)
                }
                HeldStill => (xy, 0.),
            }
        }
        for (atom_key, atom) in &world.atoms.atom_map {
            let movement = motion.atoms.get(atom_key).unwrap_or(&HeldStill);
            let (xy, angle) = apply_movement(atom.pos, *movement, portion);
            let f_atom = FloatAtom {
                pos: xy,
                rot: angle,
                atom_type: atom.atom_type,
                connections: atom.connections,
            };
            self.atoms_xy.push(f_atom);
        }
        assert_eq!(motion.arms.len(), world.arms.len());
        for a_index in 0..world.arms.len() {
            let arm = &world.arms[a_index];
            let mut arm_len = arm.len as f32;
            let motion = motion.arms[a_index];
            let new_motion = match motion {
                Move(m) => m,
                LengthAdjust(l) => {
                    arm_len += (l as f32) * portion;
                    HeldStill
                }
                Pivot(_) => HeldStill,
            };
            let (xy, angle) = apply_movement(arm.pos, new_motion, portion);
            let new_arm = FloatArm {
                pos: xy,
                rot: angle + rot_to_angle(arm.rot),
                len: arm_len * 2.,
                arm_type: arm.arm_type,
                grabbing: arm.grabbing,
            };
            self.arms_xy.push(new_arm);
        }
    }

    /// Sets `self` to the float view of `world`, without motion information.
    pub fn generate_static(&mut self, world: &World) {
        self.arms_xy.clear();
        self.atoms_xy.clear();
        for (_atom_key, atom) in &world.atoms.atom_map {
            let xy = pos_to_xy(atom.pos);
            let f_atom = FloatAtom {
                pos: xy,
                rot: 0.,
                atom_type: atom.atom_type,
                connections: atom.connections,
            };
            self.atoms_xy.push(f_atom);
        }
        for a_index in 0..world.arms.len() {
            let arm = &world.arms[a_index];
            let arm_len = arm.len as f32;
            let new_arm = FloatArm {
                pos: pos_to_xy(arm.pos),
                rot: rot_to_angle(arm.rot),
                len: arm_len * 2.,
                arm_type: arm.arm_type,
                grabbing: arm.grabbing,
            };
            self.arms_xy.push(new_arm);
        }
    }
}

/// Running the world
impl World {
    /// Sets up a move for the specified atom and all atoms connected to it
    fn premove_atoms(
        &self,
        motion: &mut WorldStepInfo,
        atom_key: AtomKey,
        movement: Movement,
    ) -> SimResult<()> {
        let mut moving_atoms = Vec::new();

        let insert_motion = |motion: &mut WorldStepInfo, moving_atoms: &mut Vec<AtomKey>, key| {
            let maybe_move = motion.atoms.get(key).copied();
            if let Some(curr_move) = maybe_move {
                if curr_move != movement {
                    //println!("was {:?},applying {:?}", curr_move, movement);
                    let error_str = &"Atom moved in multiple directions!";
                    return Err(sim_error_pos(error_str, self.atoms.atom_map[key].pos));
                }
            } else {
                motion.atoms.insert(key, movement);
                moving_atoms.push(key);
            }
            Ok(maybe_move)
        };

        insert_motion(motion, &mut moving_atoms, atom_key)?;

        while let Some(this_key) = moving_atoms.pop() {
            let atom = &self.atoms.atom_map[this_key];
            for dir in Rot::ALL {
                if atom.connections[dir.to_usize()].intersects(Bonds::DYNAMIC_BOND) {
                    let newpos = atom.pos + dir.to_pos();
                    let newkey = *self
                        .atoms
                        .locs
                        .get(&newpos)
                        .expect("Inconsistent atoms (movement prep)");

                    insert_motion(motion, &mut moving_atoms, newkey)?;
                }
            }
        }
        Ok(())
    }

    /// Finalize atom movement
    pub fn apply_motion(&mut self, motion: &WorldStepInfo) -> SimResult<()> {
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
            if current.is_some() {
                let error_str = &"Multiple atoms in same hex!";
                return Err(sim_error_pos(error_str, atom.pos));
            }
        }
        Ok(())
    }

    /// Performs an instruction for a single arm.
    pub fn do_instruction(
        &mut self,
        motion: &mut WorldStepInfo,
        arm_id: usize,
        instruction: BasicInstr,
    ) -> SimResult<()> {
        assert_eq!(
            motion.arms.len(),
            arm_id,
            "Arm motion array not same as arm ID"
        );
        let arm = &mut self.arms[arm_id];
        use ArmMovement::*;
        use ArmType::*;
        use BasicInstr::*;
        use Movement::*;
        let arm_type = arm.arm_type;
        const STILL: ArmMovement = Move(HeldStill);
        let action = match instruction {
            Extend => {
                if arm_type == Piston {
                    if arm.len < 3 {
                        LengthAdjust(1)
                    } else {
                        STILL
                    }
                } else {
                    return Err(sim_error_pos("Extend on non-piston arm", arm.pos));
                }
            }
            Retract => {
                if arm_type == Piston {
                    if arm.len > 1 {
                        LengthAdjust(-1)
                    } else {
                        STILL
                    }
                } else {
                    return Err(sim_error_pos("Retract on non-piston arm", arm.pos));
                }
            }
            RotateCounterClockwise => Move(Rotation(Rot::from_i8(1), arm.pos)),
            RotateClockwise => Move(Rotation(Rot::from_i8(-1), arm.pos)),
            PivotCounterClockwise => Pivot(Rot::from_i8(1)),
            PivotClockwise => Pivot(Rot::from_i8(-1)),
            Drop => {
                if arm_type != VanBerlo {
                    for r in Rot::ALL {
                        let grabbed_atom = arm.atoms_grabbed[r.to_usize()];
                        if !grabbed_atom.is_null() {
                            motion.drop_conduit_check.insert(grabbed_atom);
                        }
                    }
                    arm.atoms_grabbed = [AtomKey::null(); 6];
                    arm.grabbing = false;
                }
                STILL
            }
            Grab => {
                if !arm.grabbing && arm_type != VanBerlo {
                    arm.grabbing = true;
                    for r in Rot::step_by(arm.arm_type.angles_between_arm()) {
                        let grab_pos = arm.pos + ((arm.rot.normalize() + r).dist_to_pos(arm.len));
                        let null_key = AtomKey::null();
                        let current = self
                            .atoms
                            .locs
                            .get(&grab_pos)
                            .filter(|&key| !self.atoms.atom_map[*key].is_berlo)
                            .unwrap_or(&null_key);
                        arm.atoms_grabbed[r.to_usize()] = *current;
                    }
                }
                STILL
            }
            Forward => {
                if let Some(&x) = self.track_maps.plus.get(&arm.pos) {
                    if x == Pos::zeros() {
                        STILL
                    } else {
                        Move(Linear(x))
                    }
                } else {
                    if !self.track_maps.minus.contains_key(&arm.pos) {
                        return Err(sim_error_pos(
                            "Forward on arm that's not on a track",
                            arm.pos,
                        ));
                    }
                    STILL
                }
            }
            Back => {
                if let Some(&x) = self.track_maps.minus.get(&arm.pos) {
                    if x == Pos::zeros() {
                        STILL
                    } else {
                        Move(Linear(x))
                    }
                } else {
                    if !self.track_maps.plus.contains_key(&arm.pos) {
                        return Err(sim_error_pos("Back on arm that's not on a track", arm.pos));
                    }
                    STILL
                }
            }
            Empty => STILL,
        };
        let rotation_store = arm.rot;
        for atom in arm.atoms_grabbed {
            if !atom.is_null() {
                let atom_movement = match action {
                    ArmMovement::Pivot(r) => Movement::Rotation(r, self.atoms.atom_map[atom].pos),
                    ArmMovement::Move(m) => m,
                    ArmMovement::LengthAdjust(a) => Linear(rotation_store.normalize().to_pos() * a),
                };
                match motion.atoms.insert(atom, atom_movement) {
                    Some(x) if x != atom_movement => {
                        //println!("was {:?},applying {:?}", atom_movement, x);
                        let error_str = "Atom moved in multiple directions!";
                        return Err(sim_error_pos(error_str, self.atoms.atom_map[atom].pos));
                    }
                    _ => (),
                }
            }
        }
        motion.arms.push(action);
        Ok(())
    }

    fn process_inputs(&mut self, motion: &mut WorldStepInfo) -> SimResult<()> {
        use GlyphType::*;
        for glyph in self.glyphs.iter_mut() {
            let atoms = &mut self.atoms;
            if let Input { pattern, .. } = &mut glyph.glyph_type {
                if pattern.iter().all(|a| atoms.check_empty(motion, a.pos)) {
                    for a in pattern {
                        atoms.create_atom(a.clone())?;
                    }
                }
            }
        }
        Ok(())
    }

    fn process_outputs(&mut self, motion: &mut WorldStepInfo) {
        use GlyphType::*;
        for glyph in self.glyphs.iter_mut() {
            let atoms = &mut self.atoms;
            match &mut glyph.glyph_type {
                Output {
                    pattern,
                    count: output_count,
                    ..
                } => {
                    let full_match = pattern.iter().all(|a| -> bool {
                        let try_key = atoms.locs.get(&a.pos);
                        if let Some(&atom_key) = try_key {
                            &atoms.atom_map[atom_key] == a && !motion.atoms.contains_key(atom_key)
                        } else {
                            false
                        }
                    });
                    if full_match {
                        for a in pattern {
                            atoms.destroy_atom_at(a.pos);
                        }
                        if *output_count > 0 {
                            *output_count -= 1;
                        }
                    }
                }
                OutputRepeating {
                    pattern,
                    count: output_count,
                    ..
                } => {
                    let full_match = pattern.iter().all(|a| -> bool {
                        let try_key = atoms.locs.get(&a.pos);
                        if let Some(&atom_key) = try_key {
                            &atoms.atom_map[atom_key] == a
                        } else {
                            false
                        }
                    });
                    if full_match {
                        //TODO: Accurately detect partial infinites
                        if *output_count > 0 {
                            *output_count = 0;
                        }
                    }
                }
                _ => {}
            }
        }
    }

    fn process_glyphs(&mut self, motion: &mut WorldStepInfo, first_half: bool) -> SimResult<()> {
        fn try_bond(atoms: &mut WorldAtoms, loc1: &Pos, loc2: &Pos) {
            let rot = Rot::from_unit_pos(loc2 - loc1).unwrap();
            if let (Some(&key1), Some(&key2)) = (atoms.locs.get(loc1), atoms.locs.get(loc2)) {
                let [atom1, atom2] = atoms
                    .atom_map
                    .get_disjoint_mut([key1, key2])
                    .expect("Inconsistent atoms!");
                if atom1.is_berlo || atom2.is_berlo {
                    return;
                }
                let bond1 = atom1.connections[rot.to_usize()];
                assert_eq!(
                    atom2.connections[rot.opp().to_usize()],
                    bond1,
                    "Inconsistent bonds"
                );
                if bond1 == Bonds::NO_BOND {
                    atom1.connections[rot.to_usize()] = Bonds::NORMAL;
                    atom2.connections[rot.opp().to_usize()] = Bonds::NORMAL;
                }
            }
        }
        fn try_triplex_bond(atoms: &mut WorldAtoms, loc1: &Pos, loc2: &Pos, bond_type: Bonds) {
            let rot = Rot::from_unit_pos(loc2 - loc1).unwrap();
            if let (Some(&key1), Some(&key2)) = (atoms.locs.get(loc1), atoms.locs.get(loc2)) {
                let [atom1, atom2] = atoms
                    .atom_map
                    .get_disjoint_mut([key1, key2])
                    .expect("Inconsistent atoms!");
                if atom1.is_berlo || atom2.is_berlo {
                    return;
                }
                if atom1.atom_type != AtomType::Fire || atom2.atom_type != AtomType::Fire {
                    return;
                }
                let bond1 = atom1.connections[rot.to_usize()];
                assert_eq!(
                    atom2.connections[rot.opp().to_usize()],
                    bond1,
                    "Inconsistent bonds"
                );
                if (bond1 & !Bonds::TRIPLEX).is_empty() {
                    atom1.connections[rot.to_usize()] |= bond_type;
                    atom2.connections[rot.opp().to_usize()] |= bond_type;
                }
            }
        }
        use AtomType::*;
        use GlyphType::*;
        for (id, glyph) in self.glyphs.iter_mut().enumerate() {
            let atoms = &mut self.atoms;
            let rot = glyph.rot.normalize();

            let pos = glyph.pos; //primary position
            let pos_bi = glyph.absolute_position(Pos::new(1, 0)); //position for all 2-sized glyphs
            let pos_tri = glyph.absolute_position(Pos::new(0, 1)); //third (closely packed) position

            let pos_ani = glyph.absolute_position(Pos::new(1, -1));

            let pos_disp2 = glyph.absolute_position(Pos::new(1, -1));
            let pos_disp3 = glyph.absolute_position(Pos::new(0, -1));
            let pos_disp4 = glyph.absolute_position(Pos::new(-1, 0));

            let pos_multi2 = glyph.absolute_position(Pos::new(0, -1));
            let pos_multi3 = glyph.absolute_position(Pos::new(-1, 1));

            let pos_unif2 = glyph.absolute_position(Pos::new(-1, 1));
            let pos_unif3 = glyph.absolute_position(Pos::new(0, -1));
            let pos_unif4 = glyph.absolute_position(Pos::new(1, -1));

            fn pop_check_make(
                atoms: &mut WorldAtoms,
                motion: &mut WorldStepInfo,
                atom: Atom,
            ) -> SimResult<()> {
                let a = motion.spawning_atoms.pop_front().unwrap();
                assert!(a == atom);
                atoms.create_atom(a)?;
                Ok(())
            }
            fn pop_make(atoms: &mut WorldAtoms, motion: &mut WorldStepInfo) -> SimResult<()> {
                atoms.create_atom(motion.spawning_atoms.pop_front().unwrap())?;
                Ok(())
            }
            match &mut glyph.glyph_type {
                Calcification => {
                    if let Some(atom) = atoms.get_atom_mut(pos) {
                        if atom.atom_type.is_element() && !atom.is_berlo {
                            atom.atom_type = AtomType::Salt;
                        }
                    }
                }
                Animismus => {
                    if first_half {
                        let a1 = atoms.get_consumable_type(motion, pos);
                        let a2 = atoms.get_consumable_type(motion, pos_bi);
                        let o1 = atoms.check_empty(motion, pos_tri);
                        let o2 = atoms.check_empty(motion, pos_ani);
                        if (Some(Salt), Some(Salt), true, true) == (a1, a2, o1, o2) {
                            atoms.destroy_atom_at(pos);
                            atoms.destroy_atom_at(pos_bi);
                            motion.spawning_atoms.push_back(Atom::new(pos_tri, Vitae));
                            motion.spawning_atoms.push_back(Atom::new(pos_ani, Mors));
                            motion.active_glyphs.push_back(id);
                        }
                    } else if motion.active_glyphs.front() == Some(&id) {
                        pop_check_make(&mut self.atoms, motion, Atom::new(pos_tri, Vitae))?;
                        pop_check_make(&mut self.atoms, motion, Atom::new(pos_ani, Mors))?;
                        motion.active_glyphs.pop_front();
                    }
                }
                Projection => {
                    let qs = atoms.get_consumable_type(motion, pos);
                    let metal = atoms.get_type(pos_bi);
                    if let (Some(Quicksilver), Some(metal)) = (qs, metal) {
                        if let Some(newtype) = metal.promotable_metal() {
                            atoms.destroy_atom_at(pos);
                            atoms.get_atom_mut(pos_bi).unwrap().atom_type = newtype;
                        }
                    }
                }
                Dispersion => {
                    if first_half {
                        let q = atoms.get_consumable_type(motion, pos);
                        let o1 = atoms.check_empty(motion, pos_bi);
                        let o2 = atoms.check_empty(motion, pos_disp2);
                        let o3 = atoms.check_empty(motion, pos_disp3);
                        let o4 = atoms.check_empty(motion, pos_disp4);
                        if (Some(Quintessence), true, true, true, true) == (q, o1, o2, o3, o4) {
                            atoms.destroy_atom_at(pos);
                            motion.spawning_atoms.push_back(Atom::new(pos_bi, Earth));
                            motion.spawning_atoms.push_back(Atom::new(pos_disp2, Water));
                            motion.spawning_atoms.push_back(Atom::new(pos_disp3, Fire));
                            motion.spawning_atoms.push_back(Atom::new(pos_disp4, Air));
                            motion.active_glyphs.push_back(id);
                        }
                    } else if motion.active_glyphs.front() == Some(&id) {
                        pop_check_make(&mut self.atoms, motion, Atom::new(pos_bi, Earth))?;
                        pop_check_make(&mut self.atoms, motion, Atom::new(pos_disp2, Water))?;
                        pop_check_make(&mut self.atoms, motion, Atom::new(pos_disp3, Fire))?;
                        pop_check_make(&mut self.atoms, motion, Atom::new(pos_disp4, Air))?;
                        motion.active_glyphs.pop_front();
                    }
                }
                Unification => {
                    if first_half {
                        let output = atoms.check_empty(motion, pos);
                        let a1 = atoms.get_consumable_type(motion, pos_tri);
                        let a2 = atoms.get_consumable_type(motion, pos_unif2);
                        let a3 = atoms.get_consumable_type(motion, pos_unif3);
                        let a4 = atoms.get_consumable_type(motion, pos_unif4);
                        if let (true, Some(a), Some(b), Some(c), Some(d)) = (output, a1, a2, a3, a4)
                        {
                            let set = [a, b, c, d];
                            if set.contains(&Earth)
                                && set.contains(&Water)
                                && set.contains(&Fire)
                                && set.contains(&Air)
                            {
                                atoms.destroy_atom_at(pos_tri);
                                atoms.destroy_atom_at(pos_unif2);
                                atoms.destroy_atom_at(pos_unif3);
                                atoms.destroy_atom_at(pos_unif4);
                                motion
                                    .spawning_atoms
                                    .push_back(Atom::new(pos, Quintessence));
                                motion.active_glyphs.push_back(id);
                            }
                        }
                    } else if motion.active_glyphs.front() == Some(&id) {
                        pop_check_make(&mut self.atoms, motion, Atom::new(pos, Quintessence))?;
                        motion.active_glyphs.pop_front();
                    }
                }
                Purification => {
                    if first_half {
                        let a1 = atoms.get_consumable_type(motion, pos);
                        let a2 = atoms.get_consumable_type(motion, pos_bi);
                        let o = atoms.check_empty(motion, pos_tri);
                        match (a1, a2, o) {
                            (Some(a1), Some(a2), true) if a1 == a2 => {
                                let next = a1.promotable_metal();
                                if let Some(newtype) = next {
                                    atoms.destroy_atom_at(pos);
                                    atoms.destroy_atom_at(pos_bi);
                                    motion.spawning_atoms.push_back(Atom::new(pos_tri, newtype));
                                    motion.active_glyphs.push_back(id);
                                }
                            }
                            _ => (),
                        }
                    } else if motion.active_glyphs.front() == Some(&id) {
                        pop_make(&mut self.atoms, motion)?;
                        motion.active_glyphs.pop_front();
                    }
                }
                Duplication => {
                    let source = atoms.get_type(pos);
                    let salt = atoms.get_nonberlo_type(pos_bi);
                    if let (Some(elem), Some(Salt)) = (source, salt) {
                        if elem.is_element() {
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
                    try_triplex_bond(atoms, &pos, &pos_tri, Bonds::TRIPLEX_Y);
                    try_triplex_bond(atoms, &pos_bi, &pos_tri, Bonds::TRIPLEX_R);
                }
                Unbonding => {
                    if let (Some(&key1), Some(&key2)) =
                        (atoms.locs.get(&pos), atoms.locs.get(&pos_bi))
                    {
                        let bond1 = atoms.atom_map[key1].connections[rot.to_usize()];
                        assert_eq!(
                            atoms.atom_map[key2].connections[rot.opp().to_usize()],
                            bond1,
                            "Inconsistent bonds"
                        );
                        if bond1.intersects(Bonds::DYNAMIC_BOND) {
                            atoms.atom_map[key1].connections[rot.to_usize()] = Bonds::NO_BOND;
                            atoms.atom_map[key2].connections[rot.opp().to_usize()] = Bonds::NO_BOND;
                            motion.recent_bonds.insert(key1);
                            motion.recent_bonds.insert(key2);
                        }
                    }
                }
                Disposal => {
                    if atoms.get_consumable_type_immediate(motion, pos).is_some() {
                        atoms.destroy_atom_at(pos);
                    }
                }
                Conduit {
                    locs: atom_teleport,
                    id: conduit_id,
                } => {
                    if first_half {
                        let conduit_info = self
                            .conduit_pairs
                            .get_mut(conduit_id)
                            .expect("conduit info not found!");
                        World::conduit_process(
                            atoms,
                            id,
                            atom_teleport,
                            glyph.pos,
                            conduit_info,
                            motion,
                        );
                    } else if motion.active_glyphs.front() == Some(&id) {
                        while motion.active_glyphs.front() == Some(&id) {
                            motion.active_glyphs.pop_front();
                            pop_make(&mut self.atoms, motion)?;
                        }
                    }
                }
                Equilibrium
                | Track { .. }
                | Input { .. }
                | Output { .. }
                | OutputRepeating { .. } => {}
            }
        }
        if !first_half {
            assert_eq!(
                (motion.active_glyphs.len(), motion.spawning_atoms.len()),
                (0, 0)
            );
        }
        Ok(())
    }

    fn conduit_process(
        atoms: &mut WorldAtoms,
        glyph_id: usize,
        positions: &Vec<Pos>,
        origin: Pos,
        conduit_info: &mut ConduitInfo,
        motion: &mut WorldStepInfo,
    ) {
        let mut viewed_atoms = FxHashSet::<AtomKey>::default();
        let mut check_atoms = VecDeque::<AtomKey>::new();

        let mut offset_pos = conduit_info.offset_pos;
        let mut offset_rot = conduit_info.offset_rot.normalize();
        if conduit_info.vecids.0 == glyph_id {
            offset_pos *= -1;
            offset_rot = -offset_rot;
        } else {
            assert!(conduit_info.vecids.1 == glyph_id, "Conduit invalid vec id");
        }
        for pos in positions {
            //For each position on the conduit, if it has an atom on it:
            if let Some(key) = atoms.locs.get(pos) {
                //Closure to check if all atoms in the molecule are on the conduit
                let mut atoms_send_check = |first_key: AtomKey| -> bool {
                    check_atoms.push_back(first_key);
                    while let Some(this_key) = check_atoms.pop_front() {
                        if viewed_atoms.contains(&this_key) {
                            continue;
                        }
                        if motion.atoms.contains_key(this_key) {
                            return false;
                        }
                        viewed_atoms.insert(this_key);

                        let atom = &atoms.atom_map[this_key];
                        for dir in Rot::ALL {
                            if atom.connections[dir.to_usize()].intersects(Bonds::DYNAMIC_BOND) {
                                let newpos = atom.pos + dir.to_pos();
                                if !positions.contains(&newpos) {
                                    return false;
                                }
                                let newkey = *atoms
                                    .locs
                                    .get(&newpos)
                                    .expect("Inconsistent atoms (conduits)");
                                check_atoms.push_back(newkey);
                            }
                        }
                    }
                    true
                };
                //If the atom was just dropped and all connected are on the conduit, move the atoms to the new position via spawning
                if motion.drop_conduit_check.contains(key) && atoms_send_check(*key) {
                    for a in &viewed_atoms {
                        let mut atom = atoms.take_atom(*a);
                        atom.pos = rotate_around(atom.pos, offset_rot, origin) + offset_pos;
                        atom.rotate_connections(offset_rot);
                        motion.spawning_atoms.push_back(atom);
                        motion.active_glyphs.push_back(glyph_id);
                    }
                }
                viewed_atoms.clear();
                check_atoms.clear();
            }
        }
    }

    pub fn mark_area_and_collide<'a>(
        &mut self,
        float_world: &FloatWorld,
        spawning_atoms: impl Iterator<Item = &'a Atom>,
    ) -> SimResult<()> {
        //atom radius = 29/41 or 1/sqrt(2)
        //spawning atom = 15/41
        //arm base = 20/41
        //grabber, cabinet = 24/41, 20/41
        const ATOM_RADIUS: f32 = 29. / 41.;
        const ARM_RADIUS: f32 = 20. / 41.;
        const SPAWNING_ATOM_RADIUS: f32 = 15. / 41.;
        const ATOM_ATOM_RADIUS_SQUARED: f32 = (ATOM_RADIUS * 2.) * (ATOM_RADIUS * 2.);
        const ATOM_ARM_RADIUS_SQUARED: f32 =
            (ATOM_RADIUS + ARM_RADIUS) * (ATOM_RADIUS + ARM_RADIUS);
        const ATOM_SPAWN_RADIUS_SQUARED: f32 =
            (ATOM_RADIUS + SPAWNING_ATOM_RADIUS) * (ATOM_RADIUS + SPAWNING_ATOM_RADIUS);
        fn make_candidates(primary: Pos) -> [Pos; 7] {
            [
                Pos::new(primary.x, primary.y),
                Pos::new(primary.x + 1, primary.y),
                Pos::new(primary.x - 1, primary.y),
                Pos::new(primary.x, primary.y + 1),
                Pos::new(primary.x, primary.y - 1),
                Pos::new(primary.x - 1, primary.y + 1),
                Pos::new(primary.x + 1, primary.y - 1),
            ]
        }
        fn mark_point(area_set: &mut FxHashSet<Pos>, point: XYPos) {
            let primary = xy_to_simple_pos(point);
            let candidates = make_candidates(primary);
            for hex in candidates {
                let distance = nalgebra::distance_squared(&pos_to_xy(hex), &point);
                if distance < ATOM_ATOM_RADIUS_SQUARED {
                    area_set.insert(hex);
                }
            }
        }
        fn collide(
            atom_collisions: &mut FxHashMap<Pos, SmallVec<[XYPos; 2]>>,
            point: XYPos,
            check_dist: f32,
            error_str: &'static str,
        ) -> SimResult<()> {
            let primary = xy_to_simple_pos(point);
            let candidates = make_candidates(primary);
            for check in candidates {
                let possible_vec = atom_collisions.get(&check);
                if let Some(atoms) = possible_vec {
                    for atom_point in atoms {
                        if nalgebra::distance_squared(atom_point, &point) < check_dist {
                            let error_point = nalgebra::center(atom_point, &point);
                            return Err(SimError {
                                error_str,
                                location: error_point,
                            });
                        }
                    }
                }
            }
            Ok(())
        }
        let mut atom_collisions =
            FxHashMap::with_capacity_and_hasher(float_world.atoms_xy.len(), Default::default());
        for atom in &float_world.atoms_xy {
            let primary = xy_to_simple_pos(atom.pos);
            mark_point(&mut self.area_touched, atom.pos);
            collide(
                &mut atom_collisions,
                atom.pos,
                ATOM_ATOM_RADIUS_SQUARED,
                "atom/atom collision!",
            )?;
            atom_collisions
                .entry(primary)
                .or_insert_with(SmallVec::new)
                .push(atom.pos);
        }
        for arm in &float_world.arms_xy {
            mark_point(&mut self.area_touched, arm.pos);
            collide(
                &mut atom_collisions,
                arm.pos,
                ATOM_ARM_RADIUS_SQUARED,
                "atom/arm collision!",
            )?;
            for r in Rot::step_by(arm.arm_type.angles_between_arm()) {
                let angle = arm.rot + rot_to_angle(r.raw_nonneg());
                let offset = nalgebra::Rotation2::new(angle) * XYVec::new(arm.len, 0.);
                mark_point(&mut self.area_touched, arm.pos + offset);
                //arm lengths are doubled in floatworld
                if arm.len > 3.0 {
                    let offset = nalgebra::Rotation2::new(angle) * XYVec::new(2., 0.);
                    mark_point(&mut self.area_touched, arm.pos + offset);
                }
                if arm.len > 5.0 {
                    let offset = nalgebra::Rotation2::new(angle) * XYVec::new(4., 0.);
                    mark_point(&mut self.area_touched, arm.pos + offset);
                }
            }
        }
        for spawning_atom in spawning_atoms {
            collide(
                &mut atom_collisions,
                pos_to_xy(spawning_atom.pos),
                ATOM_SPAWN_RADIUS_SQUARED,
                "atom/spawning atom collision!",
            )?;
        }
        Ok(())
    }

    /// Runs one full step of the simulation. `float_world` is only used if `mark_area` is true.
    pub fn run_step(
        &mut self,
        mark_area: bool,
        motion: &mut WorldStepInfo,
        float_world: &mut FloatWorld,
        instructions: &[BasicInstr],
    ) -> SimResult<()> {
        self.prepare_step(motion, instructions)?;
        if mark_area {
            let substep_count = self.substep_count(motion);
            for substep in 0..substep_count {
                let portion = substep as f32 / substep_count as f32;
                float_world.regenerate(self, motion, portion);
                self.mark_area_and_collide(float_world, motion.spawning_atoms.iter())?;
            }
        }
        self.finalize_step(motion)?;
        Ok(())
    }

    /// Start a simulation step. This overwrites `motion` with information about
    /// the step.
    pub fn prepare_step(
        &mut self,
        motion: &mut WorldStepInfo,
        instructions: &[BasicInstr],
    ) -> SimResult<()> {
        motion.clear();
        assert_eq!(instructions.len(), self.arms.len());
        for (i, &instruction) in instructions.iter().enumerate() {
            self.do_instruction(motion, i, instruction)?;
        }
        self.process_inputs(motion)?;
        self.process_glyphs(motion, true)?;
        self.process_outputs(motion);

        for i in 0..self.arms.len() {
            let position = self.arms[i].pos;
            for j in i + 1..self.arms.len() {
                if position == self.arms[j].pos {
                    let location = pos_to_xy(position);
                    return Err(SimError {
                        location,
                        error_str: "Arm-Arm collision!",
                    });
                }
            }
        }
        let tmp = motion.atoms.clone();
        motion.atoms.clear();
        for (atom, movement) in tmp {
            self.premove_atoms(motion, atom, movement)?;
        }
        motion.recent_bonds.clear();
        Ok(())
    }

    /// Returns the substep count to be used for the current step, i.e. how many
    /// intermediate timesteps should be checked.
    pub fn substep_count(&self, motion: &WorldStepInfo) -> usize {
        let mut max_radius: f64 = 1.;
        for (atom_key, movement) in &motion.atoms {
            if let Movement::Rotation(_rot, center) = movement {
                let atom_pos = self.atoms.atom_map[atom_key].pos;
                let distance = nalgebra::distance(&pos_to_xy(atom_pos), &pos_to_xy(*center)) as f64;
                if max_radius < distance {
                    max_radius = distance;
                }
            }
        }
        max_radius *= 2.;
        usize::pow(2, max_radius.log2().round() as u32).max(8)
        //usize::max(10, (max_radius*2.0) as usize)
    }

    /// Finish a simulation step.
    pub fn finalize_step(&mut self, motion: &mut WorldStepInfo) -> SimResult<()> {
        self.apply_motion(motion)?;
        self.process_inputs(motion)?;
        self.process_glyphs(motion, false)?;
        motion.clear();
        for arm in &self.arms {
            for atomkey in arm.atoms_grabbed {
                motion.atoms.insert(atomkey, Movement::HeldStill);
                //does nothing if atomkey is null
            }
        }
        self.process_outputs(motion);
        motion.clear();
        self.timestep += 1;
        Ok(())
    }

    /// Returns `true` when all outputs been produced fully.
    pub fn is_complete(&self) -> bool {
        let mut all_outputs_full = true;
        for g in &self.glyphs {
            if let GlyphType::Output { count: i, .. }
            | GlyphType::OutputRepeating { count: i, .. } = g.glyph_type
            {
                all_outputs_full &= i == 0;
            }
        }
        all_outputs_full
    }
}

/// Setup stuff
impl World {
    fn add_track(track_maps: &mut TrackMaps, track_pos: &[Pos]) -> Result<()> {
        ensure!(!track_pos.is_empty(), "Track of length 0!");
        //first do all except the possible loopback
        for track_pair in track_pos.windows(2) {
            let (t_prev, t_now) = (track_pair[0], track_pair[1]);
            let offset = t_now - t_prev;
            track_maps.minus.entry(t_now).or_insert(-offset);
            track_maps.plus.entry(t_prev).or_insert(offset);
        }
        let first = *track_pos.first().unwrap();
        let last = *track_pos.last().unwrap();
        let offset = first - last;
        //check for looping on first/last
        if track_pos.len() > 2 && Rot::from_unit_pos(offset).is_some() {
            track_maps.minus.entry(first).or_insert(-offset);
            track_maps.plus.entry(last).or_insert(offset);
        } else {
            //If not looping, block future track overlaps here
            track_maps.minus.entry(first).or_insert(Pos::new(0, 0));
            track_maps.plus.entry(last).or_insert(Pos::new(0, 0));
        }
        Ok(())
    }

    /// Remove {0,0} track offsets that were used to block track overlaps
    fn clean_track(track_maps: &mut TrackMaps) {
        let pos_nil = Pos::new(0, 0);
        track_maps.plus.retain(|_, v| -> bool { v != &pos_nil });
        track_maps.minus.retain(|_, v| -> bool { v != &pos_nil });
    }

    /// The main initialization function. Creates the initial state of the world
    /// given an `InitialWorld`.
    pub fn setup_sim(init: &InitialWorld) -> Result<Self> {
        let (area_touched, _has_overlap) = init.mark_initial_area_and_detect_overlap();
        let mut world = World {
            timestep: 0,
            atoms: WorldAtoms::new(),
            area_touched,
            glyphs: init.glyphs.clone(),
            arms: init.arms.clone(),
            track_maps: Default::default(),
            cost: 0,
            conduit_pairs: Default::default(),
        };
        for g in &mut world.glyphs {
            use GlyphType::*;
            world.cost += match &g.glyph_type {
                Calcification | Bonding | Unbonding => 10,
                Animismus | Projection | Dispersion | Purification => 20,
                Duplication | Unification | TriplexBond => 20,
                MultiBond => 30,
                Disposal
                | Equilibrium
                | Output { .. }
                | OutputRepeating { .. }
                | Input { .. }
                | Conduit { .. } => 0,
                Track { locs: v } => (v.len() as i32) * 5,
            };
            if let Track { locs: track_data } = &g.glyph_type {
                World::add_track(&mut world.track_maps, track_data)?;
            }
        }
        World::clean_track(&mut world.track_maps);
        for a in &mut world.arms {
            use ArmType::*;
            world.cost += match a.arm_type {
                PlainArm => 20,
                DoubleArm | TripleArm | HexArm | VanBerlo => 30,
                Piston => 40,
            };
            if a.arm_type == VanBerlo {
                use AtomType::*;
                a.grabbing = true;
                const ATOM_SETUP: [AtomType; 6] = [Salt, Water, Air, Salt, Fire, Earth];
                for (r, &atom_type) in Rot::ALL.into_iter().zip(ATOM_SETUP.iter()) {
                    let pos = a.pos + (r + a.rot.normalize()).to_pos();
                    let key = world.atoms.create_atom(Atom {
                        pos,
                        atom_type,
                        connections: [Bonds::NO_BOND; 6],
                        is_berlo: true,
                    })?;
                    a.atoms_grabbed[r.to_usize()] = key;
                }
            }
        }

        let mut unfinished_conduit_count = 0;
        for (id, glyph) in world.glyphs.iter_mut().enumerate() {
            if let GlyphType::Input { pattern, .. } = &mut glyph.glyph_type {
                if pattern
                    .iter()
                    .all(|a| !world.atoms.locs.contains_key(&a.pos))
                {
                    for a in pattern {
                        world.atoms.create_atom(a.clone())?;
                    }
                }
            }
            if let GlyphType::Conduit { id: conduit_id, .. } = &glyph.glyph_type {
                if let Some(pair_data) = world.conduit_pairs.get_mut(conduit_id) {
                    ensure!(
                        pair_data.vecids.0 == pair_data.vecids.1,
                        "Conduit {conduit_id} already modified!"
                    );
                    pair_data.vecids.1 = id;
                    unfinished_conduit_count -= 1;
                    pair_data.offset_pos -= glyph.pos;
                    pair_data.offset_rot = (pair_data.offset_rot - glyph.rot).normalize().raw_nonneg();
                } else {
                    let conduit_info = ConduitInfo {
                        vecids: (id, id),
                        offset_pos: glyph.pos,
                        offset_rot: glyph.rot,
                    };
                    world.conduit_pairs.insert(*conduit_id, conduit_info);
                    unfinished_conduit_count += 1;
                }
            }
        }
        ensure!(
            unfinished_conduit_count == 0,
            "Conduits not properly matched!"
        );
        Ok(world)
    }
}

#[derive(Debug, Clone)]
pub struct WorldWithTapes {
    pub world: World,
    pub tapes: Vec<Tape<BasicInstr>>,
    pub repeat_length: usize,
}

impl WorldWithTapes {
    /// Normalizes `original_tape` to resolve Repeat and Reset instructions.
    fn normalize_instructions(
        original_arm: &Arm,
        original_tape: &Tape<Instr>,
        track_maps: &TrackMaps,
    ) -> Result<Tape<BasicInstr>> {
        use ArmType::*;
        use Instr::*;
        let arm_type = original_arm.arm_type;
        let old_instructions = &original_tape.instructions;
        let mut instructions: Vec<BasicInstr> = Vec::with_capacity(old_instructions.len() + 8);
        let mut repeat_source = 0;
        let mut repeat_ending = 0;
        let mut any_nonrepeat = false;
        let mut curr = 0;
        let mut track_steps = 0;
        let mut rot_diff = 0;
        let mut len = original_arm.len;
        let mut known_grab = false;
        let mut position_check = original_arm.pos;
        fn get_track_loop_length(track_map: &TrackMap, pos: Pos) -> Option<i32> {
            let mut my_pos = pos;
            let mut loop_length = 0;
            loop {
                let offset = track_map.get(&my_pos)?;
                my_pos += offset;
                loop_length += 1;
                if my_pos == pos {
                    return Some(loop_length);
                }
                if loop_length as usize > track_map.len() {
                    return None;
                }
            }
        }
        let track_loop = get_track_loop_length(&track_maps.plus, original_arm.pos);
        while curr < old_instructions.len() {
            let instr = old_instructions[curr];
            if !any_nonrepeat && !matches!(instr, Repeat | Empty) {
                any_nonrepeat = true;
                repeat_source = curr;
            }
            if let Ok(basic_instr) = instr.try_into() {
                instructions.push(basic_instr);
                curr += 1;
            }
            let mut basic_move = |instr: BasicInstr| -> Result<()> {
                match instr {
                    BasicInstr::Extend => {
                        if arm_type == Piston && len < 3 {
                            len += 1;
                        }
                    }
                    BasicInstr::Retract => {
                        if arm_type == Piston && len > 1 {
                            len -= 1;
                        }
                    }
                    BasicInstr::RotateCounterClockwise => {
                        rot_diff += 1;
                    }
                    BasicInstr::RotateClockwise => {
                        rot_diff -= 1;
                    }
                    BasicInstr::Grab => known_grab = true,
                    BasicInstr::Drop => known_grab = false,
                    BasicInstr::Forward => {
                        if let Some(offset) = track_maps.plus.get(&position_check) {
                            track_steps += 1;
                            position_check += offset;
                        }
                    }
                    BasicInstr::Back => {
                        if let Some(offset) = track_maps.minus.get(&position_check) {
                            track_steps -= 1;
                            position_check += offset;
                        }
                    }
                    BasicInstr::PivotCounterClockwise
                    | BasicInstr::PivotClockwise
                    | BasicInstr::Empty => {}
                }
                Ok(())
            };
            match instr {
                Extend
                | Retract
                | RotateCounterClockwise
                | RotateClockwise
                | Grab
                | Drop
                | Forward
                | Back
                | PivotCounterClockwise
                | PivotClockwise => {
                    basic_move(instr.try_into().unwrap())?;
                    repeat_ending = curr;
                }
                Empty => {}

                Repeat => {
                    let rep_len = repeat_ending - repeat_source;
                    if rep_len == 0 {
                        instructions.push(BasicInstr::Empty);
                        curr += 1;
                    } else {
                        for i in 0..rep_len {
                            let copied_instr = instructions[repeat_source + i];
                            ensure!(
                                i == 0
                                    || old_instructions.get(curr + i).unwrap_or(&Empty) == &Empty,
                                "Repeat instruction {:?} overlaps with {:?} on {}/{}. Repeat {}-{}",
                                copied_instr,
                                old_instructions.get(curr + i),
                                curr,
                                i,
                                repeat_source,
                                repeat_ending
                            );
                            instructions.push(copied_instr);
                            basic_move(copied_instr)?;
                        }
                        curr += rep_len;
                        any_nonrepeat = false;

                        known_grab = false;
                        rot_diff = 0;
                        len = original_arm.len;
                        track_steps = 0;
                        position_check = original_arm.pos;
                    }
                }
                Reset => {
                    let mut reset_vec: Vec<BasicInstr> = Vec::new();
                    if known_grab {
                        reset_vec.push(BasicInstr::Drop);
                        known_grab = false;
                    }
                    while len > original_arm.len {
                        reset_vec.push(BasicInstr::Retract);
                        len -= 1;
                    }
                    while rot_diff > 3 {
                        rot_diff -= 6;
                    }
                    while rot_diff < -3 {
                        rot_diff += 6;
                    }
                    while rot_diff > 0 {
                        reset_vec.push(BasicInstr::RotateClockwise);
                        rot_diff -= 1;
                    }
                    while rot_diff < 0 {
                        reset_vec.push(BasicInstr::RotateCounterClockwise);
                        rot_diff += 1;
                    }
                    // look for a path forward on the track that's shorter than
                    // the path backward.
                    if let Some(loop_len) = track_loop {
                        while track_steps > loop_len / 2 {
                            track_steps -= loop_len;
                        }
                        while track_steps < -loop_len / 2 {
                            track_steps += loop_len
                        }
                    }
                    while track_steps > 0 {
                        reset_vec.push(BasicInstr::Back);
                        track_steps -= 1;
                    }
                    while track_steps < 0 {
                        reset_vec.push(BasicInstr::Forward);
                        track_steps += 1;
                    }
                    while len < original_arm.len {
                        reset_vec.push(BasicInstr::Extend);
                        len += 1;
                    }
                    position_check = original_arm.pos;

                    if reset_vec.is_empty() {
                        reset_vec.push(BasicInstr::Empty);
                    };
                    for i in 0..reset_vec.len() {
                        ensure!(
                            i == 0 || old_instructions.get(curr + i).unwrap_or(&Empty) == &Empty,
                            "Reset instruction {:?} overlaps with {:?} on {}/{} (curr {})",
                            reset_vec[i],
                            old_instructions.get(curr + i),
                            i,
                            reset_vec.len(),
                            curr,
                        );
                        instructions.push(reset_vec[i]);
                    }
                    curr += reset_vec.len();
                    repeat_ending = curr;
                }
                Noop => {
                    instructions.push(BasicInstr::Empty);
                    curr += 1;
                }
            };
        }
        instructions.shrink_to_fit();
        let new_tape = Tape {
            first: original_tape.first,
            instructions,
        };
        Ok(new_tape)
    }

    /// The main initialization function. Creates the initial state of the world
    /// (including tape) given an `InitialWorld`.
    pub fn setup_sim(init: &InitialWorld) -> Result<Self> {
        let mut self_ = Self {
            world: World::setup_sim(init)?,
            tapes: Vec::new(),
            repeat_length: 0,
        };
        assert_eq!(self_.world.arms.len(), init.tapes.len());
        for (arm, init_tape) in self_.world.arms.iter().zip(init.tapes.iter()) {
            let tape = Self::normalize_instructions(arm, init_tape, &self_.world.track_maps)?;
            self_.tapes.push(tape);
        }
        self_.world.timestep = self_.get_first_timestep();
        self_.repeat_length = compute_tape_repeat_length(&self_.tapes);
        Ok(self_)
    }

    /// Get the instructions for the given timestep.
    pub fn get_instructions_at(&self, timestep: u64) -> Vec<BasicInstr> {
        self.tapes
            .iter()
            .map(|tape| tape.get(timestep as usize, self.repeat_length))
            .collect()
    }

    /// Get the instructions for the current timestep.
    pub fn get_instructions(&self) -> Vec<BasicInstr> {
        self.get_instructions_at(self.world.timestep)
    }

    pub fn run_step(
        &mut self,
        mark_area: bool,
        motion: &mut WorldStepInfo,
        float_world: &mut FloatWorld,
    ) -> SimResult<()> {
        self.world
            .run_step(mark_area, motion, float_world, &self.get_instructions())
    }

    pub fn prepare_step(&mut self, motion: &mut WorldStepInfo) -> SimResult<()> {
        self.world.prepare_step(motion, &self.get_instructions())
    }

    fn get_first_timestep(&self) -> u64 {
        if self.tapes.is_empty() {
            return 0;
        }
        self.tapes
            .iter()
            .map(|tape| tape.first as u64)
            .fold(u64::MAX, std::cmp::min)
    }

    pub fn get_stats(&self) -> SolutionStats {
        let cycles = (self.world.timestep - self.get_first_timestep())
            .try_into()
            .unwrap();
        let cost = self.world.cost;
        let area = self.world.area_touched.len().try_into().unwrap();
        let instructions = compute_tape_instruction_count(&self.tapes)
            .try_into()
            .unwrap();
        SolutionStats {
            cycles,
            cost,
            area,
            instructions,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rotation_tests() {
        for r in Rot::ALL {
            assert_eq!(r.dist_to_pos(1), rotate(Pos::new(1, 0), r));
            assert_eq!(r.dist_to_pos(2), rotate(Pos::new(2, 0), r));
            assert_eq!(rotate(Pos::new(3, 1), r + 1), rotate(Pos::new(-1, 4), r));
        }

        let p = Pos::new(1, 0);
        let fp = pos_to_xy(p);
        let r = Rot(1);
        let rp = rotate(p, r);
        let frp = pos_to_xy(rp);
        let rfp = nalgebra::Rotation2::new(rot_to_angle(r.raw_minabs())) * fp;
        assert!(nalgebra::distance(&frp, &rfp) < 1e-6);
        let rfp = nalgebra::Rotation2::new(rot_to_angle(r.raw_nonneg())) * fp;
        assert!(nalgebra::distance(&frp, &rfp) < 1e-6);

        for r in Rot::ALL {
            assert_eq!(r.to_pos(), -(r.opp().to_pos()));
        }
    }
}
