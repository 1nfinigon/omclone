use crate::parser::*;
use crate::sim::*;

use indexmap::IndexSet;
use rand::distributions::{WeightedError, WeightedIndex};
use rustc_hash::{FxHashMap, FxHashSet, FxHasher};
use slotmap::Key;
use slotmap::SecondaryMap;
use slotmap::{new_key_type, SlotMap};
use std::collections::BTreeSet;
use std::fs::File;
use std::hash::BuildHasherDefault;
use std::io::BufWriter;

use eyre::{eyre, Result};
use rand::prelude::*;
use union_find::{QuickUnionUf, UnionBySize, UnionFind};

fn gen_atom_type(rng: &mut impl Rng) -> AtomType {
    AtomType::ALL_DYNAMIC.choose(rng).copied().unwrap()
}

fn gen_nonempty_bonds(rng: &mut impl Rng) -> Bonds {
    if rng.gen_bool(0.9) {
        Bonds::NORMAL
    } else {
        if rng.gen_bool(0.7) {
            Bonds::TRIPLEX
        } else {
            let triplex = rng.gen_range(1u8..(1 << 3));
            let bonds = Bonds::from_bits(triplex << 1).unwrap();
            assert!(bonds.difference(Bonds::TRIPLEX) == Bonds::NO_BOND);
            bonds
        }
    }
}

fn gen_molecule(rng: &mut impl Rng, size: usize) -> AtomPattern {
    assert!(size > 0);

    struct GenMoleculeState {
        pattern: AtomPattern,
        /// A map from position to index into `pattern`.
        occupied: FxHashMap<Pos, usize>,
        /// Contains empty positions that are adjacent to at least one occupied
        /// position.
        adjacencies: IndexSet<Pos, BuildHasherDefault<FxHasher>>,
        /// Contains (i,j) pairs of indices into `pattern`, of adjacent atoms.
        /// Exactly one of (i,j) and (j,i) are in this vec.
        possible_bonds: Vec<(usize, usize)>,
    }

    impl GenMoleculeState {
        fn new() -> Self {
            Self {
                pattern: AtomPattern::new(),
                occupied: FxHashMap::default(),
                adjacencies: IndexSet::default(),
                possible_bonds: Vec::new(),
            }
        }

        fn add_atom(&mut self, atom: Atom) {
            let pos = atom.pos;
            let pattern_idx = self.pattern.len();
            self.pattern.push(atom);
            let prev = self.occupied.insert(pos, pattern_idx);
            assert!(prev.is_none());
            self.adjacencies.swap_remove(&pos);
            for r in Rot::ALL {
                let pos = pos + r.to_pos();
                if let Some(other_idx) = self.occupied.get(&pos).copied() {
                    self.possible_bonds.push((pattern_idx, other_idx));
                } else {
                    // The game doesn't let you build out an input/output
                    // further than bestagon 4.
                    if hex_distance(pos) <= 4 {
                        self.adjacencies.insert(pos);
                    }
                }
            }
        }
    }

    let mut gen = GenMoleculeState::new();

    // Adds an atom, preserving the above invariants.
    // start with an atom in the centre
    gen.add_atom(Atom {
        pos: Pos::new(0, 0),
        atom_type: gen_atom_type(rng),
        connections: [Bonds::empty(); 6],
        is_berlo: false,
    });

    // generate more atoms around it
    for _ in 1..size {
        let pos = *gen
            .adjacencies
            .get_index(rng.gen_range(0..gen.adjacencies.len()))
            .unwrap();
        gen.add_atom(Atom {
            pos,
            atom_type: gen_atom_type(rng),
            connections: [Bonds::empty(); 6],
            is_berlo: false,
        });
    }

    // TODO: linear, rotational and reflectional symmetries

    // add random bonds between adjacent atoms
    let mut atom_disjoint_sets = QuickUnionUf::<UnionBySize>::new(gen.pattern.len());
    gen.possible_bonds.shuffle(rng);
    for (idx1, idx2) in gen.possible_bonds {
        let set_id1 = atom_disjoint_sets.find(idx1);
        let set_id2 = atom_disjoint_sets.find(idx2);
        let should_generate = set_id1 != set_id2 || rng.gen_bool(0.5);
        if !should_generate {
            continue;
        }
        let delta_1to2 = gen.pattern[idx2].pos - gen.pattern[idx1].pos;
        let r_1to2 = Rot::from_unit_pos(delta_1to2).unwrap();
        let r_2to1 = r_1to2.opp();
        let bonds = gen_nonempty_bonds(rng);
        gen.pattern[idx1].connections[r_1to2.to_usize()] = bonds;
        gen.pattern[idx2].connections[r_2to1.to_usize()] = bonds;
        atom_disjoint_sets.union(idx1, idx2);
    }

    assert!(
        (1..gen.pattern.len()).all(|i| atom_disjoint_sets.find(i) == atom_disjoint_sets.find(0))
    );

    gen.pattern
}

struct GenStateMolecule {
    pattern: AtomPattern,
    grabbed: Option<usize>,
}

struct GenStateOutput {
    pattern: AtomPattern,
    output_count: i32,
}

#[derive(Clone, Debug)]
struct GenState {
    world: World,
    tapes: Vec<Tape<BasicInstr>>,
}

#[derive(Copy, Clone)]
enum RevStepArmBase {
    /// New(arm_pos)
    New(Pos),
    /// Existing(arm_idx)
    Existing(usize),
}

/// A type of "reverse step" that can be taken, i.e. an action that, given a
/// current `GenState`, can potentially generate a new `GenState` with an
/// earlier timestep.
///
/// The intended workflow is:
/// -   A function `generate` takes a current `GenState`, and returns a list of
///     potential "syntactically valid" `RevStep`s. (Syntactically valid here
///     means e.g.  that any arm indexes reference real arms that exist at the
///     current state).
/// -   A function `apply` takes the current `GenState` and a selected
///     `RevStep`, and generates a new `GenState`. This operation is
///     deterministic. (In other words, given a seed `GenState` at the final
///     timestep, and a list of valid `RevStep`s, that's all that's needed to
///     walk back to the start state.)
/// -   A function `schedule` that takes the current `GenState` and:
///     -   repeatedly selects a random valid `RevStep` and attempts to `apply` it
///     -   potentially selects multiple independent `RevStep`s to execute in
///         parallel, if they involve distinct atoms and arms
///     -   resolves any errors (collisions) that arise, by inserting pause
///         frames or selecting a new parallel `RevStep` to affect the auxiliary
///         molecule in question
/// TODO: parallelization and scheduling of RevSteps doesn't exist yet
///
/// Sometimes, after `apply` applies a `RevStep` successfully to generate a
/// `GenState`, applying the same tape instructions from this new `GenState` to
/// the end will cause a different final outcome. (For example, if the `RevStep`
/// placed a glyph in a position which is passed over later in the tape.) Let's
/// call this "collateral damage". If the simulation is able to continue to the
/// end of the tape without errors, AND if each "drop" tape instruction that
/// stemmed from a `RevStep::DropOutput` drops a molecule that is not over any
/// other glyphs, then we'll just roll with it -- we'll just amend the output
/// glyphs to match the new final state.
/// TODO: this in't implemented yet.
enum RevStep {
    /// Required current state:
    /// - new arm_base is empty, or existing arm_base is not grabbing
    /// - all output positions are empty
    /// Reverse action:
    /// - (if required) spawn a new arm at arm_pos with arm_len and rot
    /// - (if required) set arm type to piston
    /// - (if required) add to tape: rotate to rot
    /// - spawn a molecule at the given output
    /// - add to tape: drop
    /// - set prev arm state to grabbed
    DropOutput {
        arm_base: RevStepArmBase,
        arm_len: i32,
        arm_rot: Rot,
        glyph_idx: usize,
    },
    /// Reverse action:
    /// - rotate arm to prev_rot via shortest sequence
    /// Eligible arms:
    /// - current arm can reach an atom if it were at the given
    ///   orientation (or doesn't exist)
    RotateMolecule {
        arm_base: RevStepArmBase,
        arm_len: i32,
        prev_arm_rot: Rot,
    },
    //PivotMolecule(Pos, usize, Rot, Rot),
    //Piston(Pos, usize, usize),
    //TrackMolecule(Pos, Pos),
    /// Reverse action:
    /// - spawn an input glyph
    /// - add to tape: grab
    /// Notes:
    /// - the input glyph _may_ have, but not
    GrabInput { arm_idx: usize },
}

struct MoleculeInfoMolecule {
    atom_pattern: Vec<AtomKey>,
    /// list of arm_idxs that grab this molecule
    grabbed_by: BTreeSet<usize>,
}

struct MoleculeInfo {
    /// map from atom_key to list of arm_idxs that grab it
    grab_map: SecondaryMap<AtomKey, Vec<usize>>,
    /// Vector of (atom_pattern, arm_idxs that grab this molecule)
    molecule_map: Vec<MoleculeInfoMolecule>,
    /// Map from position to molecule_map_idx
    locs: FxHashMap<Pos, usize>,
}

impl MoleculeInfo {
    fn new(atoms: &WorldAtoms, arms: &[Arm]) -> Self {
        let mut self_ = Self {
            grab_map: SecondaryMap::new(),
            molecule_map: Vec::new(),
            locs: FxHashMap::default(),
        };
        for (arm_idx, arm) in arms.iter().enumerate() {
            for r in Rot::ALL {
                let atom_key = arm.atoms_grabbed[r.to_usize()];
                if atom_key.is_null() {
                    continue;
                }
                self_
                    .grab_map
                    .entry(atom_key)
                    .unwrap()
                    .or_default()
                    .push(arm_idx);
            }
        }
        for atom_pattern in atoms.iter_molecules() {
            let molecule_key = self_.molecule_map.len();
            let mut grabbed_by = BTreeSet::new();
            for atom_key in atom_pattern.iter().copied() {
                let atom = atoms.atom_map.get(atom_key).unwrap();
                let old_value = self_.locs.insert(atom.pos, molecule_key);
                assert!(old_value.is_none());
                if let Some(arm_idxs) = self_.grab_map.get(atom_key) {
                    grabbed_by.extend(arm_idxs);
                }
            }
            self_.molecule_map.push(MoleculeInfoMolecule {
                atom_pattern,
                grabbed_by,
            });
        }
        self_
    }
}

struct GlyphInfo {
    /// Map from position to glyph_idx
    locs: FxHashMap<Pos, usize>,
}

impl GlyphInfo {
    fn new(glyphs: &[Glyph]) -> Self {
        let mut self_ = Self {
            locs: FxHashMap::default(),
        };
        for (glyph_idx, glyph) in glyphs.iter().enumerate() {
            for pos in glyph.positions() {
                self_.locs.insert(pos, glyph_idx);
            }
        }
        self_
    }
}

impl GenState {
    fn new(outputs: Vec<(AtomPattern, i32)>) -> Self {
        Self {
            world: World {
                timestep: 0,
                atoms: WorldAtoms::new(),
                area_touched: FxHashSet::default(),
                glyphs: outputs
                    .into_iter()
                    .enumerate()
                    .map(|(i, (pattern, count))| {
                        Glyph::new(GlyphType::Output(pattern, count, i as i32))
                    })
                    .collect(),
                arms: Vec::new(),
                track_maps: TrackMaps::default(),
                cost: 0,
                conduit_pairs: ConduitPairMap::default(),
            },
            tapes: Vec::new(),
        }
    }

    fn outputs(&self) -> impl Iterator<Item = (usize, &AtomPattern, i32)> {
        self.world
            .glyphs
            .iter()
            .enumerate()
            .filter_map(|(glyph_idx, glyph)| match &glyph.glyph_type {
                GlyphType::Output(pattern, count, _) => Some((glyph_idx, pattern, *count)),
                _ => None,
            })
    }

    fn arms_by_pos(&self) -> FxHashMap<Pos, usize> {
        self.world
            .arms
            .iter()
            .enumerate()
            .map(|(i, arm)| (arm.pos, i))
            .collect()
    }

    /// Returns revsteps with weight
    fn all_eligible_revsteps(&self, input_weight: f32) -> Vec<(RevStep, f32)> {
        let mut revsteps = Vec::new();

        let arms_by_pos = self.arms_by_pos();
        let molecules = MoleculeInfo::new(&self.world.atoms, &self.world.arms);
        let glyphs = GlyphInfo::new(&self.world.glyphs);

        {
            // DropOutput
            let mut existing_arm = Vec::new();
            let mut new_arm = Vec::new();
            for (glyph_idx, pattern, output_count) in self.outputs() {
                // current timestamp: output must have been output at least once
                // (we're going to decrease this count when revstepping)
                if output_count > 0 {
                    continue;
                }
                // current timestamp: output positions must be all clear
                if !pattern
                    .iter()
                    .all(|atom| self.world.atoms.get_type(atom.pos).is_none())
                {
                    continue;
                }

                // okay, all preconditions satisfied. now, iterate
                for atom in pattern.iter() {
                    for arm_len in 1..3 {
                        for arm_rot in Rot::ALL {
                            let arm_pos = atom.pos - arm_rot.dist_to_pos(arm_len);
                            if self.world.atoms.get_type(atom.pos).is_some() {
                                continue;
                            }
                            if let Some(arm_idx) = arms_by_pos.get(&arm_pos).copied() {
                                if self.world.arms[arm_idx].grabbing {
                                    continue;
                                }
                                existing_arm.push(RevStep::DropOutput {
                                    arm_base: RevStepArmBase::Existing(arm_idx),
                                    arm_len,
                                    arm_rot,
                                    glyph_idx,
                                });
                            } else {
                                new_arm.push(RevStep::DropOutput {
                                    arm_base: RevStepArmBase::New(arm_pos),
                                    arm_len,
                                    arm_rot,
                                    glyph_idx,
                                });
                            }
                        }
                    }
                }
            }
            let weight = 10. / existing_arm.len() as f32;
            revsteps.extend(existing_arm.into_iter().map(|r| (r, weight)));
            let weight = 10. / new_arm.len() as f32;
            revsteps.extend(new_arm.into_iter().map(|r| (r, weight)));
        }

        {
            // RotateMolecule
            // TODO: penalize if the previous RevStep on this molecule and/or
            // arm was a RotateMolecule; forbid if "and"
            let mut existing_arm = Vec::new();
            let mut new_arm = Vec::new();
            for molecule in molecules.molecule_map.iter() {
                for atom_key in molecule.atom_pattern.iter().copied() {
                    let atom = self.world.atoms.atom_map.get(atom_key).unwrap();
                    for arm_len in 1..3 {
                        for cur_arm_rot in Rot::ALL {
                            for prev_arm_rot in Rot::ALL {
                                if prev_arm_rot == cur_arm_rot {
                                    break;
                                }
                                let arm_pos = atom.pos - cur_arm_rot.dist_to_pos(arm_len);
                                if let Some(arm_idx) = arms_by_pos.get(&arm_pos).copied() {
                                    if self.world.arms[arm_idx].grabbing {
                                        continue;
                                    }
                                    existing_arm.push(RevStep::RotateMolecule {
                                        arm_base: RevStepArmBase::Existing(arm_idx),
                                        arm_len,
                                        prev_arm_rot,
                                    });
                                } else {
                                    new_arm.push(RevStep::RotateMolecule {
                                        arm_base: RevStepArmBase::New(arm_pos),
                                        arm_len,
                                        prev_arm_rot,
                                    });
                                }
                            }
                        }
                    }
                }
            }
            let weight = 10. / existing_arm.len() as f32;
            revsteps.extend(existing_arm.into_iter().map(|r| (r, weight)));
            let weight = 10. / new_arm.len() as f32;
            revsteps.extend(new_arm.into_iter().map(|r| (r, weight)));
        }

        {
            // GrabInput
            for molecule in molecules.molecule_map.iter() {
                if molecule.atom_pattern.iter().copied().any(|atom_key| {
                    let atom = self.world.atoms.atom_map.get(atom_key).unwrap();
                    glyphs.locs.contains_key(&atom.pos)
                }) {
                    continue;
                }
                if molecule.grabbed_by.len() != 1 {
                    continue;
                }
                let arm_idx = *molecule.grabbed_by.first().unwrap();

                revsteps.push((RevStep::GrabInput { arm_idx }, input_weight))
            }
        }

        revsteps
    }

    fn maybe_spawn_arm_base(
        &mut self,
        arm_base: RevStepArmBase,
        default_len: i32,
        default_rot: Rot,
    ) -> usize {
        match arm_base {
            RevStepArmBase::New(arm_base_pos) => {
                let arm_idx = self.world.arms.len();
                self.world.arms.push(Arm::new(
                    arm_base_pos,
                    default_rot.raw(),
                    default_len,
                    ArmType::PlainArm,
                ));
                self.tapes.push(Tape {
                    first: 0,
                    instructions: Vec::new(),
                });
                arm_idx
            }
            RevStepArmBase::Existing(arm_idx) => arm_idx,
        }
    }

    fn maybe_prepend_rotate(
        &mut self,
        insns_to_prepend: &mut Vec<(usize, BasicInstr)>,
        arm_idx: usize,
        prev_arm_rot: Rot,
        cur_arm_rot: Rot,
    ) {
        let delta_rot = (cur_arm_rot - prev_arm_rot).to_i32_minabs();
        let rot_instruction = if delta_rot >= 0 {
            BasicInstr::RotateCounterClockwise
        } else {
            BasicInstr::RotateClockwise
        };
        insns_to_prepend
            .extend(std::iter::repeat((arm_idx, rot_instruction)).take(delta_rot.abs() as usize));
    }

    fn maybe_prepend_extend(
        &mut self,
        insns_to_prepend: &mut Vec<(usize, BasicInstr)>,
        arm_idx: usize,
        prev_arm_len: i32,
        cur_arm_len: i32,
    ) -> Result<()> {
        let delta_len = cur_arm_len - prev_arm_len;
        if delta_len == 0 {
            return Ok(());
        }
        let extend_instruction = if delta_len >= 0 {
            BasicInstr::Extend
        } else {
            BasicInstr::Retract
        };
        match &mut self.world.arms[arm_idx].arm_type {
            arm_type @ ArmType::PlainArm => *arm_type = ArmType::Piston,
            ArmType::Piston => (),
            arm_type => return Err(eyre!("can't extend/retract this arm type: {:?}", arm_type)),
        }
        insns_to_prepend.extend(
            std::iter::repeat((arm_idx, extend_instruction)).take(delta_len.abs() as usize),
        );
        Ok(())
    }

    fn try_revstep(&self, revstep: &RevStep) -> Result<Self> {
        let mut new = self.clone();
        let mut insns_to_prepend = Vec::new();
        match revstep {
            RevStep::DropOutput {
                arm_base,
                arm_len,
                arm_rot,
                glyph_idx,
            } => {
                let arm_idx = new.maybe_spawn_arm_base(*arm_base, *arm_len, *arm_rot);
                new.maybe_prepend_rotate(
                    &mut insns_to_prepend,
                    arm_idx,
                    *arm_rot,
                    self.world.arms[arm_idx].rot.normalize(),
                );
                new.maybe_prepend_extend(
                    &mut insns_to_prepend,
                    arm_idx,
                    *arm_len,
                    self.world.arms[arm_idx].len,
                )?;

                let (atom_pattern, output_count) = {
                    if let GlyphType::Output(atom_pattern, output_count, _) =
                        &self.world.glyphs[*glyph_idx].glyph_type
                    {
                        (atom_pattern, output_count)
                    } else {
                        panic!("DropOutput but not an output glyph")
                    }
                };
                for atom in atom_pattern.iter() {}
                unimplemented!()
            }
            RevStep::RotateMolecule {
                arm_base,
                arm_len,
                prev_arm_rot,
            } => {
                let arm_idx = new.maybe_spawn_arm_base(*arm_base, *arm_len, *prev_arm_rot);
                unimplemented!()
            }
            RevStep::GrabInput { arm_idx } => {
                unimplemented!()
            }
        }
    }

    pub fn revstep(&self, rng: &mut impl Rng) -> Result<Self> {
        let revsteps = self.all_eligible_revsteps(10.);
        let mut distr = WeightedIndex::new(revsteps.iter().map(|(_, w)| *w))?;
        loop {
            let revstep_idx = rng.sample(&distr);
            match self.try_revstep(&revsteps[revstep_idx].0) {
                Ok(new_self) => break Ok(new_self),
                Err(err) => match distr.update_weights(&[(revstep_idx, &0.)]) {
                    Ok(()) => (),
                    Err(WeightedError::AllWeightsZero) => {
                        break Err(eyre!("none of the eligible revsteps worked"))
                    }
                    Err(err) => break Err(err.into()),
                },
            }
            unimplemented!()
        }
    }
}

pub fn main() -> Result<()> {
    let rng = &mut rand_pcg::Pcg64::seed_from_u64(123);
    //let rng = &mut thread_rng();
    let molecule = gen_molecule(rng, 12);

    let full_puzzle = FullPuzzle {
        puzzle_name: "revgen".to_string(),
        creator_id: 1234,
        allowed_parts: AllowedParts::all(),
        inputs: vec![molecule.clone()],
        outputs: vec![molecule.clone()],
        output_multiplier: 1,
        production: false,
    };

    write_puzzle(
        &mut BufWriter::new(File::create_new("revgen.puzzle")?),
        &full_puzzle,
    )?;

    Ok(())
}
