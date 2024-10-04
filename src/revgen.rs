use crate::parser::*;
use crate::sim::*;

use indexmap::IndexSet;
use rand::distributions::{WeightedError, WeightedIndex};
use rustc_hash::{FxHashMap, FxHashSet, FxHasher};
use slotmap::Key;
use slotmap::SecondaryMap;
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

#[derive(Clone, Debug)]
struct GenState {
    world: World,
    tapes: Vec<Tape<BasicInstr>>,
}

/// A spec for an arm that might already exist or might have to be synthesized.
#[derive(Copy, Clone, Debug)]
enum RevStepArmBase {
    Synthesize {
        arm_pos: Pos,
        arm_len: i32,
        arm_rot: Rot,
    },
    Existing {
        arm_idx: usize,
    },
}

/// A type of "reverse step" that can be taken, i.e. an action that, given a
/// current `GenState`, can potentially generate a new `GenState` with an
/// earlier timestep.
///
/// The intention of a `RevStep` is that each will be focused on doing something
/// potentially productive, targeted at (but not necessarily only affecting)
/// one _molecule_ at a time. For example, `RevStep::DropOutput` says that "the
/// specified arm has just dropped a molecule into the output glyph that it is
/// hovered over", which means that in order to generate this `RevStep` in the
/// first place, we need to select an arm that is over an output glyph (or
/// conjure such an arm out of thin air), and then in order to apply this
/// `RevStep` backwards (to take us from the current timestep T to the previous
/// timestep T-1) we need to synthesize the molecule on the output glyph, and
/// set the arm as grabbing.
///
/// Sometimes, after applying a `RevStep` successfully to generate a previous
/// `GenState`, applying the same tape instructions from this new `GenState` to
/// the end will cause a different final outcome. (For example, if the `RevStep`
/// placed a glyph in a position which is passed over later in the tape.) Let's
/// call this "collateral damage". If the simulation is able to continue to the
/// end of the tape without errors, then we'll just roll with it -- we'll just
/// amend the output glyphs to match the new final state.
///
/// The intended workflow is:
/// -   A function `generate_eligible_revsteps` takes a current `GenState`, and
///     returns a list of potential "syntactically valid" `RevStep`s.
///     (Syntactically valid here means e.g.  that any arm indexes reference
///     real arms that exist at the current state).
/// -   A function `with_revstep_applied` takes the current `GenState` and a
///     selected `RevStep`, and returns a new `GenState`. This operation is
///     deterministic. (In other words, given a seed `GenState` at the final
///     timestep, and a list of valid `RevStep`s, that's all that's needed to
///     walk back to the start state.)
/// -   A function `with_schedule_applied` that takes the current `GenState`
///     and:
///     -   repeatedly selects a random valid `RevStep` and attempts to `apply` it
///     -   potentially selects multiple independent `RevStep`s to execute in
///         parallel, if they involve distinct atoms and arms
///     -   resolves any errors (collisions) that arise, by inserting pause
///         frames or selecting a new parallel `RevStep` to affect the auxiliary
///         molecule in question
/// -   A function `check_and_update_outputs` that takes a `GenState` and
///     returns whether the world and tape are valid at all, updating output
///     glyphs as necessary (which could have changed due to collateral damage,
///     see below)
/// -   A function `eval` that takes a `GenState` and returns an evaluation,
///     e.g. penalizing unnecessary actions like HandoverMolecule -> DropOutput
/// -   A function `beam_search` that maintains a beam of n `GenState`s and
///     searches for a valid and highly-evaluating that also has `GrabInput`
///     revsteps for each of the molecules to start with
///
/// TODO list:
/// -   multiarm support: all revsteps that currently affect just the molecule
///     in the single-arm direction should be updated to support a non-empty
///     subset of all arm positions that the arm is over
/// -   eval
/// -   beam_search
///
/// A single forward timestep is:
/// -   execute instructions (grab/drop immediate, others buffered)
/// -   spawn inputs, do glyphs (first half -- consuming inputs), consume outputs
/// -   (in substeps) collision detection in float world
/// -   apply buffered motion to world
/// -   spawn inputs, do glyphs (second half -- spawning outputs)
///
/// So, a reverse timestep without collision detection is:
/// -   execute inverse instructions (grab/drop buffer 1, others buffer 2)
/// -   consume inputs, rev glyphs (second half -- consuming outputs)
/// -   apply buffer 2 motion to world
/// -   consume inputs, rev glyphs (first half -- spawning inputs), spawn outputs
/// -   apply buffer 1 motion to world
///
/// (Recall that a `RevStep` is zero-or-more timesteps, though.)
#[derive(Clone, Debug)]
enum RevStep {
    /// Required current state:
    /// - arm_base needs synthesizing, or existing arm_base is not grabbing
    /// - arm is over at least one output glyph with count > 0
    /// - all output positions are empty
    /// Reverse action:
    /// - (if required) spawn a new arm at arm_pos with arm_len and arm_rot
    /// - (if required) set arm type to piston
    /// - (if required) add to tape: rotate to arm_rot
    /// - (if required) add to tape: extend to arm_len
    /// - spawn a molecule at the given output
    /// - decrement output glyph count
    /// - add to tape: drop
    /// - set prev arm state to grabbed
    DropOutput {
        arm_base: RevStepArmBase,
        arm_len: i32,
        arm_rot: Rot,
        glyph_idx: usize,
    },
    /// Required current state:
    /// - existing arm_base is currently grabbing one or more molecules
    /// Reverse action:
    /// - (if required) other arms that are grabbing:
    ///   - add to tape of other arm: grab
    ///   - set previous other arm state to dropped
    /// - add to tape: rotate arm (to prev_rot via shortest sequence)
    /// - move molecule(s) currently grabbed
    /// - Check postconditions of glyphs under moved molecules (e.g. no previous
    /// salt atom
    RotateMolecule { arm_idx: usize, prev_arm_rot: Rot },
    /// Required current state:
    /// - prev_arm is over at least one molecule
    /// - molecule is currently not held, or held by a different arm from
    ///   prev_arm_base
    /// - prev_arm_base needs creating, or existing arm_base is not grabbing
    /// Reverse action:
    /// - (if required) spawn a new arm at arm_pos with arm_len and arm_rot
    /// - (if required) set arm type to piston
    /// - (if required) add to tape: rotate to arm_rot
    /// - (if required) add to tape: extend to arm_len
    /// - add to tape of current arm: grab
    /// - set current arm state to dropped (ALL atoms)
    /// - add to tape of prev arm: drop
    /// - set prev_arm_base arm state to grabbed
    HandoverMolecule {
        prev_arm_base: RevStepArmBase,
        prev_arm_len: i32,
        prev_arm_rot: Rot,
    },
    //PivotMolecule(Pos, usize, Rot, Rot),
    //Piston(Pos, usize, usize),
    //TrackMolecule(Pos, Pos),
    /// Required current state:
    /// - existing arm_idx is already grabbing
    /// - _every_ grabbed molecule is either over an exact equal input glyph, or
    ///   over nothing at all
    /// Reverse action:
    /// - spawn an input glyph
    /// - add to tape: grab
    /// - set prev arm state to ungrabbed
    /// - remove all molecules
    GrabInput { arm_idx: usize },
}

struct WorldInfoMolecule {
    atom_pattern: Vec<AtomKey>,
    /// list of arm_idxs that grab this molecule
    grabbed_by: BTreeSet<usize>,
}

/// Extra information that can be computed on a `World` for convenience/fast
/// access.
struct WorldInfo {
    /// map from atom_key to list of arm_idxs that grab it
    grab_map: SecondaryMap<AtomKey, Vec<usize>>,
    /// Vector of (atom_pattern, arm_idxs that grab this molecule)
    molecule_map: Vec<WorldInfoMolecule>,
    /// Map from position to molecule_map_idx
    molecule_locs: FxHashMap<Pos, usize>,
    /// Map from position to glyph_idx
    glyph_locs: FxHashMap<Pos, usize>,
    /// Map from position to arm_idx
    arm_locs: FxHashMap<Pos, usize>,
}

impl WorldInfo {
    fn new(world: &World) -> Self {
        let mut self_ = Self {
            grab_map: SecondaryMap::new(),
            molecule_map: Vec::new(),
            molecule_locs: FxHashMap::default(),
            glyph_locs: FxHashMap::default(),
            arm_locs: FxHashMap::default(),
        };
        for (arm_idx, arm) in world.arms.iter().enumerate() {
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
        for atom_pattern in world.atoms.iter_molecules() {
            let molecule_key = self_.molecule_map.len();
            let mut grabbed_by = BTreeSet::new();
            for atom_key in atom_pattern.iter().copied() {
                let atom = world.atoms.atom_map.get(atom_key).unwrap();
                let old_value = self_.molecule_locs.insert(atom.pos, molecule_key);
                assert!(old_value.is_none());
                if let Some(arm_idxs) = self_.grab_map.get(atom_key) {
                    grabbed_by.extend(arm_idxs);
                }
            }
            self_.molecule_map.push(WorldInfoMolecule {
                atom_pattern,
                grabbed_by,
            });
        }
        for (glyph_idx, glyph) in world.glyphs.iter().enumerate() {
            for pos in glyph.positions() {
                self_.glyph_locs.insert(pos, glyph_idx);
            }
        }
        for (i, arm) in world.arms.iter().enumerate() {
            self_.arm_locs.insert(arm.pos, i);
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
                    .map(|(pattern, count)| {
                        Glyph::new(GlyphType::Output {
                            pattern,
                            count,
                            id: -1,
                        })
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

    fn outputs(&self) -> impl Iterator<Item = (usize, &AtomPattern, &i32, &i32)> {
        self.world
            .glyphs
            .iter()
            .enumerate()
            .filter_map(|(glyph_idx, glyph)| match &glyph.glyph_type {
                GlyphType::Output { pattern, count, id } => Some((glyph_idx, pattern, count, id)),
                _ => None,
            })
    }

    /// Returns revsteps with weight
    fn generate_eligible_revsteps(
        &self,
        world_info: &WorldInfo,
        input_weight: f32,
    ) -> Vec<(RevStep, f32)> {
        let mut revsteps = Vec::new();

        {
            // DropOutput
            let mut existing_arm = Vec::new();
            let mut synth_arm = Vec::new();
            for (glyph_idx, pattern, &output_count, _id) in self.outputs() {
                // current timestamp: output must have been output at least once
                // (we're going to decrease this count when revstepping)
                if output_count == 0 {
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
                            if let Some(arm_idx) = world_info.arm_locs.get(&arm_pos).copied() {
                                if self.world.arms[arm_idx].grabbing {
                                    continue;
                                }
                                existing_arm.push(RevStep::DropOutput {
                                    arm_base: RevStepArmBase::Existing { arm_idx },
                                    arm_len,
                                    arm_rot,
                                    glyph_idx,
                                });
                            } else {
                                synth_arm.push(RevStep::DropOutput {
                                    arm_base: RevStepArmBase::Synthesize {
                                        arm_pos,
                                        arm_len,
                                        arm_rot,
                                    },
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
            let weight = 10. / synth_arm.len() as f32;
            revsteps.extend(synth_arm.into_iter().map(|r| (r, weight)));
        }

        {
            // RotateMolecule
            let mut existing_arm = Vec::new();
            for (arm_idx, arm) in self.world.arms.iter().enumerate() {
                if arm.atoms_grabbed[0].is_null() {
                    continue;
                }
                let cur_arm_rot = arm.rot.normalize();
                for prev_arm_rot in Rot::ALL {
                    if prev_arm_rot == cur_arm_rot {
                        continue;
                    }
                    existing_arm.push(RevStep::RotateMolecule {
                        arm_idx,
                        prev_arm_rot,
                    });
                }
            }
            let weight = 10. / existing_arm.len() as f32;
            revsteps.extend(existing_arm.into_iter().map(|r| (r, weight)));
        }

        {
            // HandoverMolecule
            let mut existing_arm = Vec::new();
            let mut synth_arm = Vec::new();
            for molecule in world_info.molecule_map.iter() {
                for atom_key in molecule.atom_pattern.iter().copied() {
                    let atom = self.world.atoms.atom_map.get(atom_key).unwrap();
                    for arm_len in 1..3 {
                        for cur_arm_rot in Rot::ALL {
                            for prev_arm_rot in Rot::ALL {
                                if prev_arm_rot == cur_arm_rot {
                                    continue;
                                }
                                let arm_pos = atom.pos - cur_arm_rot.dist_to_pos(arm_len);
                                if let Some(arm_idx) = world_info.arm_locs.get(&arm_pos).copied() {
                                    if self.world.arms[arm_idx].grabbing {
                                        continue;
                                    }
                                    existing_arm.push(RevStep::HandoverMolecule {
                                        prev_arm_base: RevStepArmBase::Existing { arm_idx },
                                        prev_arm_rot,
                                        prev_arm_len: arm_len,
                                    });
                                } else {
                                    synth_arm.push(RevStep::HandoverMolecule {
                                        prev_arm_base: RevStepArmBase::Synthesize {
                                            arm_pos,
                                            arm_len,
                                            arm_rot: cur_arm_rot,
                                        },
                                        prev_arm_rot,
                                        prev_arm_len: arm_len,
                                    });
                                }
                            }
                        }
                    }
                }
            }
            let weight = 10. / existing_arm.len() as f32;
            revsteps.extend(existing_arm.into_iter().map(|r| (r, weight)));
            let weight = 10. / synth_arm.len() as f32;
            revsteps.extend(synth_arm.into_iter().map(|r| (r, weight)));
        }

        {
            // GrabInput
            for molecule in world_info.molecule_map.iter() {
                if molecule.atom_pattern.iter().copied().any(|atom_key| {
                    let atom = self.world.atoms.atom_map.get(atom_key).unwrap();
                    world_info.glyph_locs.contains_key(&atom.pos)
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

    fn maybe_spawn_arm_base(&mut self, arm_base: RevStepArmBase) -> usize {
        match arm_base {
            RevStepArmBase::Synthesize {
                arm_pos,
                arm_len,
                arm_rot,
            } => {
                let arm_idx = self.world.arms.len();
                self.world
                    .arms
                    .push(Arm::new(arm_pos, arm_rot.raw_minabs(), arm_len, ArmType::PlainArm));
                self.tapes.push(Tape {
                    first: 0,
                    instructions: Vec::new(),
                });
                arm_idx
            }
            RevStepArmBase::Existing { arm_idx } => arm_idx,
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
        if delta_rot == 0 {
            return;
        }
        let instruction = if delta_rot >= 0 {
            BasicInstr::RotateCounterClockwise
        } else {
            BasicInstr::RotateClockwise
        };
        insns_to_prepend
            .extend(std::iter::repeat((arm_idx, instruction)).take(delta_rot.abs() as usize));
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
        let instruction = if delta_len >= 0 {
            BasicInstr::Extend
        } else {
            BasicInstr::Retract
        };
        insns_to_prepend
            .extend(std::iter::repeat((arm_idx, instruction)).take(delta_len.abs() as usize));

        // upgrade the arm to a piston if necessary
        match &mut self.world.arms[arm_idx].arm_type {
            arm_type @ ArmType::PlainArm => *arm_type = ArmType::Piston,
            ArmType::Piston => (),
            arm_type => return Err(eyre!("can't extend/retract this arm type: {:?}", arm_type)),
        }
        Ok(())
    }

    fn maybe_spawn_input_glyph(
        &mut self,
        world_info: &WorldInfo,
        molecule_idx: usize,
    ) -> Result<()> {
        let molecule = &world_info.molecule_map[molecule_idx];
        match world_info
            .glyph_locs
            .get(&self.world.atoms.atom_map[*molecule.atom_pattern.first().unwrap()].pos)
            .map(|&glyph_idx| &self.world.glyphs[glyph_idx].glyph_type)
        {
            Some(GlyphType::Input { pattern, .. }) => {
                let glyph_keys: FxHashSet<AtomKey> = pattern
                    .iter()
                    .map(|atom| {
                        let atom_key = self.world.atoms.locs[&atom.pos];
                        if &self.world.atoms.atom_map[atom_key] != atom {
                            return Err(eyre!("wrong atom in input glyph"));
                        }
                        Ok(atom_key)
                    })
                    .collect::<Result<FxHashSet<AtomKey>>>()?;
                let molecule_keys: FxHashSet<AtomKey> =
                    molecule.atom_pattern.iter().copied().collect();
                // check that the input glyph pattern matches
                if molecule_keys != glyph_keys {
                    return Err(eyre!("wrong input glyph"));
                }
            }
            Some(glyph_type) => {
                return Err(eyre!("wrong glyph: {:?}", glyph_type));
            }
            None => {
                // spawn the input glyph
                self.world.glyphs.push(Glyph::new(GlyphType::Input {
                    pattern: molecule
                        .atom_pattern
                        .iter()
                        .map(|&atom_key| self.world.atoms.atom_map[atom_key].clone())
                        .collect(),
                    id: -1,
                }));
            }
        }
        Ok(())
    }

    fn despawn_molecule(&mut self, world_info: &WorldInfo, molecule_idx: usize) {
        for atom_key in world_info.molecule_map[molecule_idx]
            .atom_pattern
            .iter()
            .copied()
        {
            self.world.atoms.take_atom(atom_key);
        }
    }

    fn with_revstep_applied(&self, world_info: &WorldInfo, revstep: &RevStep) -> Result<Self> {
        let mut new = self.clone();
        let mut insns_to_prepend = Vec::new();
        match revstep {
            RevStep::DropOutput {
                arm_base,
                arm_len,
                arm_rot,
                glyph_idx,
            } => {
                let arm_idx = new.maybe_spawn_arm_base(*arm_base);
                new.maybe_prepend_rotate(
                    &mut insns_to_prepend,
                    arm_idx,
                    *arm_rot,
                    new.world.arms[arm_idx].rot.normalize(),
                );
                new.maybe_prepend_extend(
                    &mut insns_to_prepend,
                    arm_idx,
                    *arm_len,
                    new.world.arms[arm_idx].len,
                )?;

                let (atom_pattern, output_count) = {
                    if let GlyphType::Output {
                        pattern: atom_pattern,
                        count: output_count,
                        ..
                    } = &mut new.world.glyphs[*glyph_idx].glyph_type
                    {
                        (&*atom_pattern, output_count)
                    } else {
                        panic!("DropOutput but not an output glyph")
                    }
                };
                for atom in atom_pattern {
                    new.world.atoms.create_atom(atom.clone())?;
                }
                *output_count = *output_count + 1;

                insns_to_prepend.push((arm_idx, BasicInstr::Drop));
            }
            &RevStep::RotateMolecule {
                arm_idx,
                prev_arm_rot,
            } => {
                let cur_arm_rot = self.world.arms[arm_idx].rot.normalize();
                let atom_pos = self.world.arms[arm_idx].pos
                    + cur_arm_rot.dist_to_pos(self.world.arms[arm_idx].len);
                assert!(prev_arm_rot != cur_arm_rot);

                assert!(new.world.arms[arm_idx].grabbing);

                // force any existing arm that's holding the molecule to have
                // previously dropped it.
                for &other_arm_idx in world_info.molecule_map[world_info.molecule_locs[&atom_pos]]
                    .grabbed_by
                    .iter()
                {
                    insns_to_prepend.push((other_arm_idx, BasicInstr::Grab));
                    new.world.arms[arm_idx].grabbing = false;
                    new.world.arms[arm_idx].atoms_grabbed = [AtomKey::null(); 6];
                }

                new.maybe_prepend_rotate(&mut insns_to_prepend, arm_idx, prev_arm_rot, cur_arm_rot);
            }
            &RevStep::HandoverMolecule {
                prev_arm_base,
                prev_arm_len,
                prev_arm_rot,
            } => {
                let arm_idx = new.maybe_spawn_arm_base(prev_arm_base);
                let arm_pos = new.world.arms[arm_idx].pos;
                let cur_arm_rot = new.world.arms[arm_idx].rot.normalize();
                let cur_arm_len = new.world.arms[arm_idx].len;
                let atom_pos = arm_pos + cur_arm_rot.dist_to_pos(cur_arm_len);
                new.maybe_prepend_rotate(
                    &mut insns_to_prepend,
                    arm_idx,
                    prev_arm_rot,
                    new.world.arms[arm_idx].rot.normalize(),
                );
                new.maybe_prepend_extend(
                    &mut insns_to_prepend,
                    arm_idx,
                    prev_arm_len,
                    new.world.arms[arm_idx].len,
                )?;

                // if the arm is new, or is currently not grabbing, insert the
                // instructions to make this true (given that previously it was
                // grabbing)
                let cur_arm_grab = new.world.arms[arm_idx].grabbing;
                if !cur_arm_grab {
                    insns_to_prepend.push((arm_idx, BasicInstr::Drop));
                    // TODO: multiarm support. A multiarm can choose which
                    // collateral molecules to pick up and rotate incidentally
                }

                // force any existing arm that's holding the molecule to have
                // previously dropped it.
                for &other_arm_idx in world_info.molecule_map[world_info.molecule_locs[&atom_pos]]
                    .grabbed_by
                    .iter()
                {
                    insns_to_prepend.push((other_arm_idx, BasicInstr::Grab));
                }
            }
            &RevStep::GrabInput { arm_idx } => {
                let arm = &self.world.arms[arm_idx];
                assert!(arm.grabbing);
                for r in Rot::step_by(arm.arm_type.angles_between_arm()) {
                    let r = r + arm.rot.normalize();
                    let atom_key = arm.atoms_grabbed[r.to_usize()];
                    if atom_key.is_null() {
                        continue;
                    }
                    let atom_pos = arm.pos + r.dist_to_pos(arm.len);
                    let molecule_idx = *world_info.molecule_locs.get(&atom_pos).unwrap();

                    new.maybe_spawn_input_glyph(world_info, molecule_idx)?;
                }
                insns_to_prepend.push((arm_idx, BasicInstr::Grab));
            }
        }

        // now, actually prepend the instructions
        new.try_prepend_instructions(insns_to_prepend);

        Ok(new)
    }

    /// Prepend the instructions into the tape, and also execute them in the world
    fn try_prepend_instructions(&mut self, insns_to_prepend: Vec<(usize, BasicInstr)>) {
        let mut insns_to_prepend_by_arm_idx = vec![Vec::new(); self.world.arms.len()];
        for (arm_idx, insn) in insns_to_prepend {
            insns_to_prepend_by_arm_idx[arm_idx].push(insn);
        }
        let n_insns_to_prepend = insns_to_prepend_by_arm_idx
            .iter()
            .map(|insns| insns.len())
            .max()
            .unwrap();
        for arm_idx in 0..self.world.arms.len() {
            let insns = &mut self.tapes[arm_idx].instructions;
            insns.resize(insns.len() + n_insns_to_prepend, BasicInstr::Empty);
            insns.rotate_right(n_insns_to_prepend);
            for (i, insn) in insns_to_prepend_by_arm_idx[arm_idx]
                .iter()
                .copied()
                .enumerate()
            {
                insns[i] = insn;
            }
        }

        // move in the world
        let mut motion = WorldStepInfo::new();
        for timestep in (0..n_insns_to_prepend).rev() {
            motion.clear();
            for arm_idx in 0..self.world.arms.len() {
                let rev_insn = match self.tapes[arm_idx].instructions[timestep] {
                    BasicInstr::Extend => BasicInstr::Retract,
                    BasicInstr::Retract => BasicInstr::Extend,
                    BasicInstr::RotateCounterClockwise => BasicInstr::RotateClockwise,
                    BasicInstr::RotateClockwise => BasicInstr::RotateCounterClockwise,
                    fwd_insn => unimplemented!("{:?}", fwd_insn),
                };
                self.world
                    .do_instruction(&mut motion, arm_idx, rev_insn)
                    .unwrap();
            }
            self.world.apply_motion(&motion).unwrap();
        }
    }

    pub fn with_schedule_applied(&self, rng: &mut impl Rng) -> Result<Self> {
        let world_info = WorldInfo::new(&self.world);
        let revsteps = self.generate_eligible_revsteps(&world_info, 10.);

        println!("number of eligible revstep candidates: {}", revsteps.len());

        let mut distr = WeightedIndex::new(revsteps.iter().map(|(_, w)| *w))?;
        loop {
            let revstep_idx = rng.sample(&distr);
            let revstep = &revsteps[revstep_idx].0;
            println!("trying to apply {:?}", revstep);
            match self.with_revstep_applied(&world_info, revstep) {
                Ok(mut new_self) => {
                    println!("successfully applied {:?}, validating", revstep);
                    // validate that this new_self works
                    new_self.update_outputs();
                    match new_self.check() {
                        Ok(()) => {
                            println!("validation passed!");
                            break Ok(new_self);
                        }
                        Err(err) => {
                            println!(
                                "revstep {:?} failed validation, retrying: {:?}",
                                revstep, err
                            );
                        }
                    }
                }
                Err(err) => {
                    println!(
                        "revstep {:?} failed to apply, retrying with another: {:?}",
                        revstep, err
                    );
                }
            }

            // try again, but removing this revstep from the candidate pool
            match distr.update_weights(&[(revstep_idx, &0.)]) {
                Ok(()) => (),
                Err(WeightedError::AllWeightsZero) => {
                    break Err(eyre!("none of the eligible revsteps worked"))
                }
                Err(err) => break Err(err.into()),
            }
        }
    }

    /// Spawn input glyphs under all molecules
    pub fn spawn_inputs(&mut self, world_info: &WorldInfo) -> Result<()> {
        let mut insns_to_prepend = Vec::new();
        for (molecule_idx, molecule) in world_info.molecule_map.iter().enumerate() {
            for arm_idx in molecule.grabbed_by.iter().copied() {
                self.world.arms[arm_idx].grabbing = false;
                self.world.arms[arm_idx].atoms_grabbed = [AtomKey::null(); 6];
                insns_to_prepend.push((arm_idx, BasicInstr::Grab));
            }
            self.maybe_spawn_input_glyph(world_info, molecule_idx)?;
            self.despawn_molecule(world_info, molecule_idx);
        }
        Ok(())
    }

    /// Update outputs for collateral damage
    pub fn update_outputs(&mut self) {
        // TODO
    }

    /// Check that the current state `self` plays out to the end okay
    pub fn check(&self) -> Result<()> {
        // TODO
        Ok(())
    }

    pub fn to_serializable(&self) -> (FullPuzzle, FullSolution) {
        let mut world_with_tapes = WorldWithTapes {
            world: self.world.clone(),
            tapes: self.tapes.clone(),
            repeat_length: compute_tape_repeat_length(&self.tapes),
        };

        let mut inputs = Vec::new();
        let mut outputs = Vec::new();

        for glyph in world_with_tapes.world.glyphs.iter_mut() {
            match &mut glyph.glyph_type {
                GlyphType::Input { pattern, id } => {
                    *id = inputs.len().try_into().unwrap();
                    inputs.push(pattern.clone());
                }
                GlyphType::Output { pattern, id, .. } => {
                    *id = outputs.len().try_into().unwrap();
                    outputs.push(pattern.clone());
                }
                _ => (),
            }
        }

        let puzzle_name = "revgen".to_string();
        let solution_name = "revgen".to_string();
        let full_puzzle = FullPuzzle {
            puzzle_name: puzzle_name.clone(),
            creator_id: 1234,
            allowed_parts: AllowedParts::all(),
            inputs,
            outputs,
            output_multiplier: 1,
            production: false,
        };

        let full_solution = create_solution(&world_with_tapes, puzzle_name, solution_name, None);

        (full_puzzle, full_solution)
    }
}

pub fn main() -> Result<()> {
    let rng = &mut rand_pcg::Pcg64::seed_from_u64(123);
    //let rng = &mut thread_rng();
    let molecule = gen_molecule(rng, 1);

    let gen_state = GenState::new(vec![(molecule.clone(), 1)]);
    let gen_state = gen_state.with_schedule_applied(rng)?;
    let mut gen_state = gen_state.with_schedule_applied(rng)?;
    //let gen_state = gen_state.revstep(rng)?;
    gen_state.spawn_inputs(&WorldInfo::new(&gen_state.world))?;
    println!("{:?}", gen_state.world);

    let (full_puzzle, full_solution) = gen_state.to_serializable();

    write_puzzle(
        &mut BufWriter::new(File::create_new("revgen.puzzle")?),
        &full_puzzle,
    )?;
    write_solution(
        &mut BufWriter::new(File::create_new("revgen.solution")?),
        &full_solution,
    )?;

    Ok(())
}
