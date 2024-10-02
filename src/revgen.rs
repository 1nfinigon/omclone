use crate::parser::*;
use crate::sim::*;

use indexmap::IndexSet;
use rustc_hash::{FxHashMap, FxHashSet, FxHasher};
use slotmap::SecondaryMap;
use slotmap::{new_key_type, SlotMap};
use std::fs::File;
use std::hash::BuildHasherDefault;
use std::io::BufWriter;

use eyre::Result;
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

struct GenState {
    world: World,
    tapes: Vec<Tape<BasicInstr>>,
}

enum RevStep {
    /// DropOutputNewArm(arm_pos, arm_len, rot)
    /// Required current state:
    /// - arm_pos is empty
    /// - all output positions are empty
    /// Reverse action:
    /// - spawn a new arm at arm_pos with arm_len and rot
    /// - spawn a molecule at the given output
    /// - add to tape tape: drop
    /// - set prev arm state to grabbed
    DropOutputNewArm(Pos, usize, Rot),
    /// DropOutputExistingArm(arm_idx, arm_len, rot)
    /// Required current state:
    /// - arm at arm_pos is not grabbing
    /// - all output positions are empty
    /// Reverse action:
    /// - set arm to piston if required
    /// - add to tape: rotate to rot
    /// - spawn a molecule at the given output
    /// - add to tape: drop
    /// - set prev arm state to grabbed
    DropOutputExistingArm(usize, usize, Rot),
    /// RotateMolecule(arm_pos, arm_len, from_rot, to_rot)
    /// Reverse action:
    /// - rotate arm to abs_rot via shortest sequence
    /// Eligible arms:
    /// - current arm can reach an atom if it were at the given
    ///   orientation (or doesn't exist)
    RotateMolecule(Pos, usize, Rot, Rot),
    //PivotMolecule(Pos, usize, Rot, Rot),
    //Piston(Pos, usize, usize),
    //TrackMolecule(Pos, Pos),
    /// GrabInput(arm_pos, arm_len, from_rot, input_relative_pos)
    GrabInput(Pos, usize, Rot, Pos),
}

new_key_type! { pub struct MoleculeKey; }

struct MoleculeInfo {
    molecule_map: SlotMap<MoleculeKey, Vec<AtomKey>>,
    locs: FxHashMap<Pos, MoleculeKey>,
}

impl MoleculeInfo {
    fn new(atoms: &WorldAtoms) -> Self {
        let dense_atoms: Vec<_> = atoms.atom_map.iter().collect();
        let key_to_dense_idx: SecondaryMap<AtomKey, usize> =
            dense_atoms.iter().enumerate().map(|(dense_idx, (atom_key, _atom))| (*atom_key, dense_idx)).collect();
        let mut disjoint_sets = QuickUnionUf::<UnionBySize>::new(dense_atoms.len());
        for (dense_idx, (_atom_key, atom)) in dense_atoms.iter().copied().enumerate() {
            for r in Rot::ALL {
                if atom.connections[r.to_usize()].intersects(Bonds::DYNAMIC_BOND) {
                    let other_pos = atom.pos + r.to_pos();
                    let other_key = atoms
                        .locs
                        .get(&other_pos)
                        .expect("Inconsistent atoms");
                    let other_dense_idx = key_to_dense_idx[*other_key];
                    disjoint_sets.union(dense_idx, other_dense_idx);
                }
            }
        }

        let mut disjoint_set_idx_to_molecule_key = FxHashMap::default();
        let mut molecule_map = SlotMap::with_key();
        let mut locs = FxHashMap::default();
        for (dense_idx, (atom_key, atom)) in dense_atoms.iter().copied().enumerate() {
            let disjoint_set_idx = disjoint_sets.find(dense_idx);
            let molecule_key = *disjoint_set_idx_to_molecule_key.entry(disjoint_set_idx).or_insert_with(|| {
                molecule_map.insert(vec![atom_key])
            });
            locs.insert(atom.pos, molecule_key);
        }

        Self { molecule_map, locs }
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

    fn outputs(&self) -> impl Iterator<Item = (&AtomPattern, i32)> {
        self.world
            .glyphs
            .iter()
            .filter_map(|glyph| match &glyph.glyph_type {
                GlyphType::Output(pattern, count, _) => Some((pattern, *count)),
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
    fn all_eligible_revsteps(&self) -> Vec<(RevStep, f32)> {
        let mut revsteps = Vec::new();

        let arms_by_pos = self.arms_by_pos();
        let molecules = MoleculeInfo::new(&self.world.atoms);

        {
            // DropOutput
            let mut drop_output_existing_arm = Vec::new();
            let mut drop_output_new_arm = Vec::new();
            for (pattern, output_count) in self.outputs() {
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
                                drop_output_existing_arm.push(RevStep::DropOutputExistingArm(
                                    arm_idx,
                                    arm_len as usize,
                                    arm_rot,
                                ));
                            } else {
                                drop_output_new_arm.push(RevStep::DropOutputNewArm(
                                    arm_pos,
                                    arm_len as usize,
                                    arm_rot,
                                ));
                            }
                        }
                    }
                }
            }
            let weight = 10. / drop_output_existing_arm.len() as f32;
            revsteps.extend(drop_output_existing_arm.into_iter().map(|r| (r, weight)));
            let weight = 10. / drop_output_new_arm.len() as f32;
            revsteps.extend(drop_output_new_arm.into_iter().map(|r| (r, weight)));
        }

        {
            // RotateMolecule
            for (_, molecule) in molecules.molecule_map {
            }
        }

        revsteps
    }

    fn apply_revstep(&self) -> Result<Self> {
        unimplemented!()
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
