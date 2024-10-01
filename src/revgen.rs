use crate::parser::*;
use crate::sim::*;

use indexmap::IndexSet;
use rustc_hash::{FxHashMap, FxHasher};
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
            let pos = atom.pos.clone();
            let pattern_idx = self.pattern.len();
            self.pattern.push(atom);
            let prev = self.occupied.insert(pos, pattern_idx);
            assert!(prev.is_none());
            self.adjacencies.swap_remove(&pos);
            for r in 0..6 {
                let pos = pos + rot_to_pos(r);
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
        let pos = gen
            .adjacencies
            .get_index(rng.gen_range(0..gen.adjacencies.len()))
            .unwrap()
            .clone();
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
        let r_1to2 = pos_to_rot(delta_1to2).unwrap();
        let r_2to1 = pos_to_rot(-delta_1to2).unwrap();
        let bonds = gen_nonempty_bonds(rng);
        gen.pattern[idx1].connections[r_1to2 as usize] = bonds;
        gen.pattern[idx2].connections[r_2to1 as usize] = bonds;
        atom_disjoint_sets.union(idx1, idx2);
    }

    assert!((1..gen.pattern.len())
        .into_iter()
        .all(|i| atom_disjoint_sets.find(i) == atom_disjoint_sets.find(0)));

    gen.pattern
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
