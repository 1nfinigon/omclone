use crate::sim;
use smallvec::SmallVec;

pub mod constants {

    pub const N_WIDTH: usize = 16;
    pub const N_HEIGHT: usize = 16;
    pub const N_INPUTS: usize = 4;
    pub const N_OUTPUTS: usize = 4;
    pub const N_ARMS: usize = 12;
    pub const N_TRACKS: usize = 6;
    pub const N_HISTORY_CYCLES: usize = 2;
    pub const N_MAX_CYCLES: usize = 500;
    pub const N_MAX_PRODUCTS: usize = 6;

    /*
    pub const N_WIDTH: usize = 32;
    pub const N_HEIGHT: usize = 32;
    pub const N_INPUTS: usize = 10;
    pub const N_OUTPUTS: usize = 10;
    pub const N_ARMS: usize = 16;
    pub const N_TRACKS: usize = 8;
    pub const N_HISTORY_CYCLES: usize = 4;
    pub const N_MAX_CYCLES: usize = 1000;
    pub const N_MAX_PRODUCTS: usize = 6;
    */

    pub const N_MAX_AREA: usize = N_WIDTH * N_HEIGHT;
    pub const N_ORIENTATIONS: usize = 6;

    pub const N_BOND_TYPES: usize = 4;
    pub const N_ATOM_TYPES: usize = 17;
    pub const N_INSTR_TYPES: usize = 11;
}

use constants::*;

pub mod feature_offsets {
    use super::constants::*;
    use crate::sim;
    use num_traits::ToPrimitive;

    #[derive(Copy, Clone, Debug)]
    pub struct Float {
        pub offset: usize,
    }

    impl Float {
        pub fn get_offset(&self) -> usize {
            self.offset
        }

        const fn assign(offset: usize) -> (usize, Self) {
            (offset + 1, Self { offset })
        }
    }

    #[derive(Copy, Clone, Debug)]
    pub struct OneHot<const N: usize> {
        offset: usize,
    }

    impl<const N: usize> OneHot<N> {
        pub fn get_onehot_offset(&self, i: usize) -> usize {
            assert!(i < N, "one-hot index out of bounds");
            self.offset + i
        }

        pub fn get_offsets(&self) -> std::ops::Range<usize> {
            self.offset..(self.offset + N)
        }

        const fn assign(offset: usize) -> (usize, Self) {
            (offset + N, Self { offset })
        }

        const fn assign_array<const COUNT: usize>(mut offset: usize) -> (usize, [Self; COUNT]) {
            let mut array = [Self { offset: 0 }; COUNT];
            let mut i = 0;
            loop {
                if i >= COUNT {
                    break;
                }
                array[i].offset = offset;
                offset += 1;
                i += 1;
            }
            (offset, array)
        }

        const fn assign_array_2d<const COUNT1: usize, const COUNT2: usize>(
            mut offset: usize,
        ) -> (usize, [[Self; COUNT2]; COUNT1]) {
            let mut array = [[Self { offset: 0 }; COUNT2]; COUNT1];
            let mut i1 = 0;
            loop {
                if i1 >= COUNT1 {
                    break;
                }
                let mut i2 = 0;
                loop {
                    if i2 >= COUNT2 {
                        break;
                    }
                    array[i1][i2].offset = offset;
                    offset += 1;
                    i2 += 1;
                }
                i1 += 1;
            }
            (offset, array)
        }
    }

    pub type Binary = OneHot<1>;

    impl Binary {
        pub fn get_offset(&self) -> usize {
            self.offset
        }
    }

    #[derive(Clone, Debug)]
    pub struct Spatial {
        /// Information about the glyph that is in this space
        pub glyph_orientation: OneHot<N_ORIENTATIONS>,
        pub glyph_calcification: Binary,
        pub glyph_animismus: OneHot<4>,
        pub glyph_projection: OneHot<2>,
        pub glyph_purification: OneHot<3>,
        pub glyph_duplication: OneHot<2>,
        pub glyph_unification: OneHot<5>,
        pub glyph_dispersion: OneHot<5>,
        pub glyph_bonding: OneHot<2>,
        pub glyph_unbonding: OneHot<2>,
        pub glyph_triplex_bond: OneHot<2>,
        pub glyph_multi_bond: OneHot<2>,
        pub glyph_disposal: OneHot<7>,

        /// Information about the track that is in this space
        //track_id: OneHot<N_TRACKS>,
        pub track_plus_dir: OneHot<N_ORIENTATIONS>,
        pub track_minus_dir: OneHot<N_ORIENTATIONS>,

        /// Information about the input that is in this space
        //input_id: OneHot<N_INPUTS>,
        pub input_atom_type: OneHot<N_ATOM_TYPES>,
        pub input_bonds: [[Binary; N_BOND_TYPES]; N_ORIENTATIONS],

        /// Information about the output that is in this space
        //output_id: OneHot<N_INPUTS>,
        pub output_atom_type: OneHot<N_ATOM_TYPES>,
        pub output_bonds: [[Binary; N_BOND_TYPES]; N_ORIENTATIONS],
        pub output_count_minus_one: OneHot<N_MAX_PRODUCTS>,
    }

    impl Spatial {
        const fn new_and_size() -> (Self, usize) {
            let offset = 0usize;
            let (offset, glyph_orientation) = OneHot::assign(offset);
            let (offset, glyph_calcification) = Binary::assign(offset);
            let (offset, glyph_animismus) = OneHot::assign(offset);
            let (offset, glyph_projection) = OneHot::assign(offset);
            let (offset, glyph_purification) = OneHot::assign(offset);
            let (offset, glyph_duplication) = OneHot::assign(offset);
            let (offset, glyph_unification) = OneHot::assign(offset);
            let (offset, glyph_dispersion) = OneHot::assign(offset);
            let (offset, glyph_bonding) = OneHot::assign(offset);
            let (offset, glyph_unbonding) = OneHot::assign(offset);
            let (offset, glyph_triplex_bond) = OneHot::assign(offset);
            let (offset, glyph_multi_bond) = OneHot::assign(offset);
            let (offset, glyph_disposal) = OneHot::assign(offset);
            let (offset, track_plus_dir) = OneHot::assign(offset);
            let (offset, track_minus_dir) = OneHot::assign(offset);
            let (offset, input_atom_type) = OneHot::assign(offset);
            let (offset, input_bonds) = OneHot::assign_array_2d(offset);
            let (offset, output_atom_type) = OneHot::assign(offset);
            let (offset, output_bonds) = OneHot::assign_array_2d(offset);
            let (offset, output_count_minus_one) = OneHot::assign(offset);
            let self_ = Self {
                glyph_orientation,
                glyph_calcification,
                glyph_animismus,
                glyph_projection,
                glyph_purification,
                glyph_duplication,
                glyph_unification,
                glyph_dispersion,
                glyph_bonding,
                glyph_unbonding,
                glyph_triplex_bond,
                glyph_multi_bond,
                glyph_disposal,
                track_plus_dir,
                track_minus_dir,
                input_atom_type,
                input_bonds,
                output_atom_type,
                output_bonds,
                output_count_minus_one,
            };
            (self_, offset)
        }

        pub const OFFSETS: Self = {
            let (self_, _) = Self::new_and_size();
            self_
        };

        pub const SIZE: usize = {
            let (_, size) = Self::new_and_size();
            size
        };
    }

    #[derive(Clone, Debug)]
    pub struct Spatiotemporal {
        /// Information about the square
        pub used: Binary,

        /// Information about the arm base that is in this space
        pub arm_base_length: OneHot<3>,
        pub arm_base_is_piston: Binary,
        pub arm_base_is_grabbing: Binary,
        pub arm_base_is_berlo: Binary,
        pub arm_in_orientation: [Binary; N_ORIENTATIONS],
        pub arm_in_orientation_and_holding_atom: [Binary; N_ORIENTATIONS],
        /// What instruction is this arm about to execute this timestep
        pub arm_base_instr: OneHot<N_INSTR_TYPES>,

        /// Information about the atom that is in this space
        pub atom_type: OneHot<N_ATOM_TYPES>,
        pub atom_bonds: [[Binary; N_BOND_TYPES]; N_ORIENTATIONS],
        pub atom_is_berlo: Binary,
        //atom_is_grabbed
        //molecule_is_grabbed
    }

    impl Spatiotemporal {
        const fn new_and_size() -> (Self, usize) {
            let offset = 0usize;
            let (offset, used) = Binary::assign(offset);
            let (offset, arm_base_length) = OneHot::assign(offset);
            let (offset, arm_base_is_piston) = Binary::assign(offset);
            let (offset, arm_base_is_grabbing) = Binary::assign(offset);
            let (offset, arm_base_is_berlo) = Binary::assign(offset);
            let (offset, arm_in_orientation) = Binary::assign_array(offset);
            let (offset, arm_in_orientation_and_holding_atom) = Binary::assign_array(offset);
            let (offset, arm_base_instr) = OneHot::assign(offset);
            let (offset, atom_type) = OneHot::assign(offset);
            let (offset, atom_bonds) = OneHot::assign_array_2d(offset);
            let (offset, atom_is_berlo) = Binary::assign(offset);
            let self_ = Self {
                used,
                arm_base_length,
                arm_base_is_piston,
                arm_base_is_grabbing,
                arm_base_is_berlo,
                arm_in_orientation,
                arm_in_orientation_and_holding_atom,
                arm_base_instr,
                atom_type,
                atom_bonds,
                atom_is_berlo,
            };
            (self_, offset)
        }

        pub const OFFSETS: Self = {
            let (self_, _) = Self::new_and_size();
            self_
        };

        pub const SIZE: usize = {
            let (_, size) = Self::new_and_size();
            size
        };
    }

    pub struct Temporal {
        pub cycles: Float,
    }

    impl Temporal {
        const fn new_and_size() -> (Self, usize) {
            let offset = 0usize;
            let (offset, cycles) = Float::assign(offset);
            let self_ = Self { cycles };
            (self_, offset)
        }

        pub const OFFSETS: Self = {
            let (self_, _) = Self::new_and_size();
            self_
        };

        pub const SIZE: usize = {
            let (_, size) = Self::new_and_size();
            size
        };
    }
}

pub mod features {
    use super::feature_offsets::*;
    use super::*;

    #[derive(Clone)]
    pub struct Features {
        spatial: [[[f32; Spatial::SIZE]; N_WIDTH]; N_HEIGHT],
        spatiotemporal: [[[[f32; Spatiotemporal::SIZE]; N_WIDTH]; N_HEIGHT]; N_HISTORY_CYCLES],
        temporal: [[f32; Temporal::SIZE]; N_HISTORY_CYCLES],
    }

    impl Features {
        pub fn new() -> Self {
            Self {
                spatial: [[[0f32; Spatial::SIZE]; N_WIDTH]; N_HEIGHT],
                spatiotemporal: [[[[0f32; Spatiotemporal::SIZE]; N_WIDTH]; N_HEIGHT];
                    N_HISTORY_CYCLES],
                temporal: [[0f32; Temporal::SIZE]; N_HISTORY_CYCLES],
            }
        }

        fn normalize_position(pos: sim::Pos) -> Option<(usize, usize)> {
            if 0 <= pos.x && (pos.x as usize) < N_WIDTH && 0 <= pos.y && (pos.y as usize) < N_HEIGHT
            {
                Some((pos.x as usize, pos.y as usize))
            } else {
                None
            }
        }

        fn get_spatial_mut(&mut self, pos: sim::Pos) -> Option<&mut [f32; Spatial::SIZE]> {
            Self::normalize_position(pos).map(|(x, y)| &mut self.spatial[y][x])
        }

        fn get_spatiotemporal_mut(
            &mut self,
            time: usize,
            pos: sim::Pos,
        ) -> Option<&mut [f32; Spatiotemporal::SIZE]> {
            Self::normalize_position(pos).map(|(x, y)| &mut self.spatiotemporal[time][y][x])
        }

        /// Helper function for setting features relating to an Atom.
        fn set_atom(
            atom: &sim::Atom,
            features: &mut [f32],
            offset_atom_type: OneHot<N_ATOM_TYPES>,
            offset_atom_bonds: [[Binary; N_BOND_TYPES]; N_ORIENTATIONS],
            offset_atom_is_berlo: Option<Binary>,
        ) {
            let sim::Atom {
                pos,
                atom_type,
                connections,
                is_berlo,
            } = atom;
            use num_traits::ToPrimitive;
            features[offset_atom_type.get_onehot_offset(atom_type.to_usize().unwrap())] = 1.;
            for rot in 0..6 {
                if connections[rot].intersects(sim::Bonds::NORMAL) {
                    features[offset_atom_bonds[rot][0].get_offset()] = 1.;
                }
                if connections[rot].intersects(sim::Bonds::TRIPLEX_R) {
                    features[offset_atom_bonds[rot][1].get_offset()] = 1.;
                }
                if connections[rot].intersects(sim::Bonds::TRIPLEX_K) {
                    features[offset_atom_bonds[rot][2].get_offset()] = 1.;
                }
                if connections[rot].intersects(sim::Bonds::TRIPLEX_Y) {
                    features[offset_atom_bonds[rot][3].get_offset()] = 1.;
                }
            }
            match offset_atom_is_berlo {
                Some(offset_atom_is_berlo) => {
                    if *is_berlo {
                        features[offset_atom_is_berlo.get_offset()] = 1.;
                    }
                }
                None => assert!(!is_berlo),
            }
        }

        /// Set all nontemporal data.
        #[deny(unused_variables)]
        pub fn set_nontemporal(&mut self, world: &sim::World) {
            let Spatial {
                glyph_orientation,
                glyph_calcification,
                glyph_animismus,
                glyph_projection,
                glyph_purification,
                glyph_duplication,
                glyph_unification,
                glyph_dispersion,
                glyph_bonding,
                glyph_unbonding,
                glyph_triplex_bond,
                glyph_multi_bond,
                glyph_disposal,
                track_plus_dir,
                track_minus_dir,
                input_atom_type,
                input_bonds,
                output_atom_type,
                output_bonds,
                output_count_minus_one,
            } = Spatial::OFFSETS;

            for glyph in world.glyphs.iter() {
                let positions = glyph.positions();
                for (i, position) in positions.iter().enumerate() {
                    if let Some(features) = self.get_spatial_mut(*position) {
                        features[glyph_orientation
                            .get_onehot_offset(sim::normalize_dir(glyph.rot) as usize)] = 1.;
                        use sim::GlyphType;
                        match &glyph.glyph_type {
                            GlyphType::Calcification => {
                                features[glyph_calcification.get_onehot_offset(i)] = 1.
                            }
                            GlyphType::Animismus => {
                                features[glyph_animismus.get_onehot_offset(i)] = 1.
                            }
                            GlyphType::Projection => {
                                features[glyph_projection.get_onehot_offset(i)] = 1.
                            }
                            GlyphType::Dispersion => {
                                features[glyph_dispersion.get_onehot_offset(i)] = 1.
                            }
                            GlyphType::Purification => {
                                features[glyph_purification.get_onehot_offset(i)] = 1.
                            }
                            GlyphType::Duplication => {
                                features[glyph_duplication.get_onehot_offset(i)] = 1.
                            }
                            GlyphType::Unification => {
                                features[glyph_unification.get_onehot_offset(i)] = 1.
                            }
                            GlyphType::Bonding => features[glyph_bonding.get_onehot_offset(i)] = 1.,
                            GlyphType::Unbonding => {
                                features[glyph_unbonding.get_onehot_offset(i)] = 1.
                            }
                            GlyphType::TriplexBond => {
                                features[glyph_triplex_bond.get_onehot_offset(i)] = 1.
                            }
                            GlyphType::MultiBond => {
                                features[glyph_multi_bond.get_onehot_offset(i)] = 1.
                            }
                            GlyphType::Disposal => {
                                features[glyph_disposal.get_onehot_offset(i)] = 1.
                            }
                            GlyphType::Equilibrium => (),
                            GlyphType::Track(pos_list) => {
                                assert!(pos_list[i] == *position);
                                if i >= 1 {
                                    let other_pos = positions[i - 1];
                                    let rot = sim::pos_to_rot(other_pos - position).unwrap();
                                    features[track_minus_dir.get_onehot_offset(rot as usize)] = 1.;
                                }
                                if i + 1 < positions.len() {
                                    let other_pos = positions[i + 1];
                                    let rot = sim::pos_to_rot(other_pos - position).unwrap();
                                    features[track_plus_dir.get_onehot_offset(rot as usize)] = 1.;
                                }
                            }
                            GlyphType::Conduit(_, _) => unimplemented!(),
                            GlyphType::Input(pattern, _id) => {
                                assert!(pattern[i].pos == *position);
                                Self::set_atom(
                                    &pattern[i],
                                    features,
                                    input_atom_type,
                                    input_bonds,
                                    None,
                                );
                            }
                            GlyphType::Output(pattern, count_left, _id) => {
                                assert!(pattern[i].pos == *position);
                                Self::set_atom(
                                    &pattern[i],
                                    features,
                                    output_atom_type,
                                    output_bonds,
                                    None,
                                );
                                if *count_left > 0 {
                                    features[output_count_minus_one
                                        .get_onehot_offset((*count_left as usize) - 1)] = 1.;
                                }
                            }
                            GlyphType::OutputRepeating(_, _, _) => unimplemented!(),
                        };
                    }
                }
            }
        }

        /// Sets the temporal data at relative time `time` to reflect the current world.
        /// Does _not_ set `arm_base_instr`; see `set_temporal_instr` for that.
        #[deny(unused_variables)]
        pub fn set_temporal_except_instr(&mut self, time: usize, world: &sim::World) {
            let Spatiotemporal {
                used,
                arm_base_length,
                arm_base_is_piston,
                arm_base_is_grabbing,
                arm_base_is_berlo,
                arm_in_orientation,
                arm_in_orientation_and_holding_atom,
                arm_base_instr: _,
                atom_type,
                atom_bonds,
                atom_is_berlo,
            } = Spatiotemporal::OFFSETS;

            for position in world.area_touched.iter() {
                if let Some(features) = self.get_spatiotemporal_mut(time, *position) {
                    features[used.get_offset()] = 1.;
                }
            }

            for arm in world.arms.iter() {
                let sim::Arm {
                    pos,
                    rot,
                    len,
                    arm_type,
                    grabbing,
                    atoms_grabbed,
                } = arm;
                if let Some(features) = self.get_spatiotemporal_mut(time, *pos) {
                    features[arm_base_length.get_onehot_offset((len - 1).try_into().unwrap())] = 1.;
                    if *arm_type == sim::ArmType::Piston {
                        features[arm_base_is_piston.get_offset()] = 1.;
                    }
                    if *grabbing {
                        features[arm_base_is_grabbing.get_offset()] = 1.;
                    }
                    if *arm_type == sim::ArmType::VanBerlo {
                        features[arm_base_is_berlo.get_offset()] = 1.;
                    }
                    for rel_r in (0..6).step_by(arm_type.angles_between_arm() as usize) {
                        let abs_r = sim::normalize_dir(rel_r + rot);
                        features[arm_in_orientation[abs_r as usize].get_offset()] = 1.;
                        use slotmap::Key;
                        if !atoms_grabbed[rel_r as usize].is_null() {
                            features[arm_in_orientation_and_holding_atom[abs_r as usize]
                                .get_offset()] = 1.;
                        }
                    }
                }
            }

            for atom in world.atoms.atom_map.values() {
                let pos = atom.pos;
                if let Some(features) = self.get_spatiotemporal_mut(time, pos) {
                    Self::set_atom(atom, features, atom_type, atom_bonds, Some(atom_is_berlo));
                }
            }

            let Temporal { cycles } = Temporal::OFFSETS;
            self.temporal[time][cycles.get_offset()] = (world.timestep as f64 / 100.).tanh() as f32;
        }

        /// Set all timesteps' data to this world. Used for initialization.
        pub fn init_all_temporal(&mut self, world: &sim::World) {
            self.set_temporal_except_instr(0, world);
            for time in 1..N_HISTORY_CYCLES {
                self.spatiotemporal[time] = self.spatiotemporal[0];
                self.temporal[time] = self.temporal[0];
            }
        }

        pub fn set_temporal_instr(
            &mut self,
            time: usize,
            world: &sim::World,
            arm_idx: usize,
            instr: sim::BasicInstr,
        ) {
            let Spatiotemporal { arm_base_instr, .. } = Spatiotemporal::OFFSETS;
            let arm = &world.arms[arm_idx];
            let pos = arm.pos;
            if let Some(features) = self.get_spatiotemporal_mut(time, pos) {
                use num_traits::ToPrimitive;
                features[arm_base_instr.get_onehot_offset(instr.to_usize().unwrap())] = 1.;
            }
        }

        /// Copies all temporal data one timestep later. The current timestep's
        /// temporal data is left unmodified.
        pub fn shift_temporal(&mut self) {
            self.spatiotemporal
                .copy_within(0..(N_HISTORY_CYCLES - 1), 1);
        }

        /// Fully erases the current timestep's temporal data.
        pub fn clear_temporal(&mut self, time: usize) {
            self.spatiotemporal[0] = [[[0f32; Spatiotemporal::SIZE]; N_WIDTH]; N_HEIGHT];
            self.temporal[0] = [0f32; Temporal::SIZE];
        }

        /// Erases the current timestep's temporal instr data.
        pub fn clear_temporal_instr(&mut self, time: usize) {
            let Spatiotemporal { arm_base_instr, .. } = Spatiotemporal::OFFSETS;
            for y in 0..N_WIDTH {
                for x in 0..N_HEIGHT {
                    let features = &mut self.spatiotemporal[time][y][x];
                    for offset in arm_base_instr.get_offsets() {
                        features[offset] = 0.;
                    }
                }
            }
        }
    }
}

pub use features::Features;
