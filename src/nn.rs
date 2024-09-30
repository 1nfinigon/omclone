use crate::sim;
use eyre::Result;

pub mod constants {

    pub const N_WIDTH: usize = 16;
    pub const N_HEIGHT: usize = 16;
    pub const N_HISTORY_CYCLES: usize = 4;
    pub const N_MAX_PRODUCTS: usize = 7;

    pub const N_ORIENTATIONS: usize = 6;

    pub const N_BOND_TYPES: usize = 4;
    pub const N_ATOM_TYPES: usize = 17;
}

use constants::*;

pub mod feature_offsets {
    use super::constants::*;
    use crate::sim::BasicInstr;

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
        pub glyph_triplex_bond: OneHot<3>,
        pub glyph_multi_bond: OneHot<4>,
        pub glyph_disposal: Binary,

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
        pub arm_base_instr: OneHot<{ BasicInstr::N_TYPES }>,

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
        pub cycles_remaining: Float,
    }

    impl Temporal {
        const fn new_and_size() -> (Self, usize) {
            let offset = 0usize;
            let (offset, cycles) = Float::assign(offset);
            let (offset, cycles_remaining) = Float::assign(offset);
            let self_ = Self {
                cycles,
                cycles_remaining,
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
}

trait SparseCooAsSlice {
    fn set_slice(&mut self, coord: &[usize], value: f32);
}

#[derive(Clone, Debug)]
pub struct SparseCoo<const N: usize> {
    /// flattened representation; indices.len() == nonzero_count * N
    indices: Vec<i64>,
    values: Vec<f32>,
    size: [i64; N],
}

impl<const N: usize> SparseCoo<N> {
    fn new(size: [usize; N]) -> Self {
        let size: Vec<_> = size.iter().map(|s| *s as i64).collect();
        Self {
            indices: Vec::new(),
            values: Vec::new(),
            size: size.try_into().unwrap(),
        }
    }

    fn set(&mut self, coords: &[usize; N], value: f32) {
        for (&coord, &size) in coords.iter().zip(self.size.iter()) {
            assert!(coord < size as usize);
            self.indices.push(coord.try_into().unwrap());
        }
        self.values.push(value);
    }

    fn retain_coords_mut<F: FnMut(&mut [i64]) -> bool>(&mut self, mut f: F) {
        let mut new_i = 0;
        for old_i in 0..self.values.len() {
            if f(&mut self.indices[old_i * N..(old_i + 1) * N]) {
                if old_i != new_i {
                    self.indices
                        .copy_within(old_i * N..(old_i + 1) * N, new_i * N);
                    self.values[new_i] = self.values[old_i];
                }
                new_i += 1;
            }
        }
        self.indices.truncate(new_i * N);
        self.values.truncate(new_i);
    }

    fn slice_all_but_one_dim(&mut self, dim: usize, other_coords: &[usize]) -> SparseCoo1DSlice {
        assert_eq!(other_coords.len(), N - 1);
        SparseCoo1DSlice {
            underlying: self,
            dim,
            other_coords: other_coords.to_owned(),
        }
    }

    fn indices_tensor(&self) -> Result<tch::Tensor> {
        Ok(tch::Tensor::f_from_slice(&self.indices[..])?
            .f_view([-1, N as i64])?
            .f_t_()?)
    }

    fn values_tensor(&self) -> Result<tch::Tensor> {
        Ok(tch::Tensor::f_from_slice(&self.values[..])?)
    }

    fn size_tensor(&self) -> Result<tch::Tensor> {
        Ok(tch::Tensor::f_from_slice(&self.size[..])?)
    }

    pub fn to_dense_tensor(&self, options: (tch::Kind, tch::Device)) -> Result<tch::Tensor> {
        let indices = self.indices_tensor()?;
        let values = self.values_tensor()?;
        let tensor = tch::Tensor::f_sparse_coo_tensor_indices_size(
            &indices,
            &values,
            &self.size[..],
            options,
            true,
        )?
        .f_to_dense(None, false)?
        .f_to(options.1)?;
        Ok(tensor)
    }
}

pub struct SparseCooTensorsForSerializing {
    pub indices: tch::Tensor,
    pub values: tch::Tensor,
    pub size: tch::Tensor,
}

impl<const N: usize> SparseCoo<N> {
    pub fn to_tensors_for_serializing(&self) -> Result<SparseCooTensorsForSerializing> {
        Ok(SparseCooTensorsForSerializing {
            indices: self.indices_tensor()?,
            values: self.values_tensor()?,
            size: self.size_tensor()?,
        })
    }
}

impl<const N: usize> SparseCooAsSlice for SparseCoo<N> {
    fn set_slice(&mut self, coord: &[usize], value: f32) {
        self.set(coord.try_into().unwrap(), value)
    }
}

struct SparseCoo1DSlice<'a> {
    underlying: &'a mut dyn SparseCooAsSlice,
    dim: usize,
    other_coords: Vec<usize>,
}

impl<'a> SparseCoo1DSlice<'a> {
    fn set(&mut self, this_coord: usize, value: f32) {
        let n_dims = self.other_coords.len() + 1;
        let mut coord = vec![0usize; n_dims];
        for (i, coord_elt) in coord.iter_mut().enumerate() {
            *coord_elt = match i.cmp(&self.dim) {
                std::cmp::Ordering::Less => self.other_coords[i],
                std::cmp::Ordering::Equal => this_coord,
                std::cmp::Ordering::Greater => self.other_coords[i - 1],
            };
        }
        self.underlying.set_slice(&coord[..], value);
    }
}

pub mod features {
    use super::feature_offsets::*;
    use super::*;

    pub fn normalize_position(pos: sim::Pos) -> Option<(usize, usize)> {
        if 0 <= pos.x && (pos.x as usize) < N_WIDTH && 0 <= pos.y && (pos.y as usize) < N_HEIGHT {
            Some((pos.x as usize, pos.y as usize))
        } else {
            None
        }
    }

    #[derive(Clone)]
    pub struct Features {
        pub spatial: SparseCoo<3>,
        pub spatiotemporal: SparseCoo<4>,
        pub temporal: SparseCoo<2>,
    }

    impl Features {
        pub fn new() -> Self {
            Self {
                spatial: SparseCoo::new([Spatial::SIZE, N_HEIGHT, N_WIDTH]),
                spatiotemporal: SparseCoo::new([
                    Spatiotemporal::SIZE,
                    N_HISTORY_CYCLES,
                    N_HEIGHT,
                    N_WIDTH,
                ]),
                temporal: SparseCoo::new([Temporal::SIZE, N_HISTORY_CYCLES]),
            }
        }

        fn get_spatial_mut(&mut self, pos: sim::Pos) -> Option<SparseCoo1DSlice> {
            normalize_position(pos).map(|(x, y)| self.spatial.slice_all_but_one_dim(0, &[y, x]))
        }

        fn get_spatiotemporal_mut(
            &mut self,
            time: usize,
            pos: sim::Pos,
        ) -> Option<SparseCoo1DSlice> {
            normalize_position(pos)
                .map(|(x, y)| self.spatiotemporal.slice_all_but_one_dim(0, &[time, y, x]))
        }

        /// Helper function for setting features relating to an Atom.
        fn set_atom(
            atom: &sim::Atom,
            features: &mut SparseCoo1DSlice,
            offset_atom_type: OneHot<N_ATOM_TYPES>,
            offset_atom_bonds: [[Binary; N_BOND_TYPES]; N_ORIENTATIONS],
            offset_atom_is_berlo: Option<Binary>,
        ) {
            let sim::Atom {
                pos: _,
                atom_type,
                connections,
                is_berlo,
            } = atom;
            use num_traits::ToPrimitive;
            features.set(
                offset_atom_type.get_onehot_offset(atom_type.to_usize().unwrap()),
                1.,
            );
            for rot in 0..6 {
                if connections[rot].intersects(sim::Bonds::NORMAL) {
                    features.set(offset_atom_bonds[rot][0].get_offset(), 1.);
                }
                if connections[rot].intersects(sim::Bonds::TRIPLEX_R) {
                    features.set(offset_atom_bonds[rot][1].get_offset(), 1.);
                }
                if connections[rot].intersects(sim::Bonds::TRIPLEX_K) {
                    features.set(offset_atom_bonds[rot][2].get_offset(), 1.);
                }
                if connections[rot].intersects(sim::Bonds::TRIPLEX_Y) {
                    features.set(offset_atom_bonds[rot][3].get_offset(), 1.);
                }
            }
            match offset_atom_is_berlo {
                Some(offset_atom_is_berlo) => {
                    if *is_berlo {
                        features.set(offset_atom_is_berlo.get_offset(), 1.);
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
                    if let Some(mut features) = self.get_spatial_mut(*position) {
                        features.set(
                            glyph_orientation
                                .get_onehot_offset(sim::normalize_dir(glyph.rot) as usize),
                            1.,
                        );
                        use sim::GlyphType;
                        match &glyph.glyph_type {
                            GlyphType::Calcification => {
                                features.set(glyph_calcification.get_onehot_offset(i), 1.)
                            }
                            GlyphType::Animismus => {
                                features.set(glyph_animismus.get_onehot_offset(i), 1.)
                            }
                            GlyphType::Projection => {
                                features.set(glyph_projection.get_onehot_offset(i), 1.)
                            }
                            GlyphType::Dispersion => {
                                features.set(glyph_dispersion.get_onehot_offset(i), 1.)
                            }
                            GlyphType::Purification => {
                                features.set(glyph_purification.get_onehot_offset(i), 1.)
                            }
                            GlyphType::Duplication => {
                                features.set(glyph_duplication.get_onehot_offset(i), 1.)
                            }
                            GlyphType::Unification => {
                                features.set(glyph_unification.get_onehot_offset(i), 1.)
                            }
                            GlyphType::Bonding => {
                                features.set(glyph_bonding.get_onehot_offset(i), 1.)
                            }
                            GlyphType::Unbonding => {
                                features.set(glyph_unbonding.get_onehot_offset(i), 1.)
                            }
                            GlyphType::TriplexBond => {
                                features.set(glyph_triplex_bond.get_onehot_offset(i), 1.)
                            }
                            GlyphType::MultiBond => {
                                features.set(glyph_multi_bond.get_onehot_offset(i), 1.)
                            }
                            GlyphType::Disposal => {
                                if i == 0 {
                                    features.set(glyph_disposal.get_offset(), 1.)
                                }
                            }
                            GlyphType::Equilibrium => (),
                            GlyphType::Track(pos_list) => {
                                assert!(pos_list[i] == *position);
                                if i >= 1 {
                                    let other_pos = positions[i - 1];
                                    let rot = sim::pos_to_rot(other_pos - position).unwrap();
                                    features
                                        .set(track_minus_dir.get_onehot_offset(rot as usize), 1.);
                                }
                                if i + 1 < positions.len() {
                                    let other_pos = positions[i + 1];
                                    let rot = sim::pos_to_rot(other_pos - position).unwrap();
                                    features
                                        .set(track_plus_dir.get_onehot_offset(rot as usize), 1.);
                                }
                            }
                            GlyphType::Conduit(_, _) => unimplemented!(),
                            GlyphType::Input(pattern, _id) => {
                                assert!(pattern[i].pos == *position);
                                Self::set_atom(
                                    &pattern[i],
                                    &mut features,
                                    input_atom_type,
                                    input_bonds,
                                    None,
                                );
                            }
                            GlyphType::Output(pattern, count_left, _id) => {
                                assert!(pattern[i].pos == *position);
                                Self::set_atom(
                                    &pattern[i],
                                    &mut features,
                                    output_atom_type,
                                    output_bonds,
                                    None,
                                );
                                if *count_left > 0 {
                                    features.set(
                                        output_count_minus_one.get_onehot_offset(
                                            (*count_left as usize).min(N_MAX_PRODUCTS) - 1,
                                        ),
                                        1.,
                                    );
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
        pub fn set_temporal_except_instr(
            &mut self,
            time: usize,
            world: &sim::World,
            cycles_remaining_value: u64,
        ) {
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
                if let Some(mut features) = self.get_spatiotemporal_mut(time, *position) {
                    features.set(used.get_offset(), 1.);
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
                if let Some(mut features) = self.get_spatiotemporal_mut(time, *pos) {
                    features.set(
                        arm_base_length.get_onehot_offset((len - 1).try_into().unwrap()),
                        1.,
                    );
                    if *arm_type == sim::ArmType::Piston {
                        features.set(arm_base_is_piston.get_offset(), 1.);
                    }
                    if *grabbing {
                        features.set(arm_base_is_grabbing.get_offset(), 1.);
                    }
                    if *arm_type == sim::ArmType::VanBerlo {
                        features.set(arm_base_is_berlo.get_offset(), 1.);
                    }
                    for rel_r in (0..6).step_by(arm_type.angles_between_arm() as usize) {
                        let abs_r = sim::normalize_dir(rel_r + rot);
                        features.set(arm_in_orientation[abs_r as usize].get_offset(), 1.);
                        use slotmap::Key;
                        if !atoms_grabbed[rel_r as usize].is_null() {
                            features.set(
                                arm_in_orientation_and_holding_atom[abs_r as usize].get_offset(),
                                1.,
                            );
                        }
                    }
                }
            }

            for atom in world.atoms.atom_map.values() {
                let pos = atom.pos;
                if let Some(mut features) = self.get_spatiotemporal_mut(time, pos) {
                    Self::set_atom(
                        atom,
                        &mut features,
                        atom_type,
                        atom_bonds,
                        Some(atom_is_berlo),
                    );
                }
            }

            let Temporal {
                cycles,
                cycles_remaining,
            } = Temporal::OFFSETS;
            self.temporal.set(
                &[cycles.get_offset(), time],
                (world.timestep as f64 / 100.).tanh() as f32,
            );
            self.temporal.set(
                &[cycles_remaining.get_offset(), time],
                (cycles_remaining_value as f64 / 100.).tanh() as f32,
            );
        }

        /// Set all timesteps' data to this world. Used for initialization.
        pub fn init_all_temporal(&mut self, world: &sim::World, cycles_remaining_value: u64) {
            // TODO: minor optim, could do just time and then do a blind copy for all other ts
            for t in 0..N_HISTORY_CYCLES {
                self.set_temporal_except_instr(t, world, cycles_remaining_value);
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
            if let Some(mut features) = self.get_spatiotemporal_mut(time, pos) {
                use num_traits::ToPrimitive;
                features.set(
                    arm_base_instr.get_onehot_offset(instr.to_usize().unwrap()),
                    1.,
                );
            }
        }

        /// Copies all temporal data one timestep later. The current timestep's
        /// temporal data is cleared.
        pub fn shift_temporal(&mut self) {
            self.spatiotemporal.retain_coords_mut(|c| {
                if c[1] + 1 >= N_HISTORY_CYCLES as i64 {
                    false
                } else {
                    c[1] += 1;
                    true
                }
            });
            self.temporal.retain_coords_mut(|c| {
                if c[1] + 1 >= N_HISTORY_CYCLES as i64 {
                    false
                } else {
                    c[1] += 1;
                    true
                }
            });
        }
    }
}

pub use features::Features;

pub mod model {
    use super::*;
    use crate::sim::BasicInstr;
    use eyre;
    use tch;

    const MODEL_FILENAME: &str = "pytorch/model.pt";

    pub struct Model {
        module: tch::CModule,
        device: tch::Device,
    }

    pub struct Evaluation {
        /// Softmaxed from policy head
        pub policy: [f32; BasicInstr::N_TYPES],

        /// Softmaxed outcome prediction from value head
        pub win: f32,
        //pub loss_by_cycles: f32,
        //loss_by_area: f32,

        //cycles_left: f32, -- think about making this N(0, 1) in the raw output. Maybe interpret this as scaled relative to the given cycle limit?
        //area_left: f32,
    }

    impl Model {
        pub fn load() -> eyre::Result<Self> {
            let device = tch::Device::cuda_if_available();
            let mut module = tch::CModule::load_on_device(MODEL_FILENAME, device)?;
            module.f_set_eval()?;
            Ok(Self { module, device })
        }

        pub fn forward(
            &self,
            features: &Features,
            x: usize,
            y: usize,
            is_root: bool,
        ) -> eyre::Result<Evaluation> {
            assert!(x < constants::N_WIDTH);
            assert!(y < constants::N_HEIGHT);

            let options = (tch::Kind::Float, self.device);

            let input = [
                tch::IValue::Tensor(features.spatial.to_dense_tensor(options)?.f_unsqueeze(0)?),
                tch::IValue::Tensor(
                    features
                        .spatiotemporal
                        .to_dense_tensor(options)?
                        .f_unsqueeze(0)?,
                ),
                tch::IValue::Tensor(features.temporal.to_dense_tensor(options)?.f_unsqueeze(0)?),
                tch::IValue::Tensor(
                    tch::Tensor::f_from_slice(&[if is_root { 1.03 } else { 1. }])?
                        .f_to(self.device)?,
                ),
            ];

            let output = tch::no_grad(|| self.module.forward_is(&input))?;

            let (policy_tensor, value_tensor): (tch::Tensor, tch::Tensor) = output.try_into()?;

            let mut policy = [0f32; BasicInstr::N_TYPES];

            assert_eq!(
                policy_tensor.size4()?,
                (
                    1,
                    BasicInstr::N_TYPES as i64,
                    constants::N_HEIGHT as i64,
                    constants::N_WIDTH as i64
                )
            );
            let mut policy_sum = 0.;
            for (i_instr, policy_elt) in policy.iter_mut().enumerate() {
                let this_policy =
                    policy_tensor.f_double_value(&[0, i_instr as i64, y as i64, x as i64])? as f32;
                assert!(this_policy >= 0.);
                *policy_elt = this_policy;
                policy_sum += this_policy;
            }
            assert!((policy_sum - 1.).abs() < 1e-6);

            assert_eq!(value_tensor.size2()?, (1, 2));
            let win = value_tensor.f_double_value(&[0, 0])? as f32;
            assert!(win >= 0.);
            let loss_by_cycles = value_tensor.f_double_value(&[0, 1])? as f32;
            assert!(loss_by_cycles >= 0.);
            let value_sum = win + loss_by_cycles;
            assert!((value_sum - 1.).abs() < 1e-6);

            Ok(Evaluation {
                policy,
                win,
                //loss_by_cycles,
            })
        }
    }
}

pub use model::Model;
