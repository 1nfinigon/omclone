use std::collections::BTreeMap;

use crate::sim;
use eyre::{ensure, eyre, OptionExt, Result};

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
}

trait ToDenseTensor {
    fn to_dense_tensor(
        &self,
        tracy_client: tracy_client::Client,
        options: (tch::Kind, tch::Device),
    ) -> Result<tch::Tensor>;
}

impl<const N: usize> ToDenseTensor for SparseCoo<N> {
    fn to_dense_tensor(
        &self,
        tracy_client: tracy_client::Client,
        options: (tch::Kind, tch::Device),
    ) -> Result<tch::Tensor> {
        let _span = tracy_client.clone().span(
            tracy_client::span_location!("SparseCoo::to_dense_vector"),
            0,
        );
        let indices = self.indices_tensor()?;
        let values = self.values_tensor()?;
        let tensor = {
            let _span = tracy_client
                .clone()
                .span(tracy_client::span_location!("load sparse"), 0);
            tch::Tensor::f_sparse_coo_tensor_indices_size(
                &indices,
                &values,
                &self.size[..],
                options,
                true,
            )?
        };
        let tensor = {
            let _span = tracy_client
                .clone()
                .span(tracy_client::span_location!("to device"), 0);
            tensor.f_to(options.1)?
        };
        let tensor = {
            let _span = tracy_client
                .clone()
                .span(tracy_client::span_location!("to_dense"), 0);
            tensor.f_to_dense(None, false)?
        };
        Ok(tensor)
    }
}

impl<const N: usize> ToDenseTensor for [&SparseCoo<N>] {
    fn to_dense_tensor(
        &self,
        tracy_client: tracy_client::Client,
        options: (tch::Kind, tch::Device),
    ) -> Result<tch::Tensor> {
        let _span = tracy_client.clone().span(
            tracy_client::span_location!("[SparseCoo]::to_dense_vector"),
            0,
        );

        let batch_size = self.len();
        ensure!(batch_size > 0);

        let size = self
            .iter()
            .map(|sample| sample.size)
            .try_fold(None, |prev_size, this_size| {
                if let Some(prev_size) = prev_size {
                    if prev_size != this_size {
                        return Err(eyre!(
                            "mismatched sizes in SparseCoo batch: {:?} vs {:?}",
                            prev_size,
                            this_size
                        ));
                    }
                }
                Ok(Some(this_size))
            })?
            .ok_or_eyre("SparseCoo batch is empty")?;

        let mut all_indices = Vec::with_capacity(batch_size);
        let mut all_values = Vec::with_capacity(batch_size);

        for (batch_index, sample) in self.into_iter().enumerate() {
            let indices = sample.indices_tensor()?;
            let values = sample.values_tensor()?;

            let (n, num_nonzero) = indices.size2()?;
            assert_eq!(n, N as i64);

            // prepend the batch index, so that indices becomes a tensor of size (n +
            // 1, num_nonzero).
            let indices = tch::Tensor::f_cat(
                &[
                    tch::Tensor::f_full(
                        [1, num_nonzero],
                        batch_index as i64,
                        (tch::Kind::Int64, tch::Device::Cpu),
                    )?,
                    indices,
                ],
                0,
            )?;

            all_indices.push(indices);
            all_values.push(values);
        }

        let all_indices = tch::Tensor::f_cat(&all_indices, 1)?;
        let all_values = tch::Tensor::f_cat(&all_values, 0)?;
        let size_with_batch_dim: Vec<_> = std::iter::once(batch_size as i64).chain(size).collect();

        let tensor = {
            let _span = tracy_client
                .clone()
                .span(tracy_client::span_location!("load sparse"), 0);
            tch::Tensor::f_sparse_coo_tensor_indices_size(
                &all_indices,
                &all_values,
                &size_with_batch_dim,
                options,
                true,
            )?
        };
        let tensor = {
            let _span = tracy_client
                .clone()
                .span(tracy_client::span_location!("to device"), 0);
            tensor.f_to(options.1)?
        };
        let tensor = {
            let _span = tracy_client
                .clone()
                .span(tracy_client::span_location!("to_dense"), 0);
            tensor.f_to_dense(None, false)?
        };
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
    use sim::Rot;

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
                            glyph_orientation.get_onehot_offset(glyph.rot.normalize().to_usize()),
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
                            GlyphType::Track { locs: pos_list } => {
                                assert!(pos_list[i] == *position);
                                if i >= 1 {
                                    let other_pos = positions[i - 1];
                                    let rot = Rot::from_unit_pos(other_pos - position).unwrap();
                                    features
                                        .set(track_minus_dir.get_onehot_offset(rot.to_usize()), 1.);
                                }
                                if i + 1 < positions.len() {
                                    let other_pos = positions[i + 1];
                                    let rot = Rot::from_unit_pos(other_pos - position).unwrap();
                                    features
                                        .set(track_plus_dir.get_onehot_offset(rot.to_usize()), 1.);
                                }
                            }
                            GlyphType::Conduit { .. } => unimplemented!(),
                            GlyphType::Input { pattern, .. } => {
                                assert!(pattern[i].pos == *position);
                                Self::set_atom(
                                    &pattern[i],
                                    &mut features,
                                    input_atom_type,
                                    input_bonds,
                                    None,
                                );
                            }
                            GlyphType::Output {
                                pattern,
                                count: count_left,
                                ..
                            } => {
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
                            GlyphType::OutputRepeating { .. } => unimplemented!(),
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
                    for rel_r in Rot::step_by(arm_type.angles_between_arm()) {
                        let abs_r = rel_r + rot.normalize();
                        features.set(arm_in_orientation[abs_r.to_usize()].get_offset(), 1.);
                        use slotmap::Key;
                        if !atoms_grabbed[rel_r.to_usize()].is_null() {
                            features.set(
                                arm_in_orientation_and_holding_atom[abs_r.to_usize()].get_offset(),
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

pub fn get_best_device() -> eyre::Result<tch::Device> {
    assert_eq!(
        std::env::var_os("CUDA_DEVICE_ORDER"),
        Some("PCI_BUS_ID".into())
    );
    assert_eq!(std::env::var_os("CUDA_VISIBLE_DEVICES"), None);

    if tch::Cuda::is_available() {
        let nvml = nvml_wrapper::Nvml::init()?;

        // This needs to be sorted in ascending key order.
        let mut device_power_by_pci_bus_id = BTreeMap::new();

        for nvml_index in 0..nvml.device_count()? {
            if let Ok(device) = nvml.device_by_index(nvml_index) {
                let pci_bus_id = device.pci_info()?.bus_id;
                let power = device.power_usage()?;
                device_power_by_pci_bus_id.insert(pci_bus_id, power);
            }
        }
        println!(
            "Enumerated {} CUDA devices",
            device_power_by_pci_bus_id.len()
        );
        Ok(tch::Device::Cuda(
            device_power_by_pci_bus_id
                .into_iter()
                .enumerate()
                .min_by_key(|(_, (_, power))| *power)
                .map(|(i, _)| i)
                .unwrap(),
        ))
    } else {
        Ok(tch::Device::Cpu)
    }
}

pub mod model {
    use std::path::Path;

    use super::*;
    use crate::sim::BasicInstr;
    use eyre::{self, OptionExt};
    use tch;

    pub struct Model {
        pub name: String,
        module: tch::CModule,
        device: tch::Device,
        tracy_client: tracy_client::Client,
        dirichlet_distr: rand_distr::Dirichlet<f32>,
    }

    impl Model {
        pub fn load(
            model_path: impl AsRef<Path>,
            device: tch::Device,
            tracy_client: tracy_client::Client,
        ) -> eyre::Result<Self> {
            let name = model_path
                .as_ref()
                .file_stem()
                .unwrap()
                .to_string_lossy()
                .into_owned();

            let mut module = tch::CModule::load_on_device(model_path.as_ref(), device)?;
            module.f_set_eval()?;
            Ok(Self {
                name,
                module,
                device,
                tracy_client,
                dirichlet_distr: rand_distr::Dirichlet::new(&[1.0; BasicInstr::N_TYPES]).unwrap(),
            })
        }

        pub fn load_latest(
            device: tch::Device,
            tracy_client: tracy_client::Client,
        ) -> eyre::Result<Self> {
            // look for the latest model under test/net/mainline
            let latest_model = std::fs::read_dir("test/net/mainline")
                .unwrap()
                .flatten()
                .filter_map(|f| {
                    let ftype = f.file_type().unwrap();
                    if ftype.is_file() {
                        f.path()
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .and_then(|s| s.parse::<usize>().ok())
                            .map(|n| (n, f.path()))
                    } else {
                        None
                    }
                })
                .max_by_key(|(n, _)| *n)
                .map(|(_, path)| path)
                .unwrap();
            println!("Using latest model: {:?}", latest_model);
            Self::load(latest_model, device, tracy_client)
        }
    }

    use crate::search;
    use crate::search_state;
    use rand::prelude::*;

    impl search::BatchEval for Model {
        fn max_batch_size(&self) -> usize {
            128
        }

        fn batch_eval(
            &self,
            states: Vec<(search_state::State, bool)>,
        ) -> Result<Vec<search::EvalResult>> {
            let rng = &mut rand::thread_rng();
            let batch_size = states.len();

            let options = (tch::Kind::Float, self.device);

            let input = {
                let _span = self
                    .tracy_client
                    .clone()
                    .span(tracy_client::span_location!("input munging"), 0);

                let spatial: Vec<_> = states
                    .iter()
                    .map(|(state, _)| &state.nn_features.spatial)
                    .collect();
                let spatial = spatial.to_dense_tensor(self.tracy_client.clone(), options)?;
                let spatiotemporal: Vec<_> = states
                    .iter()
                    .map(|(state, _)| &state.nn_features.spatiotemporal)
                    .collect();
                let spatiotemporal =
                    spatiotemporal.to_dense_tensor(self.tracy_client.clone(), options)?;
                let temporal: Vec<_> = states
                    .iter()
                    .map(|(state, _)| &state.nn_features.temporal)
                    .collect();
                let temporal = temporal.to_dense_tensor(self.tracy_client.clone(), options)?;

                let policy_softmax_temperature: Vec<_> = states
                    .iter()
                    .map(|(_, is_root)| if *is_root { 1.03 } else { 1. })
                    .collect();
                let policy_softmax_temperature =
                    tch::Tensor::f_from_slice(&policy_softmax_temperature)?.f_to(self.device)?;

                assert_eq!(policy_softmax_temperature.size1()?, batch_size as i64);

                [
                    tch::IValue::Tensor(spatial),
                    tch::IValue::Tensor(spatiotemporal),
                    tch::IValue::Tensor(temporal),
                    tch::IValue::Tensor(policy_softmax_temperature),
                ]
            };

            let output = {
                let _span = self
                    .tracy_client
                    .clone()
                    .span(tracy_client::span_location!("exec"), 0);
                tch::no_grad(|| self.module.forward_is(&input))?
            };

            let (policy_tensor, value_tensor): (tch::Tensor, tch::Tensor) = output.try_into()?;

            assert_eq!(
                policy_tensor.size4()?,
                (
                    batch_size as i64,
                    BasicInstr::N_TYPES as i64,
                    constants::N_HEIGHT as i64,
                    constants::N_WIDTH as i64
                )
            );
            assert_eq!(value_tensor.size2()?, (batch_size as i64, 2));

            let _span = self
                .tracy_client
                .clone()
                .span(tracy_client::span_location!("output munging"), 0);

            states
                .iter()
                .enumerate()
                .map(|(batch_index, (state, is_root))| {
                    let (win, mut policy) = {
                        let next_arm_index = state.instr_buffer.len();
                        let next_arm_pos = state.world.arms[next_arm_index].pos;
                        let (x, y) = features::normalize_position(next_arm_pos)
                            .ok_or_eyre("arm out of nn bounds")?;
                        assert!(x < constants::N_WIDTH);
                        assert!(y < constants::N_HEIGHT);

                        let policy = {
                            let mut policy = [0f32; BasicInstr::N_TYPES];

                            let mut policy_sum = 0.;
                            let _span = self
                                .tracy_client
                                .clone()
                                .span(tracy_client::span_location!("policy extraction"), 0);
                            let policy_tensor = policy_tensor
                                .f_select(3, x as i64)?
                                .f_select(2, y as i64)?
                                .f_select(0, batch_index as i64)?
                                .f_to(tch::Device::Cpu)?;
                            for (i_instr, policy_elt) in policy.iter_mut().enumerate() {
                                let this_policy =
                                    policy_tensor.f_double_value(&[i_instr as i64])? as f32;
                                assert!(this_policy >= 0.);
                                *policy_elt = this_policy;
                                policy_sum += this_policy;
                            }
                            assert!((policy_sum - 1.).abs() < 1e-6);
                            policy
                        };

                        let win = {
                            let _span = self
                                .tracy_client
                                .clone()
                                .span(tracy_client::span_location!("value extraction"), 0);
                            let win = value_tensor.f_double_value(&[batch_index as i64, 0])? as f32;
                            assert!(win >= 0.);
                            let loss_by_cycles =
                                value_tensor.f_double_value(&[batch_index as i64, 1])? as f32;
                            assert!(loss_by_cycles >= 0.);
                            let value_sum = win + loss_by_cycles;
                            assert!((value_sum - 1.).abs() < 1e-6);
                            win
                        };

                        (win, policy)
                    };

                    if *is_root {
                        // add Dirichlet noise
                        // TODO: for this problem, we should stretch out the
                        // Dirichlet noise to more than just the root.
                        // TODO: filter to only valid moves
                        let noise = self.dirichlet_distr.sample(rng);
                        for (policy, noise) in policy.iter_mut().zip(noise) {
                            const EPS: f32 = 0.25;
                            *policy = (1. - EPS) * *policy + EPS * noise;
                        }
                    }

                    Ok(search::EvalResult {
                        utility: win,
                        policy,
                    })
                })
                .collect()
        }
    }
}

pub use model::Model;
