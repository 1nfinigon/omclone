use crate::sim;
use omnn_feature::*;

const N_WIDTH: usize = 16;
const N_HEIGHT: usize = 16;
const N_ARMS: usize = 12;
const N_TRACKS: usize = 6;
const N_INPUTS: usize = 4;
const N_OUTPUTS: usize = 2;
const N_HISTORY_CYCLES: usize = 2;
const N_MAX_CYCLES: usize = 500;
const N_MAX_PRODUCTS: usize = 6;

/*
const N_WIDTH: usize = 32;
const N_HEIGHT: usize = 32;
const N_ARMS: usize = 16;
const N_TRACKS: usize = 8;
const N_INPUTS: usize = 5;
const N_OUTPUTS: usize = 3;
const N_HISTORY_CYCLES: usize = 4;
const N_MAX_CYCLES: usize = 1000;
const N_MAX_PRODUCTS: usize = 6;
*/

const N_MAX_AREA: usize = N_WIDTH * N_HEIGHT;
const N_ORIENTATIONS: usize = 6;

#[derive(Copy, Clone, OmnnFeature)]
struct ArmGen<const L1: usize, const L2: usize, const L3: usize> {
    length_1: OneHot<L1>,
    length_2: OneHot<L2>,
    length_3: OneHot<L3>,
}

#[derive(Copy, Clone, OmnnFeature)]
struct Arm {
    exists: Binary,
    grabbing: Binary,
    holding_atom: Binary,
    plain_arm: [ArmGen<2, 3, 4>; 6],
    double_arm: [ArmGen<3, 5, 7>; 3],
    triple_arm: [ArmGen<4, 7, 10>; 2],
    hex_arm: [ArmGen<7, 13, 19>; 1],
    piston: [ArmGen<2, 3, 4>; 6],
    van_berlo: [OneHot<7>; 6],
}

#[derive(Copy, Clone, OmnnFeature)]
struct Glyph<const N: usize>([OneHot<N_ORIENTATIONS>; N]);

#[derive(Copy, Clone, OmnnFeature)]
struct Glyphs {
    calcification: Binary,
    animismus: Glyph<4>,
    projection: Glyph<2>,
    purification: Glyph<3>,
    duplication: Glyph<2>,
    unification: Glyph<5>,
    dispersion: Glyph<5>,
    bonding: Glyph<2>,
    unbonding: Glyph<2>,
    triplex_bond: Glyph<2>,
    multi_bond: Glyph<2>,
    disposal: Glyph<7>,
}

#[derive(Copy, Clone, OmnnFeature)]
struct Track {
    plus_dir: OneHot<N_ORIENTATIONS>,
    minus_dir: OneHot<N_ORIENTATIONS>,
    arm_is_on_track: OneHot<N_ARMS>,
}

#[derive(Copy, Clone, OmnnFeature)]
struct Bond {
    normal: Binary,
    triplex_r: Binary,
    triplex_k: Binary,
    triplex_y: Binary,
}

impl OmnnFeatureWrite<sim::Bonds> for Bond {
    fn write_internal(&self, data: sim::Bonds, offset: &mut u64, output: &mut Vec<u64>) {
        let Self {
            normal,
            triplex_r,
            triplex_k,
            triplex_y,
        } = self;
        normal.write(data.intersects(sim::Bonds::NORMAL), offset, output);
        triplex_r.write(data.intersects(sim::Bonds::TRIPLEX_R), offset, output);
        triplex_k.write(data.intersects(sim::Bonds::TRIPLEX_K), offset, output);
        triplex_y.write(data.intersects(sim::Bonds::TRIPLEX_Y), offset, output);
    }
}

#[derive(Copy, Clone, OmnnFeature)]
struct AtomType(OneHot<17>);

impl OmnnFeatureWrite<sim::AtomType> for AtomType {
    fn write_internal(&self, data: sim::AtomType, offset: &mut u64, output: &mut Vec<u64>) {
        use num_traits::ToPrimitive;
        self.0.write(data.to_usize().unwrap(), offset, output);
    }
}

#[derive(Copy, Clone, OmnnFeature)]
struct Atom {
    exists: Binary,
    grabbed: Binary,
    molecule_grabbed: Binary,
    atom_type: AtomType,
    bonds: [Bond; N_ORIENTATIONS],
}

#[derive(Copy, Clone, OmnnFeature)]
struct Input {
    atom_type: AtomType,
    bonds: [Bond; N_ORIENTATIONS],
}

#[derive(Copy, Clone, OmnnFeature)]
struct Output {
    atom_type: AtomType,
    bonds: [Bond; N_ORIENTATIONS],
}

#[derive(Copy, Clone, OmnnFeature)]
struct SpatiotemporalState {
    arms: [Arm; N_ARMS],
    atom: Atom,
}

#[derive(Copy, Clone, OmnnFeature)]
struct SpatialState {
    glyphs: [Glyphs; N_ORIENTATIONS],
    tracks: [Track; N_TRACKS],
    inputs: [Input; N_INPUTS],
    outputs: [Output; N_OUTPUTS],
}

#[derive(Copy, Clone, OmnnFeature)]
struct TemporalState {
    products_left: OneHot<N_MAX_PRODUCTS>,
}

#[derive(Copy, Clone, OmnnFeature)]
struct GlobalState {
    cycles_left: OneHot<N_MAX_CYCLES>,
    area_left: OneHot<N_MAX_AREA>,
}

#[derive(Copy, Clone, OmnnFeature)]
struct State {
    spatial: [[SpatialState; N_WIDTH]; N_HEIGHT],
    spatiotemporal: [[[SpatiotemporalState; N_HEIGHT]; N_WIDTH]; N_HISTORY_CYCLES],
    temporal: [TemporalState; N_HISTORY_CYCLES],
    global: GlobalState,
}
