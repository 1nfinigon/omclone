
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct NonNan(pub f32);

impl NonNan {
    pub fn new(val: f32) -> Option<NonNan> {
        if val.is_nan() {
            None
        } else {
            Some(NonNan(val))
        }
    }
}

impl Eq for NonNan {}

#[allow(clippy::derive_ord_xor_partial_ord)]
impl Ord for NonNan {
    fn cmp(&self, other: &NonNan) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}
