pub use omnn_feature_derive::OmnnFeature;

pub trait OmnnFeature: Clone + Copy {
    fn new() -> Self;
    fn size(&self) -> usize;
}

pub trait OmnnFeatureWrite<Data>: OmnnFeature {
    fn write_internal(&self, data: Data, offset: &mut u64, output: &mut Vec<u64>);
    #[inline(always)]
    fn write(&self, data: Data, offset: &mut u64, output: &mut Vec<u64>) {
        let old_offset = *offset;
        self.write_internal(data, offset, output);
        let new_offset = *offset;
        assert_eq!((new_offset - old_offset) as usize, self.size());
    }
}

#[derive(Copy, Clone)]
pub struct Binary {}

impl OmnnFeature for Binary {
    #[inline(always)]
    fn new() -> Self { Self {} }
    #[inline(always)]
    fn size(&self) -> usize {
        1
    }
}

impl OmnnFeatureWrite<bool> for Binary {
    #[inline(always)]
    fn write_internal(&self, data: bool, offset: &mut u64, output: &mut Vec<u64>) {
        if data {
            output.push(*offset);
        }
        *offset += 1;
    }
}

impl<T: OmnnFeature, const N: usize> OmnnFeature for [T; N]
{
    #[inline(always)]
    fn new() -> Self { [T::new(); N] }
    #[inline(always)]
    fn size(&self) -> usize {
        if N == 0 {
            0
        } else {
            self[0].size() * N
        }
    }
}

#[derive(Copy, Clone)]
pub struct OneHot<const N: usize> {}

impl<const N: usize> OmnnFeature for OneHot<N> {
    #[inline(always)]
    fn new() -> Self { Self {} }
    #[inline(always)]
    fn size(&self) -> usize {
        N
    }
}

impl<const N: usize> OmnnFeatureWrite<usize> for OneHot<N> {
    #[inline(always)]
    fn write_internal(&self, data: usize, offset: &mut u64, output: &mut Vec<u64>) {
        assert!(data < N);
        output.push(*offset + data as u64);
        *offset += self.size() as u64;
    }
}

impl<Data, T: OmnnFeatureWrite<Data>> OmnnFeatureWrite<Option<Data>> for T {
    #[inline(always)]
    fn write_internal(&self, data: Option<Data>, offset: &mut u64, output: &mut Vec<u64>) {
        if let Some(data) = data {
            self.write(data, offset, output);
        } else {
            *offset += self.size() as u64;
        }
    }
}
