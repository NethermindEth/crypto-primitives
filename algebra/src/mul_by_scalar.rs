#[cfg(feature = "crypto_bigint")]
use crate::from_ref::FromRef;
#[cfg(feature = "crypto_bigint")]
use crate::ring::crypto_bigint_int::Int;
use num_traits::CheckedMul;

pub trait MulByScalar<Rhs>: Sized {
    /// Multiplies the current element by a scalar from the right (usually - a
    /// coefficient to obtain a linear combination).
    /// Returns `None` if the multiplication would overflow.
    fn mul_by_scalar(&self, rhs: Rhs) -> Option<Self>;
}

macro_rules! impl_mul_by_scalar_for_primitives {
    ($($t:ty),*) => {
        $(
            impl MulByScalar<&$t> for $t {
                fn mul_by_scalar(&self, rhs: &$t) -> Option<Self> {
                    self.checked_mul(rhs)
                }
            }
        )*
    };
}

impl_mul_by_scalar_for_primitives!(i8, i16, i32, i64, i128);

#[cfg(feature = "crypto_bigint")]
impl<const LIMBS: usize, const LIMBS2: usize> MulByScalar<&Int<LIMBS2>> for Int<LIMBS> {
    fn mul_by_scalar(&self, rhs: &Int<LIMBS2>) -> Option<Self> {
        if LIMBS < LIMBS2 {
            return None; // Cannot multiply if the left operand has fewer limbs than the right
        }
        self.checked_mul(&rhs.resize())
    }
}

#[cfg(feature = "crypto_bigint")]
macro_rules! impl_mul_int_by_primitive_scalar {
    ($($t:ty),*) => {
        $(
            impl<const LIMBS: usize> MulByScalar<&$t> for Int<LIMBS> {
                fn mul_by_scalar(&self, rhs: &$t) -> Option<Self> {
                    self.checked_mul(&Self::from_ref(rhs))
                }
            }
        )*
    };
}

#[cfg(feature = "crypto_bigint")]
impl_mul_int_by_primitive_scalar!(i8, i16, i32, i64, i128);
