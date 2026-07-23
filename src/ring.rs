//! This module defines **rings** - [`semirings`](crate::semiring) where every
//! element has an additive inverse (and hence negation is defined).
//!
//! See the [crate-level documentation](crate) for the overall element/config
//! structure.
//!
//! This module additionally defines several specialized flavors of rings for
//! working with integers.

#[cfg(feature = "crypto_bigint")]
pub mod crypto_bigint_int;

use crate::{
    ConstIntSemiring, ConstIntSemiringConfig, ConstSemiring, FixedConfig, IntSemiring,
    IntSemiringConfig, IntSemiringWithDivRem, IntSemiringWithShifts, Semiring, SemiringConfig,
    SetElement, helpers::define_blanket_trait,
};
use core::ops::Neg;
use num_traits::{CheckedNeg, Signed};

define_blanket_trait! {
    /// See [module-level documentation](crate::ring).
    pub trait Ring: SetElement + Semiring + Neg<Output = Self> + CheckedNeg
}

define_blanket_trait! {
    /// [`Ring`] with a bunch of values known at compile time.
    pub trait ConstRing: Ring + ConstSemiring
}

define_blanket_trait! {
    /// Ring of integers, usually denoted as `Z`.
    pub trait IntRing: Ring + IntSemiringWithDivRem + Signed
}

define_blanket_trait! {
    pub trait IntRingWithShifts: IntRing + IntSemiringWithShifts
}

define_blanket_trait! {
    pub trait ConstIntRing: IntRing + ConstIntSemiring + From<i8>
}

/// See [module-level documentation](crate::ring).
pub trait RingConfig: SemiringConfig {
    /// -x
    fn neg(&self, x: &Self::Element) -> Self::Element;

    /// -x;
    fn checked_neg(&self, x: &Self::Element) -> Option<Self::Element>;

    /// x = -x;
    fn neg_assign(&self, x: &mut Self::Element) {
        *x = self.neg(x);
    }
}

define_blanket_trait! {
    pub trait IntRingConfig: RingConfig + IntSemiringConfig
}

pub trait ConstIntRingConfig: IntRingConfig + ConstIntSemiringConfig {
    /// |x|
    fn abs(&self, x: &Self::Element) -> Self::Element;

    /// Checked absolute value. Returns `None` if the result cannot be
    /// represented.
    fn checked_abs(&self, x: &Self::Element) -> Option<Self::Element>;

    fn signum(&self, x: &Self::Element) -> Self::Element;

    fn is_positive(&self, x: &Self::Element) -> bool;

    fn is_negative(&self, x: &Self::Element) -> bool;
}

impl<R: Ring> RingConfig for FixedConfig<R> {
    #[inline(always)]
    fn neg(&self, x: &Self::Element) -> Self::Element {
        x.clone().neg()
    }

    fn checked_neg(&self, x: &Self::Element) -> Option<Self::Element> {
        x.checked_neg()
    }
}

impl<R: ConstIntRing> ConstIntRingConfig for FixedConfig<R> {
    fn abs(&self, x: &Self::Element) -> Self::Element {
        x.abs()
    }

    fn checked_abs(&self, x: &Self::Element) -> Option<Self::Element> {
        if *x == R::min_value() {
            None
        } else {
            Some(x.abs())
        }
    }

    fn signum(&self, x: &Self::Element) -> Self::Element {
        x.signum()
    }

    fn is_positive(&self, x: &Self::Element) -> bool {
        x.is_positive()
    }

    fn is_negative(&self, x: &Self::Element) -> bool {
        x.is_negative()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ensure_type_implements_trait;

    #[test]
    fn ensure_traits() {
        ensure_type_implements_trait!(i8, ConstIntRing);
        ensure_type_implements_trait!(i128, ConstIntRing);

        ensure_type_implements_trait!(FixedConfig<i8>, ConstIntRingConfig);
        ensure_type_implements_trait!(FixedConfig<i128>, ConstIntRingConfig);
    }
}
