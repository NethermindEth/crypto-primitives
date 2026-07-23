//! This module defines **semirings** - sets equipped with addition and
//! multiplication.
//!
//! Even though subtraction is not required (additive inverses are not
//! guaranteed to exist), we include subtraction in the definition for
//! convenience.
//!
//! See the [crate-level documentation](crate) for the overall element/config
//! structure.
//!
//! This module additionally defines several specialized flavors of semirings
//! for working with integers.

#[cfg(feature = "ark_ff")]
pub mod ark_ff_bigint;
pub mod boolean;
#[cfg(feature = "crypto_bigint")]
pub mod crypto_bigint_boxed_uint;
#[cfg(feature = "crypto_bigint")]
pub mod crypto_bigint_uint;

use crate::{
    FixedConfig, ParseStrConfig, SetConfig, SetElement,
    helpers::{define_blanket_trait, delegate_to_ref_binary},
};
use core::{
    fmt::Display,
    hash::Hash,
    iter::{Product, Sum},
    ops::{
        Add, AddAssign, Div, DivAssign, Mul, MulAssign, Rem, RemAssign, Shl, ShlAssign, Shr,
        ShrAssign, Sub, SubAssign,
    },
    str::FromStr,
};
use num_traits::{
    Bounded, CheckedAdd, CheckedDiv, CheckedMul, CheckedRem, CheckedSub, ConstOne, ConstZero, One,
    Pow, Zero,
};
use pastey::paste;

define_blanket_trait! {
    /// See [module-level documentation](crate::semiring).
    pub trait Semiring:
        SetElement
        // Core traits
        + Sized
        + Display
        + Hash
        + Default
        // Arithmetic operations consuming rhs
        + CheckedAdd
        + CheckedSub
        + CheckedMul
        + AddAssign
        + SubAssign
        + MulAssign
        + Pow<u32, Output=Self>
        // Arithmetic operations with rhs reference
        + for<'a> Add<&'a Self, Output=Self>
        + for<'a> Sub<&'a Self, Output=Self>
        + for<'a> Mul<&'a Self, Output=Self>
        + for<'a> AddAssign<&'a Self>
        + for<'a> SubAssign<&'a Self>
        + for<'a> MulAssign<&'a Self>
        // Aggregate operations
        + Sum
        + Product
        + for<'a> Sum<&'a Self>
        + for<'a> Product<&'a Self>
        // Other
        + Zero
        + One
        + From<bool>
}

define_blanket_trait! {
    /// [`Semiring`] with a bunch of values known at compile time.
    pub trait ConstSemiring: Semiring + ConstZero + ConstOne
}

/// Semiring of integers.
pub trait IntSemiring: Semiring + Ord {
    fn is_odd(&self) -> bool;

    fn is_even(&self) -> bool;
}

define_blanket_trait! {
    /// [`IntSemiring`] that defines (integer) division and remainder operations.
    pub trait IntSemiringWithDivRem:
        IntSemiring
        + CheckedDiv
        + CheckedRem
        + for<'a> Div<&'a Self, Output = Self>
        + for<'a> Rem<&'a Self, Output = Self>
        + DivAssign
        + RemAssign
        + for<'a> DivAssign<&'a Self>
        + for<'a> RemAssign<&'a Self>
}

define_blanket_trait! {
    pub trait IntSemiringWithShifts:
        IntSemiring + Shl<u32> + Shr<u32> + ShlAssign<u32> + ShrAssign<u32>
}

define_blanket_trait! {
    pub trait ConstIntSemiring: IntSemiring + ConstSemiring + Bounded + FromStr
}

macro_rules! primitive_int_semiring {
    ($t:ident) => {
        impl IntSemiring for $t {
            fn is_odd(&self) -> bool {
                *self & 1 == 1
            }

            fn is_even(&self) -> bool {
                *self & 1 == 0
            }
        }
    };
}

primitive_int_semiring!(i8);
primitive_int_semiring!(i16);
primitive_int_semiring!(i32);
primitive_int_semiring!(i64);
primitive_int_semiring!(i128);
primitive_int_semiring!(u8);
primitive_int_semiring!(u16);
primitive_int_semiring!(u32);
primitive_int_semiring!(u64);
primitive_int_semiring!(u128);

/// See [module-level documentation](crate::semiring).
pub trait SemiringConfig: SetConfig {
    fn is_zero(&self, value: &Self::Element) -> bool;

    fn zero(&self) -> Self::Element;

    fn one(&self) -> Self::Element;

    //
    // Operations on refs
    //

    /// x + y
    fn add(&self, x: &Self::Element, y: &Self::Element) -> Self::Element;

    /// x - y
    fn sub(&self, x: &Self::Element, y: &Self::Element) -> Self::Element;

    /// x * y
    fn mul(&self, x: &Self::Element, y: &Self::Element) -> Self::Element;

    /// x ** y
    fn pow_u32(&self, x: &Self::Element, y: u32) -> Self::Element;

    //
    // Checked operations on refs
    //

    /// x + y
    fn checked_add(&self, x: &Self::Element, y: &Self::Element) -> Option<Self::Element>;

    /// x - y
    fn checked_sub(&self, x: &Self::Element, y: &Self::Element) -> Option<Self::Element>;

    /// x * y
    fn checked_mul(&self, x: &Self::Element, y: &Self::Element) -> Option<Self::Element>;

    /// x ** y
    fn checked_pow_u32(&self, x: &Self::Element, y: u32) -> Option<Self::Element>;

    //
    // Operations on mutable refs
    //

    delegate_to_ref_binary! {
        /// x += y
        add
    }

    delegate_to_ref_binary! {
        /// x -= y
        sub
    }

    delegate_to_ref_binary! {
        /// x *= y
        mul
    }

    delegate_to_ref_binary! {
        /// x **= y
        pow_u32(u32)
    }

    //
    // Aggregate operations
    //

    fn sum<I: Iterator<Item = Self::Element>>(&self, mut iter: I) -> Self::Element {
        let mut acc = iter.next().unwrap_or(self.zero());
        for x in iter {
            self.add_assign(&mut acc, &x)
        }
        acc
    }

    fn sum_refs<'a, I: Iterator<Item = &'a Self::Element> + 'a>(
        &self,
        mut iter: I,
    ) -> Self::Element {
        let mut acc = iter.next().cloned().unwrap_or(self.zero());
        for x in iter {
            self.add_assign(&mut acc, x)
        }
        acc
    }

    fn product<I: Iterator<Item = Self::Element>>(&self, mut iter: I) -> Self::Element {
        let mut acc = iter.next().unwrap_or(self.one());
        for x in iter {
            self.mul_assign(&mut acc, &x)
        }
        acc
    }

    fn product_refs<'a, I: Iterator<Item = &'a Self::Element>>(
        &self,
        mut iter: I,
    ) -> Self::Element {
        let mut acc = iter.next().cloned().unwrap_or(self.one());
        for x in iter {
            self.mul_assign(&mut acc, x)
        }
        acc
    }
}

pub trait IntSemiringConfig: SemiringConfig {
    fn is_odd(&self, x: &Self::Element) -> bool;

    fn is_even(&self, x: &Self::Element) -> bool;
}

pub trait ConstIntSemiringConfig: IntSemiringConfig + ParseStrConfig {
    fn min(&self) -> Self::Element;

    fn max(&self) -> Self::Element;
}

// Delegating to the element's own (potentially overflowing) operators is the
// whole point of this bridge.
#[allow(clippy::arithmetic_side_effects)]
impl<S: Semiring> SemiringConfig for FixedConfig<S> {
    #[inline(always)]
    fn is_zero(&self, value: &Self::Element) -> bool {
        value.is_zero()
    }

    #[inline(always)]
    fn zero(&self) -> Self::Element {
        S::zero()
    }

    #[inline(always)]
    fn one(&self) -> Self::Element {
        S::one()
    }

    #[inline(always)]
    fn add(&self, x: &Self::Element, y: &Self::Element) -> Self::Element {
        x.clone() + y
    }

    #[inline(always)]
    fn sub(&self, x: &Self::Element, y: &Self::Element) -> Self::Element {
        x.clone() - y
    }

    #[inline(always)]
    fn mul(&self, x: &Self::Element, y: &Self::Element) -> Self::Element {
        x.clone() * y
    }

    #[inline(always)]
    fn pow_u32(&self, x: &Self::Element, y: u32) -> Self::Element {
        x.clone().pow(y)
    }

    fn checked_add(&self, x: &Self::Element, y: &Self::Element) -> Option<Self::Element> {
        x.checked_add(y)
    }

    fn checked_sub(&self, x: &Self::Element, y: &Self::Element) -> Option<Self::Element> {
        x.checked_sub(y)
    }

    fn checked_mul(&self, x: &Self::Element, y: &Self::Element) -> Option<Self::Element> {
        x.checked_mul(y)
    }

    fn checked_pow_u32(&self, x: &Self::Element, y: u32) -> Option<Self::Element> {
        num_traits::checked_pow(x.clone(), usize::try_from(y).ok()?)
    }

    fn add_assign(&self, x: &mut Self::Element, y: &Self::Element) {
        *x += y
    }

    fn sub_assign(&self, x: &mut Self::Element, y: &Self::Element) {
        *x -= y;
    }

    fn mul_assign(&self, x: &mut Self::Element, y: &Self::Element) {
        *x *= y
    }
}

impl<S: SetElement + FromStr> ParseStrConfig for FixedConfig<S> {
    fn parse_str(&self, str: &str) -> Option<Self::Element> {
        str.parse().ok()
    }
}

impl<S: IntSemiring> IntSemiringConfig for FixedConfig<S> {
    fn is_odd(&self, x: &Self::Element) -> bool {
        IntSemiring::is_odd(x)
    }

    fn is_even(&self, x: &Self::Element) -> bool {
        IntSemiring::is_even(x)
    }
}

impl<S: ConstIntSemiring> ConstIntSemiringConfig for FixedConfig<S> {
    fn min(&self) -> Self::Element {
        S::min_value()
    }

    fn max(&self) -> Self::Element {
        S::max_value()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ensure_type_implements_trait;

    #[test]
    fn ensure_traits() {
        ensure_type_implements_trait!(u8, ConstIntSemiring);
        ensure_type_implements_trait!(i128, ConstIntSemiring);

        ensure_type_implements_trait!(FixedConfig<u8>, ConstIntSemiringConfig);
        ensure_type_implements_trait!(FixedConfig<i128>, ConstIntSemiringConfig);
    }
}
