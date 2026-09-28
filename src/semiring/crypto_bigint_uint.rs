use super::*;
use crate::{
    Wrapper, boolean::Boolean, crypto_bigint_int::Int, helpers::pow_via_repeated_squaring,
};
use core::{
    cmp::Ordering,
    fmt::{Debug, Display, Formatter, LowerHex, Result as FmtResult, UpperHex},
    hash::{Hash, Hasher},
    iter::{Product, Sum},
    ops::{
        Add, AddAssign, Div, Mul, MulAssign, Rem, RemAssign, Shl, ShlAssign, Shr, ShrAssign, Sub,
        SubAssign,
    },
    str::FromStr,
};
use crypto_bigint::{DivVartime, Integer, Limb, UintRef, Word};
use num_traits::{
    CheckedAdd, CheckedMul, CheckedRem, CheckedSub, ConstOne, ConstZero, FromPrimitive, One, Pow,
    ToPrimitive, WrappingAdd, WrappingMul, WrappingSub, Zero, float::FloatCore,
};
use pastey::paste;
#[cfg(feature = "rand")]
use rand::{distr::StandardUniform, prelude::*, rand_core::TryRng};
#[cfg(feature = "zerocopy")]
use zerocopy_derive::*;

#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "zerocopy", derive(KnownLayout))]
#[repr(transparent)]
pub struct Uint<const LIMBS: usize>(pub crypto_bigint::Uint<LIMBS>);

impl<const LIMBS: usize> Uint<LIMBS> {
    pub const MAX: Self = Self(crypto_bigint::Uint::MAX);

    /// Wraps a given value into this wrapper type
    #[inline(always)]
    pub const fn new(value: crypto_bigint::Uint<LIMBS>) -> Self {
        Self(value)
    }

    #[inline(always)]
    pub const fn new_ref(value: &crypto_bigint::Uint<LIMBS>) -> &Self {
        // Safety: Uint<LIMBS> is #[repr(transparent)] and is guaranteed to have the
        // same memory layout as crypto_bigint::Uint
        unsafe { &*(value as *const crypto_bigint::Uint<LIMBS> as *const Self) }
    }

    #[inline(always)]
    pub const fn new_ref_mut(value: &mut crypto_bigint::Uint<LIMBS>) -> &mut Self {
        // Safety: Uint<LIMBS> is #[repr(transparent)] and is guaranteed to have the
        // same memory layout as crypto_bigint::Uint
        unsafe { &mut *(value as *mut crypto_bigint::Uint<LIMBS> as *mut Self) }
    }

    /// See [crypto_bigint::Uint::from_words]
    #[inline(always)]
    pub const fn from_words(arr: [Word; LIMBS]) -> Self {
        Self(crypto_bigint::Uint::from_words(arr))
    }

    /// See [crypto_bigint::Uint::to_words]
    #[inline]
    pub const fn to_words(self) -> [Word; LIMBS] {
        self.0.to_words()
    }

    /// See [crypto_bigint::Uint::as_words]
    pub const fn as_words(&self) -> &[Word; LIMBS] {
        self.0.as_words()
    }

    /// See [crypto_bigint::Uint::as_mut_words]
    pub const fn as_mut_words(&mut self) -> &mut [Word; LIMBS] {
        self.0.as_mut_words()
    }

    /// See [crypto_bigint::Uint::as_limbs]
    pub const fn as_limbs(&self) -> &[Limb; LIMBS] {
        self.0.as_limbs()
    }

    /// See [crypto_bigint::Uint::as_mut_limbs]
    pub const fn as_mut_limbs(&mut self) -> &mut [Limb; LIMBS] {
        self.0.as_mut_limbs()
    }

    /// See [crypto_bigint::Uint::to_limbs]
    pub const fn to_limbs(self) -> [Limb; LIMBS] {
        self.0.to_limbs()
    }

    /// See [crypto_bigint::Uint::resize]
    #[inline(always)]
    pub const fn resize<const T: usize>(&self) -> Uint<T> {
        Uint::<T>(self.0.resize())
    }

    pub const fn checked_resize<const T: usize>(&self) -> Option<Uint<T>> {
        match checked_resize::<LIMBS, T>(&self.0) {
            None => None,
            Some(inner) => Some(Uint(inner)),
        }
    }

    /// See [crypto_bigint::Uint::as_int]
    pub const fn as_int(&self) -> &Int<LIMBS> {
        Int::new_ref(self.0.as_int())
    }

    /// See [crypto_bigint::Uint::from_be_hex]
    pub const fn from_be_hex(hex: &str) -> Self {
        Self(crypto_bigint::Uint::<LIMBS>::from_be_hex(hex))
    }

    /// See [crypto_bigint::Uint::from_le_hex]
    pub const fn from_le_hex(hex: &str) -> Self {
        Self(crypto_bigint::Uint::<LIMBS>::from_le_hex(hex))
    }

    /// Returns $(m, e)$ with $\mathsf{self} \approx m \cdot 2^e$, where $m$ is the top 64 bits of
    /// `self` rounded to odd: its LSB is set if any truncated bit is set, so that rounding $m$
    /// to a float with round-to-nearest-even is correct. Mirrors `num_bigint`'s
    /// `high_bits_to_u64`.
    fn float_mantissa_and_exponent(&self) -> (u64, u32) {
        let Some(exponent) = self.0.bits_vartime().checked_sub(u64::BITS) else {
            return (self.0.resize::<WORD_FACTOR>().into(), 0);
        };
        // exponent < bits <= BITS, so shifts don't panic
        let high = self.0.shr_vartime(exponent);
        let sticky = u64::from(high.shl_vartime(exponent) != self.0);
        let mantissa = u64::from(high.resize::<WORD_FACTOR>()) | sticky;
        (mantissa, exponent)
    }
}

const fn checked_resize<const SRC: usize, const DST: usize>(
    num: &crypto_bigint::Uint<SRC>,
) -> Option<crypto_bigint::Uint<DST>> {
    // Compile-time check: widening or same-size resize is infallible
    if const { SRC > DST } {
        let max = Uint::<DST>::MAX.0.resize();
        let cmp = num.cmp_vartime(&max);
        if cmp.is_gt() {
            return None;
        }
    }
    Some(num.resize())
}

impl<const LIMBS: usize> Uint<LIMBS> {
    /// Total size of the represented integer in bits.
    pub const BITS: u32 = crypto_bigint::Uint::<LIMBS>::BITS;
    /// Total size of the represented integer in bytes.
    pub const BYTES: usize = crypto_bigint::Uint::<LIMBS>::BYTES;
    /// The number of limbs used on this platform.
    pub const LIMBS: usize = LIMBS;
}

//
// Core traits
//

impl<const LIMBS: usize> AsRef<UintRef> for Uint<LIMBS> {
    #[inline(always)]
    fn as_ref(&self) -> &UintRef {
        self.inner().as_ref()
    }
}

impl<const LIMBS: usize> AsMut<UintRef> for Uint<LIMBS> {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut UintRef {
        self.inner_mut().as_mut()
    }
}

impl<const LIMBS: usize> Debug for Uint<LIMBS> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        Debug::fmt(&self.0, f)
    }
}

impl<const LIMBS: usize> Display for Uint<LIMBS> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        Display::fmt(&self.0, f)
    }
}

impl<const LIMBS: usize> Default for Uint<LIMBS> {
    #[inline(always)]
    fn default() -> Self {
        Self::ZERO
    }
}

impl<const LIMBS: usize> PartialOrd for Uint<LIMBS> {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Implemented manually to use `cmp_vartime`
impl<const LIMBS: usize> Ord for Uint<LIMBS> {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp_vartime(&other.0)
    }
}

impl<const LIMBS: usize> LowerHex for Uint<LIMBS> {
    #[inline(always)]
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        LowerHex::fmt(&self.0, f)
    }
}

impl<const LIMBS: usize> UpperHex for Uint<LIMBS> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        UpperHex::fmt(&self.0, f)
    }
}

impl<const LIMBS: usize> Hash for Uint<LIMBS> {
    #[inline(always)]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state)
    }
}

impl<const LIMBS: usize> FromStr for Uint<LIMBS> {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (radix, s) = if let Some(s) = s.strip_prefix("0x") {
            (16, s)
        } else {
            (10, s)
        };
        let uint =
            crypto_bigint::Uint::<LIMBS>::from_str_radix_vartime(s, radix).map_err(|_| ())?;
        Ok(Self(uint))
    }
}

//
// Zero and One traits
//

impl<const LIMBS: usize> Zero for Uint<LIMBS> {
    #[inline(always)]
    fn zero() -> Self {
        Self::ZERO
    }

    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.0.is_zero().to_bool_vartime()
    }
}

impl<const LIMBS: usize> One for Uint<LIMBS> {
    #[inline(always)]
    fn one() -> Self {
        Self::ONE
    }
}

impl<const LIMBS: usize> ConstZero for Uint<LIMBS> {
    const ZERO: Self = Self(crypto_bigint::Uint::ZERO);
}

impl<const LIMBS: usize> ConstOne for Uint<LIMBS> {
    const ONE: Self = Self(crypto_bigint::Uint::ONE);
}

//
// Basic arithmetic operations
//

macro_rules! impl_basic_and_wrapping_op {
    ($trait_name:tt, $trait_op:tt) => {
        paste! {
            impl<const LIMBS: usize> $trait_name for Uint<LIMBS> {
                type Output = Self;

                #[inline(always)]
                fn $trait_op(self, rhs: Self) -> Self::Output {
                    self.$trait_op(&rhs)
                }
            }

            impl<'a, const LIMBS: usize> $trait_name<&'a Self> for Uint<LIMBS> {
                type Output = Self;

                #[inline(always)]
                fn $trait_op(self, rhs: &'a Self) -> Self::Output {
                    if cfg!(debug_assertions) {
                        // In debug mode
                        Self(self.0.$trait_op(&rhs.0))
                    } else {
                        // In release mode, wrap around silently
                        self.[<wrapping_ $trait_op>](rhs)
                    }
                }
            }

            impl<const LIMBS: usize> [<Wrapping $trait_name>] for Uint<LIMBS> {
                #[inline(always)]
                fn [<wrapping_ $trait_op>](&self, rhs: &Self) -> Self {
                    Self(self.0.[<wrapping_ $trait_op>](&rhs.0))
                }
            }
        }
    };
}

impl_basic_and_wrapping_op!(Add, add);
impl_basic_and_wrapping_op!(Sub, sub);
impl_basic_and_wrapping_op!(Mul, mul);

impl<const LIMBS: usize> Div for Uint<LIMBS> {
    type Output = Self;

    #[inline(always)]
    fn div(self, rhs: Self) -> Self::Output {
        self.div(&rhs)
    }
}

impl<'a, const LIMBS: usize> Div<&'a Self> for Uint<LIMBS> {
    type Output = Self;

    fn div(self, rhs: &'a Self) -> Self::Output {
        let non_zero = crypto_bigint::NonZero::new(rhs.0).expect("division by zero");
        Self(self.0.div_vartime(&non_zero))
    }
}

impl<const LIMBS: usize> Rem for Uint<LIMBS> {
    type Output = Self;

    #[inline(always)]
    fn rem(self, rhs: Self) -> Self::Output {
        self.rem(&rhs)
    }
}

impl<'a, const LIMBS: usize> Rem<&'a Self> for Uint<LIMBS> {
    type Output = Self;

    fn rem(self, rhs: &'a Self) -> Self::Output {
        let non_zero = crypto_bigint::NonZero::new(rhs.0).expect("division by zero");
        Self(self.0.rem_vartime(&non_zero))
    }
}

impl<const LIMBS: usize> Shl<u32> for Uint<LIMBS> {
    type Output = Self;

    #[inline(always)]
    fn shl(self, rhs: u32) -> Self::Output {
        Self(self.0.shl(rhs))
    }
}

impl<const LIMBS: usize> Shr<u32> for Uint<LIMBS> {
    type Output = Self;

    #[inline(always)]
    fn shr(self, rhs: u32) -> Self::Output {
        Self(self.0.shr(rhs))
    }
}

impl<const LIMBS: usize> Pow<u32> for Uint<LIMBS> {
    type Output = Self;

    fn pow(self, rhs: u32) -> Self::Output {
        pow_via_repeated_squaring!(self, rhs, Self::ONE)
    }
}

//
// Checked arithmetic operations
//

impl<const LIMBS: usize> CheckedAdd for Uint<LIMBS> {
    fn checked_add(&self, other: &Self) -> Option<Self> {
        let (result, overflow) = self.0.carrying_add(&other.0, crypto_bigint::Limb::ZERO);
        if overflow.0 != 0 {
            None
        } else {
            Some(Self(result))
        }
    }
}

impl<const LIMBS: usize> CheckedSub for Uint<LIMBS> {
    fn checked_sub(&self, other: &Self) -> Option<Self> {
        let (result, borrow) = self.0.borrowing_sub(&other.0, crypto_bigint::Limb::ZERO);
        if borrow.0 != 0 {
            None
        } else {
            Some(Self(result))
        }
    }
}

impl<const LIMBS: usize> CheckedMul for Uint<LIMBS> {
    fn checked_mul(&self, other: &Self) -> Option<Self> {
        // Use widening_mul which returns (lo, hi)
        let (lo, hi) = self.0.widening_mul(&other.0);
        if hi.is_zero().to_bool_vartime() {
            Some(Self(lo))
        } else {
            None
        }
    }
}

impl<const LIMBS: usize> CheckedRem for Uint<LIMBS> {
    fn checked_rem(&self, other: &Self) -> Option<Self> {
        let non_zero = crypto_bigint::NonZero::new(other.0).into_option()?;
        Some(Self(self.0.rem(&non_zero)))
    }
}

//
// Arithmetic assign operations
//

macro_rules! impl_assign_op {
    ($trait_name:tt, $trait_op:tt) => {
        impl<const LIMBS: usize> $trait_name<Self> for Uint<LIMBS> {
            #[inline(always)]
            fn $trait_op(&mut self, rhs: Self) {
                self.$trait_op(&rhs);
            }
        }

        impl<'a, const LIMBS: usize> $trait_name<&'a Self> for Uint<LIMBS> {
            #[inline(always)]
            fn $trait_op(&mut self, rhs: &'a Self) {
                self.0.$trait_op(&rhs.0);
            }
        }
    };
}

impl_assign_op!(AddAssign, add_assign);
impl_assign_op!(SubAssign, sub_assign);
impl_assign_op!(MulAssign, mul_assign);

impl<const LIMBS: usize> RemAssign for Uint<LIMBS> {
    #[inline(always)]
    fn rem_assign(&mut self, rhs: Self) {
        self.rem_assign(&rhs);
    }
}

impl<'a, const LIMBS: usize> RemAssign<&'a Self> for Uint<LIMBS> {
    #![allow(clippy::arithmetic_side_effects)]
    fn rem_assign(&mut self, rhs: &'a Self) {
        let non_zero = crypto_bigint::NonZero::new(rhs.0).expect("division by zero");
        self.0 %= non_zero;
    }
}

impl<const LIMBS: usize> ShlAssign<u32> for Uint<LIMBS> {
    #[inline(always)]
    fn shl_assign(&mut self, rhs: u32) {
        self.0.shl_assign(rhs);
    }
}

impl<const LIMBS: usize> ShrAssign<u32> for Uint<LIMBS> {
    #[inline(always)]
    fn shr_assign(&mut self, rhs: u32) {
        self.0.shr_assign(rhs);
    }
}

//
// Aggregate operations
//

impl<const LIMBS: usize> Sum for Uint<LIMBS> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| {
            acc.checked_add(&x).expect("overflow in sum")
        })
    }
}

impl<'a, const LIMBS: usize> Sum<&'a Self> for Uint<LIMBS> {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| {
            acc.checked_add(x).expect("overflow in sum")
        })
    }
}

impl<const LIMBS: usize> Product for Uint<LIMBS> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| {
            acc.checked_mul(&x).expect("overflow in product")
        })
    }
}

impl<'a, const LIMBS: usize> Product<&'a Self> for Uint<LIMBS> {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| {
            acc.checked_mul(x).expect("overflow in product")
        })
    }
}

//
// Conversions
//

impl<const LIMBS: usize> From<crypto_bigint::Uint<LIMBS>> for Uint<LIMBS> {
    #[inline(always)]
    fn from(value: crypto_bigint::Uint<LIMBS>) -> Self {
        Self(value)
    }
}

impl<const LIMBS: usize> From<Uint<LIMBS>> for crypto_bigint::Uint<LIMBS> {
    #[inline(always)]
    fn from(value: Uint<LIMBS>) -> Self {
        value.0
    }
}

impl<'a, const LIMBS: usize> From<&'a crypto_bigint::Uint<LIMBS>> for &'a Uint<LIMBS> {
    #[inline(always)]
    fn from(value: &'a crypto_bigint::Uint<LIMBS>) -> Self {
        Uint::new_ref(value)
    }
}

impl<'a, const LIMBS: usize> From<&'a Uint<LIMBS>> for &'a crypto_bigint::Uint<LIMBS> {
    #[inline(always)]
    fn from(value: &'a Uint<LIMBS>) -> Self {
        &value.0
    }
}

impl<const LIMBS: usize> From<bool> for Uint<LIMBS> {
    #[inline(always)]
    fn from(value: bool) -> Self {
        Self(crypto_bigint::Uint::<LIMBS>::from(u8::from(value)))
    }
}

impl<const LIMBS: usize> From<Boolean> for Uint<LIMBS> {
    #[inline(always)]
    fn from(value: Boolean) -> Self {
        Self::from(value.into_inner())
    }
}

macro_rules! impl_from_primitive {
    ($($t:ty),+) => {
        $(
            impl<const LIMBS: usize> From<$t> for Uint<LIMBS> {
                fn from(value: $t) -> Self {
                    assert!(core::mem::size_of::<$t>() <= crypto_bigint::Uint::<LIMBS>::BYTES,
                            "`{}` is too large to fit into `Uint<{LIMBS}>`", stringify!($t));
                    Self(crypto_bigint::Uint::<LIMBS>::from(value))
                }
            }

            impl<'a, const LIMBS: usize> From<&'a $t> for Uint<LIMBS> {
                #[inline(always)]
                fn from(value: &$t) -> Self {
                    Self::from(*value)
                }
            }

            impl<const LIMBS: usize> Uint<LIMBS> {
            paste! {
                /// Create a Uint from a primitive type.
                pub const fn [<from_ $t>](n: $t) -> Self {
                    Self(crypto_bigint::Uint::<LIMBS>::[<from_ $t>](n))
                }
            }
            }
        )+
    };
}

impl_from_primitive!(u8, u16, u32, u64, u128);

impl<const LIMBS: usize, const LIMBS2: usize> TryFrom<&crypto_bigint::Uint<LIMBS2>>
    for Uint<LIMBS>
{
    type Error = ();

    fn try_from(num: &crypto_bigint::Uint<LIMBS2>) -> Result<Self, Self::Error> {
        checked_resize(num).map(Self).ok_or(())
    }
}

impl<const LIMBS: usize> ToPrimitive for Uint<LIMBS> {
    #[inline]
    fn to_i64(&self) -> Option<i64> {
        self.to_u128()?.to_i64()
    }

    #[inline]
    fn to_i128(&self) -> Option<i128> {
        self.to_u128()?.to_i128()
    }

    #[inline]
    fn to_u64(&self) -> Option<u64> {
        self.to_u128()?.to_u64()
    }

    #[inline]
    fn to_u128(&self) -> Option<u128> {
        let value: U128 = self.checked_resize()?;
        Some(value.0.into())
    }

    #[allow(clippy::cast_precision_loss)] // Rounding is intended
    #[inline]
    fn to_f32(&self) -> Option<f32> {
        let (mantissa, exponent) = self.float_mantissa_and_exponent();
        match i32::try_from(exponent) {
            Ok(exponent) if exponent <= f32::MAX_EXP => {
                Some(mantissa as f32 * FloatCore::powi(2.0_f32, exponent))
            }
            _ => Some(f32::INFINITY),
        }
    }

    #[allow(clippy::cast_precision_loss)] // Rounding is intended
    #[inline]
    fn to_f64(&self) -> Option<f64> {
        let (mantissa, exponent) = self.float_mantissa_and_exponent();
        match i32::try_from(exponent) {
            Ok(exponent) if exponent <= f64::MAX_EXP => {
                Some(mantissa as f64 * FloatCore::powi(2.0_f64, exponent))
            }
            _ => Some(f64::INFINITY),
        }
    }
}

// Inherent `from_u*` shadow the trait methods, so values are built via `U64`/`U128`
impl<const LIMBS: usize> FromPrimitive for Uint<LIMBS> {
    #[inline]
    fn from_i64(n: i64) -> Option<Self> {
        U64::from_u64(u64::try_from(n).ok()?).checked_resize()
    }

    #[inline]
    fn from_i128(n: i128) -> Option<Self> {
        U128::from_u128(u128::try_from(n).ok()?).checked_resize()
    }

    #[inline]
    fn from_u64(n: u64) -> Option<Self> {
        U64::from_u64(n).checked_resize()
    }

    #[inline]
    fn from_u128(n: u128) -> Option<Self> {
        U128::from_u128(n).checked_resize()
    }

    #[inline]
    fn from_f64(n: f64) -> Option<Self> {
        if !n.is_finite() {
            return None;
        }
        // Truncate toward zero, matching `as` casts and `num_bigint`
        let n = FloatCore::trunc(n);
        if n.is_zero() {
            return Some(Self::ZERO);
        }
        let (mantissa, exponent, sign) = FloatCore::integer_decode(n);
        if sign < 0 {
            return None;
        }
        let Ok(exponent) = u32::try_from(exponent) else {
            // `n` is an integer, so the shifted-out bits are zero
            return U64::from_u64(mantissa >> exponent.unsigned_abs()).checked_resize();
        };
        let mantissa: Self = U64::from_u64(mantissa).checked_resize()?;
        if mantissa.0.bits_vartime().checked_add(exponent)? > Self::BITS {
            return None;
        }
        // exponent < BITS, so the shift doesn't panic
        Some(Self(mantissa.0.shl_vartime(exponent)))
    }
}

//
// Wrapper
//

impl<const LIMBS: usize> Wrapper for Uint<LIMBS> {
    type Inner = crypto_bigint::Uint<LIMBS>;

    #[inline(always)]
    fn inner(&self) -> &Self::Inner {
        &self.0
    }

    #[inline(always)]
    fn inner_mut(&mut self) -> &mut Self::Inner {
        &mut self.0
    }

    #[inline(always)]
    fn into_inner(self) -> Self::Inner {
        self.0
    }

    #[inline(always)]
    fn new_unchecked(inner: Self::Inner) -> Self {
        Self(inner)
    }
}

//
// Semiring
//

impl<const LIMBS: usize> Bounded for Uint<LIMBS> {
    #[inline(always)]
    fn min_value() -> Self {
        Self::ZERO
    }

    #[inline(always)]
    fn max_value() -> Self {
        Self::MAX
    }
}

impl<const LIMBS: usize> IntSemiring for Uint<LIMBS> {
    #[inline(always)]
    fn is_odd(&self) -> bool {
        self.0.is_odd().into()
    }

    #[inline(always)]
    fn is_even(&self) -> bool {
        self.0.is_even().into()
    }
}

//
// RNG
//

#[cfg(feature = "rand")]
impl<const LIMBS: usize> Distribution<Uint<LIMBS>> for StandardUniform {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Uint<LIMBS> {
        crypto_bigint::Random::random_from_rng(rng)
    }
}

#[cfg(feature = "rand")]
impl<const LIMBS: usize> crypto_bigint::Random for Uint<LIMBS> {
    fn try_random_from_rng<R: TryRng + ?Sized>(rng: &mut R) -> Result<Self, R::Error> {
        crypto_bigint::Uint::try_random_from_rng(rng).map(Self)
    }
}

//
// Serialization and Deserialization
//

#[cfg(feature = "serde")]
impl<'de, const LIMBS: usize> serde::Deserialize<'de> for Uint<LIMBS>
where
    crypto_bigint::Uint<LIMBS>: crypto_bigint::Encoding,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        crypto_bigint::Uint::<LIMBS>::deserialize(deserializer).map(Self)
    }
}

#[cfg(feature = "serde")]
impl<const LIMBS: usize> serde::Serialize for Uint<LIMBS>
where
    crypto_bigint::Uint<LIMBS>: crypto_bigint::Encoding,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.0.serialize(serializer)
    }
}

//
// Zeroize
//

#[cfg(feature = "zeroize")]
impl<const LIMBS: usize> zeroize::DefaultIsZeroes for Uint<LIMBS> {}

//
// Traits from crypto_bigint
//

impl<const LIMBS: usize> crypto_bigint::CtEq for Uint<LIMBS> {
    #[inline]
    fn ct_eq(&self, other: &Self) -> crypto_bigint::Choice {
        crypto_bigint::CtEq::ct_eq(&self.0, &other.0)
    }
}

impl<const LIMBS: usize> crypto_bigint::CtGt for Uint<LIMBS> {
    #[inline]
    fn ct_gt(&self, other: &Self) -> crypto_bigint::Choice {
        crypto_bigint::CtGt::ct_gt(&self.0, &other.0)
    }
}

impl<const LIMBS: usize> crypto_bigint::CtLt for Uint<LIMBS> {
    #[inline]
    fn ct_lt(&self, other: &Self) -> crypto_bigint::Choice {
        crypto_bigint::CtLt::ct_lt(&self.0, &other.0)
    }
}

impl<const LIMBS: usize> crypto_bigint::CtSelect for Uint<LIMBS> {
    fn ct_select(&self, other: &Self, choice: crypto_bigint::Choice) -> Self {
        crypto_bigint::CtSelect::ct_select(&self.0, &other.0, choice).into()
    }
}

impl<const LIMBS: usize> crypto_bigint::Bounded for Uint<LIMBS> {
    const BITS: u32 = Self::BITS;
    const BYTES: usize = Self::BYTES;
}

impl<const LIMBS: usize> crypto_bigint::Constants for Uint<LIMBS> {
    const MAX: Self = Self::MAX;
}

//
// Predefined uints of various sizes for convenience
//

use crate::helpers::crypto_bigint::WORD_FACTOR;
pub type U64 = Uint<{ WORD_FACTOR }>;
pub type U128 = Uint<{ 2 * WORD_FACTOR }>;
pub type U192 = Uint<{ 3 * WORD_FACTOR }>;
pub type U256 = Uint<{ 4 * WORD_FACTOR }>;
pub type U320 = Uint<{ 5 * WORD_FACTOR }>;
pub type U384 = Uint<{ 6 * WORD_FACTOR }>;
pub type U448 = Uint<{ 7 * WORD_FACTOR }>;
pub type U512 = Uint<{ 8 * WORD_FACTOR }>;
pub type U576 = Uint<{ 9 * WORD_FACTOR }>;
pub type U640 = Uint<{ 10 * WORD_FACTOR }>;
pub type U704 = Uint<{ 11 * WORD_FACTOR }>;
pub type U768 = Uint<{ 12 * WORD_FACTOR }>;
pub type U832 = Uint<{ 13 * WORD_FACTOR }>;
pub type U896 = Uint<{ 14 * WORD_FACTOR }>;
pub type U960 = Uint<{ 15 * WORD_FACTOR }>;
pub type U1024 = Uint<{ 16 * WORD_FACTOR }>;
pub type U1280 = Uint<{ 20 * WORD_FACTOR }>;
pub type U1536 = Uint<{ 24 * WORD_FACTOR }>;
pub type U1792 = Uint<{ 28 * WORD_FACTOR }>;
pub type U2048 = Uint<{ 32 * WORD_FACTOR }>;
pub type U3072 = Uint<{ 48 * WORD_FACTOR }>;
pub type U3584 = Uint<{ 56 * WORD_FACTOR }>;
pub type U4096 = Uint<{ 64 * WORD_FACTOR }>;
pub type U4224 = Uint<{ 66 * WORD_FACTOR }>;
pub type U4352 = Uint<{ 68 * WORD_FACTOR }>;
pub type U6144 = Uint<{ 96 * WORD_FACTOR }>;
pub type U8192 = Uint<{ 128 * WORD_FACTOR }>;
pub type U16384 = Uint<{ 256 * WORD_FACTOR }>;
pub type U32768 = Uint<{ 512 * WORD_FACTOR }>;

#[allow(clippy::arithmetic_side_effects, clippy::cast_lossless)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::ensure_type_implements_trait;
    use alloc::{format, string::ToString, vec::Vec};

    #[cfg(target_pointer_width = "64")]
    const WORD_FACTOR: usize = 1;
    #[cfg(target_pointer_width = "32")]
    const WORD_FACTOR: usize = 2;

    type Uint1 = Uint<WORD_FACTOR>;
    type Uint2 = Uint<{ WORD_FACTOR * 2 }>;
    type Uint4 = Uint<{ WORD_FACTOR * 4 }>;

    #[test]
    fn ensure_traits() {
        ensure_type_implements_trait!(Uint4, Wrapper);
        ensure_type_implements_trait!(Uint4, ConstIntSemiring);
        ensure_type_implements_trait!(Uint4, IntSemiringWithShifts);
    }

    #[test]
    fn basic_operations() {
        let a = Uint4::from(10_u64);
        let b = Uint4::from(5_u64);

        // Test addition
        assert_eq!(a + b, Uint4::from(15_u64));

        // Test subtraction
        assert_eq!(a - b, Uint4::from(5_u64));

        // Test multiplication
        assert_eq!(a * b, Uint4::from(50_u64));

        // Test remainder
        assert_eq!(a % b, Uint4::ZERO);

        // Test shl
        let x = Uint1::from(0x0001_u64);
        assert_eq!(x << 0, x);
        assert_eq!(x << 1, 0x0002_u64.into());
        assert_eq!(x << 15, 0x8000_u64.into());

        // Test shr
        let x = Uint4::from(0x8000_u32);
        assert_eq!(x >> 0, x);
        assert_eq!(x >> 1, 0x4000_u64.into());
        assert_eq!(x >> 15, 0x0001_u64.into());
    }

    #[test]
    #[should_panic(expected = "`shift` exceeds upper bound")]
    fn shl_panics_on_overflow() {
        let x = Uint1::from(0x0001_u64);
        let _ = x << 64;
    }

    #[test]
    fn checked_operations() {
        let a = Uint4::from(10_u64);
        let b = Uint4::from(5_u64);

        assert_eq!(a.checked_add(&b), Some(Uint4::from(15_u64)));
        assert_eq!(a.checked_sub(&b), Some(Uint4::from(5_u64)));
        assert_eq!(a.checked_mul(&b), Some(Uint4::from(50_u64)));
        assert_eq!(a.checked_rem(&b), Some(Uint4::ZERO));

        // Test underflow
        assert!(b.checked_sub(&a).is_none());

        // MIN and MAX
        assert_eq!(Uint4::MAX.checked_add(&One::one()), None);
        assert_eq!(Uint4::ZERO.checked_sub(&One::one()), None);
    }

    #[allow(clippy::op_ref)]
    #[test]
    fn reference_operations() {
        let a = Uint4::from(10_u64);
        let b = Uint4::from(5_u64);

        // Test reference-based addition
        let c = a + &b;
        assert_eq!(c, Uint4::from(15_u64));

        // Test reference-based subtraction
        let d = a - &b;
        assert_eq!(d, Uint4::from(5_u64));

        // Test reference-based multiplication
        let e = a * &b;
        assert_eq!(e, Uint4::from(50_u64));

        // Test reference-based remainder
        let f = a % &b;
        assert_eq!(f, Uint4::ZERO);
    }

    #[test]
    fn conversions() {
        // Test From<crypto_bigint::Uint> for Uint
        let original = crypto_bigint::Uint::from(123_u64);
        let wrapped: Uint4 = original.into();
        assert_eq!(wrapped.0, original);

        // Test From<Uint> for crypto_bigint::Uint
        let wrapped = Uint4::from(456_u64);
        let unwrapped: crypto_bigint::Uint<{ 4 * WORD_FACTOR }> = wrapped.into();
        assert_eq!(unwrapped, crypto_bigint::Uint::from(456_u64));

        // Test conversion methods
        let value = crypto_bigint::Uint::from(789_u64);
        let wrapped = Uint4::new(value);
        assert_eq!(wrapped.inner(), &value);
        assert_eq!(wrapped.into_inner(), value);

        assert_eq!(Uint4::from(true), Uint4::ONE);
        assert_eq!(Uint4::from(Boolean::TRUE), Uint4::ONE);
    }

    #[test]
    fn to_primitive_ints() {
        // Zero
        assert_eq!(Uint4::ZERO.to_u64(), Some(0));
        assert_eq!(Uint4::ZERO.to_i64(), Some(0));
        assert_eq!(Uint4::ZERO.to_u128(), Some(0));
        assert_eq!(Uint4::ZERO.to_i128(), Some(0));

        // Small value
        let a = Uint4::from(42_u64);
        assert_eq!(a.to_u64(), Some(42));
        assert_eq!(a.to_i64(), Some(42));
        assert_eq!(a.to_u128(), Some(42));
        assert_eq!(a.to_i128(), Some(42));

        // i64::MAX fits i64, i64::MAX + 1 does not
        let a = Uint4::from(i64::MAX as u64);
        assert_eq!(a.to_i64(), Some(i64::MAX));
        let a = a + Uint4::ONE;
        assert_eq!(a.to_i64(), None);
        assert_eq!(a.to_u64(), Some(i64::MAX as u64 + 1));

        // u64::MAX fits u64, but not i64
        let a = Uint1::MAX;
        assert_eq!(a.to_u64(), Some(u64::MAX));
        assert_eq!(a.to_i64(), None);
        assert_eq!(a.to_u128(), Some(u64::MAX as u128));
        assert_eq!(a.to_i128(), Some(u64::MAX as i128));

        // u64::MAX + 1 fits u128, but not u64
        let a = Uint2::ONE << 64;
        assert_eq!(a.to_u64(), None);
        assert_eq!(a.to_i64(), None);
        assert_eq!(a.to_u128(), Some(u64::MAX as u128 + 1));
        assert_eq!(a.to_i128(), Some(u64::MAX as i128 + 1));

        // i128::MAX fits i128, i128::MAX + 1 does not
        let a = Uint4::from(i128::MAX as u128);
        assert_eq!(a.to_i128(), Some(i128::MAX));
        let a = a + Uint4::ONE;
        assert_eq!(a.to_i128(), None);
        assert_eq!(a.to_u128(), Some(i128::MAX as u128 + 1));

        // u128::MAX fits u128, but not i128
        let a = Uint2::MAX;
        assert_eq!(a.to_u64(), None);
        assert_eq!(a.to_i64(), None);
        assert_eq!(a.to_u128(), Some(u128::MAX));
        assert_eq!(a.to_i128(), None);

        // Nonzero bits above 128 do not fit any primitive
        for shift in [128, 192, 255] {
            let a = Uint4::ONE << shift;
            assert_eq!(a.to_u64(), None);
            assert_eq!(a.to_i64(), None);
            assert_eq!(a.to_u128(), None);
            assert_eq!(a.to_i128(), None);
        }
        assert_eq!(Uint4::MAX.to_u128(), None);

        // Narrower types go through the default impls
        assert_eq!(Uint4::from(255_u64).to_u8(), Some(255));
        assert_eq!(Uint4::from(256_u64).to_u8(), None);
        assert_eq!(Uint4::from(128_u64).to_i8(), None);
    }

    #[test]
    fn to_primitive_floats() {
        // Zero and small value
        assert_eq!(Uint4::ZERO.to_f32(), Some(0.0));
        assert_eq!(Uint4::ZERO.to_f64(), Some(0.0));
        assert_eq!(Uint4::from(42_u64).to_f32(), Some(42.0));
        assert_eq!(Uint4::from(42_u64).to_f64(), Some(42.0));

        // Ties round to even
        let two_pow = |e| FloatCore::powi(2.0_f64, e);
        let a = Uint4::from((1_u64 << 53) + 1);
        assert_eq!(a.to_f64(), Some(two_pow(53)));
        let a = (Uint4::ONE << 100) + (Uint4::ONE << 47);
        assert_eq!(a.to_f64(), Some(two_pow(100)));

        // Truncated bits beyond the top 64 break the tie
        let a = a + Uint4::ONE;
        assert_eq!(a.to_f64(), Some(two_pow(100) + two_pow(48)));
        let a = (Uint4::ONE << 100) + (Uint4::ONE << 76) + Uint4::ONE;
        assert_eq!(
            a.to_f32(),
            Some(FloatCore::powi(2.0_f32, 100) + FloatCore::powi(2.0_f32, 77))
        );

        // MAX is exact, anything rounding above it is infinity
        let a = Uint2::from((1_u64 << 24) - 1) << 104;
        assert_eq!(a.to_f32(), Some(f32::MAX));
        assert_eq!(Uint2::MAX.to_f32(), Some(f32::INFINITY));
        assert_eq!(Uint4::MAX.to_f32(), Some(f32::INFINITY));
        let a = U1024::from((1_u64 << 53) - 1) << 971;
        assert_eq!(a.to_f64(), Some(f64::MAX));
        assert_eq!(U1024::MAX.to_f64(), Some(f64::INFINITY));
        assert_eq!(U1280::MAX.to_f64(), Some(f64::INFINITY));

        // 2^k and 2^k ± 1 round like a single IEEE operation across all bit lengths
        fn assert_floats_correctly_rounded<const LIMBS: usize>() {
            for k in 0..Uint::<LIMBS>::BITS {
                let pow = Uint::<LIMBS>::ONE << k;
                let pow_f32 = FloatCore::powi(2.0_f32, i32::try_from(k).unwrap());
                let pow_f64 = FloatCore::powi(2.0_f64, i32::try_from(k).unwrap());
                let top = Uint::<LIMBS>::BITS - k;
                let max_f32 = FloatCore::powi(2.0_f32, i32::try_from(top).unwrap()) - 1.0;
                let max_f64 = FloatCore::powi(2.0_f64, i32::try_from(top).unwrap()) - 1.0;
                for (a, expected_f32, expected_f64) in [
                    (pow, pow_f32, pow_f64),
                    (pow + Uint::ONE, pow_f32 + 1.0, pow_f64 + 1.0),
                    (Uint::<LIMBS>::MAX >> k, max_f32, max_f64),
                ] {
                    assert_eq!(a.to_f32(), Some(expected_f32), "{a}");
                    assert_eq!(a.to_f64(), Some(expected_f64), "{a}");
                }
            }
        }
        assert_floats_correctly_rounded::<{ WORD_FACTOR }>();
        assert_floats_correctly_rounded::<{ 2 * WORD_FACTOR }>();
        assert_floats_correctly_rounded::<{ 4 * WORD_FACTOR }>();
        assert_floats_correctly_rounded::<{ 20 * WORD_FACTOR }>();
    }

    #[test]
    fn from_primitive_ints() {
        // Inherent `from_u*` shadow the trait methods, hence the qualified calls

        // Zero and small value
        assert_eq!(<Uint4 as FromPrimitive>::from_u64(0), Some(Uint4::ZERO));
        assert_eq!(Uint4::from_i64(0), Some(Uint4::ZERO));
        assert_eq!(<Uint4 as FromPrimitive>::from_u128(0), Some(Uint4::ZERO));
        assert_eq!(Uint4::from_i128(0), Some(Uint4::ZERO));
        let a = Uint4::from(42_u64);
        assert_eq!(<Uint4 as FromPrimitive>::from_u64(42), Some(a));
        assert_eq!(Uint4::from_i64(42), Some(a));
        assert_eq!(<Uint4 as FromPrimitive>::from_u128(42), Some(a));
        assert_eq!(Uint4::from_i128(42), Some(a));

        // Negative values do not fit
        assert_eq!(Uint4::from_i64(-1), None);
        assert_eq!(Uint4::from_i64(i64::MIN), None);
        assert_eq!(Uint4::from_i128(-1), None);
        assert_eq!(Uint4::from_i128(i128::MIN), None);

        // Signed MAX values fit
        let a = Uint1::from(i64::MAX as u64);
        assert_eq!(Uint1::from_i64(i64::MAX), Some(a));
        let a = Uint2::from(i128::MAX as u128);
        assert_eq!(Uint2::from_i128(i128::MAX), Some(a));

        // u64::MAX fits 64 bits
        assert_eq!(
            <Uint1 as FromPrimitive>::from_u64(u64::MAX),
            Some(Uint1::MAX)
        );
        assert_eq!(
            <Uint1 as FromPrimitive>::from_u128(u64::MAX.into()),
            Some(Uint1::MAX)
        );

        // u64::MAX + 1 needs 128 bits
        let n = u128::from(u64::MAX) + 1;
        let i = i128::from(u64::MAX) + 1;
        assert_eq!(<Uint1 as FromPrimitive>::from_u128(n), None);
        assert_eq!(Uint1::from_i128(i), None);
        assert_eq!(
            <Uint2 as FromPrimitive>::from_u128(n),
            Some(Uint2::ONE << 64)
        );
        assert_eq!(Uint2::from_i128(i), Some(Uint2::ONE << 64));
        assert_eq!(
            <Uint4 as FromPrimitive>::from_u128(n),
            Some(Uint4::ONE << 64)
        );

        // u128::MAX fits 128 bits
        assert_eq!(
            <Uint2 as FromPrimitive>::from_u128(u128::MAX),
            Some(Uint2::MAX)
        );
        let a = (Uint4::ONE << 128) - Uint4::ONE;
        assert_eq!(<Uint4 as FromPrimitive>::from_u128(u128::MAX), Some(a));

        // Round trip through `ToPrimitive`
        for n in [0, 1, n - 1, n, i128::MAX as u128, u128::MAX] {
            let a = <Uint4 as FromPrimitive>::from_u128(n).unwrap();
            assert_eq!(a.to_u128(), Some(n));
            let a = <Uint2 as FromPrimitive>::from_u128(n).unwrap();
            assert_eq!(a.to_u128(), Some(n));
        }

        // Narrower types go through the default impls
        let a = Uint1::from(u32::MAX);
        assert_eq!(<Uint1 as FromPrimitive>::from_u32(u32::MAX), Some(a));
        assert_eq!(Uint1::from_i32(-1), None);
        assert_eq!(Uint1::from_usize(42), Some(Uint1::from(42_u64)));
        assert_eq!(Uint1::from_isize(-1), None);
    }

    #[test]
    fn from_primitive_floats() {
        // Zero, including negative zero
        assert_eq!(Uint4::from_f64(0.0), Some(Uint4::ZERO));
        assert_eq!(Uint4::from_f64(-0.0), Some(Uint4::ZERO));
        assert_eq!(Uint4::from_f32(0.0), Some(Uint4::ZERO));

        // Fractions truncate toward zero, like `as` casts
        assert_eq!(Uint4::from_f64(42.9), Some(Uint4::from(42_u64)));
        assert_eq!(Uint4::from_f64(0.9), Some(Uint4::ZERO));
        assert_eq!(Uint4::from_f64(-0.9), Some(Uint4::ZERO));
        assert_eq!(Uint4::from_f64(f64::MIN_POSITIVE), Some(Uint4::ZERO));
        assert_eq!(Uint4::from_f32(1.5), Some(Uint4::ONE));

        // Negative values, NaN and infinities do not fit
        assert_eq!(Uint4::from_f64(-1.0), None);
        assert_eq!(Uint4::from_f64(f64::MIN), None);
        assert_eq!(Uint4::from_f64(f64::NAN), None);
        assert_eq!(Uint4::from_f64(f64::INFINITY), None);
        assert_eq!(Uint4::from_f64(f64::NEG_INFINITY), None);
        assert_eq!(Uint4::from_f32(f32::NAN), None);

        // Largest f64 below 2^64 fits 64 bits, 2^64 does not
        let two_pow = |e| FloatCore::powi(2.0_f64, e);
        let a = Uint1::from(u64::MAX - ((1 << 11) - 1));
        assert_eq!(Uint1::from_f64(two_pow(64) - two_pow(11)), Some(a));
        assert_eq!(Uint1::from_f64(two_pow(64)), None);
        assert_eq!(Uint2::from_f64(two_pow(64)), Some(Uint2::ONE << 64));

        // MAX fits exactly when the type is wide enough
        let a = Uint2::from((1_u64 << 24) - 1) << 104;
        assert_eq!(Uint2::from_f32(f32::MAX), Some(a));
        assert_eq!(Uint1::from_f32(f32::MAX), None);
        let a = U1024::from((1_u64 << 53) - 1) << 971;
        assert_eq!(U1024::from_f64(f64::MAX), Some(a));
        assert_eq!(U960::from_f64(f64::MAX), None);

        // Exact values and round trip through `ToPrimitive` across all exponents
        for e in 0..f64::MAX_EXP {
            let k = e.unsigned_abs();
            let pow = two_pow(e);
            let exact = U1280::ONE << k;
            // Truncated neighbours of 2^k and 2^(k + 1): f64 spacing in [2^k, 2^(k + 1)) is
            // 2^(k - 52)
            let (next, prev) = match k.checked_sub(52) {
                Some(s) => (exact + (U1280::ONE << s), (exact << 1) - (U1280::ONE << s)),
                None => (exact, (exact << 1) - U1280::ONE),
            };
            for (n, expected) in [
                (pow, exact),
                (pow * (1.0 + f64::EPSILON), next),
                (pow * (2.0 - f64::EPSILON), prev),
            ] {
                assert_eq!(U1280::from_f64(n), Some(expected), "{n}");
                assert_eq!(expected.to_f64(), Some(FloatCore::trunc(n)), "{n}");
                // Narrower types fit iff below 2^BITS
                assert_eq!(Uint4::from_f64(n).is_some(), e < 256, "{n}");
            }
        }
    }

    #[test]
    fn pow_operation() {
        // Test basic exponentiation
        let base = Uint4::from(2_u64);

        // 2^0 = 1
        assert_eq!(base.pow(0), Uint4::one());

        // 2^1 = 2
        assert_eq!(base.pow(1), base);

        // 2^3 = 8
        assert_eq!(base.pow(3), Uint4::from(8_u64));

        // 2^10 = 1024
        assert_eq!(base.pow(10), Uint4::from(1024_u64));

        // Test with different base
        let base = Uint4::from(3_u64);

        // 3^4 = 81
        assert_eq!(base.pow(4), Uint4::from(81_u64));

        // Test with base 1
        let base = Uint4::from(1_u64);
        assert_eq!(base.pow(1000), Uint4::from(1_u64));

        // Test with base 0
        let base = Uint4::from(0_u64);
        assert_eq!(base.pow(0), Uint4::one()); // 0^0 = 1 by convention
        assert_eq!(base.pow(10), Uint4::zero()); // 0^n = 0 for n > 0
    }

    #[test]
    fn rem_assign_operations() {
        // Test RemAssign with owned value
        let mut a = Uint4::from(17_u64);
        let b = Uint4::from(5_u64);
        a %= b;
        assert_eq!(a, Uint4::from(2_u64));

        // Test RemAssign with reference
        let mut c = Uint4::from(19_u64);
        let d = Uint4::from(6_u64);
        c %= &d;
        assert_eq!(c, Uint4::from(1_u64));

        // Test with divisor 1
        let mut e = Uint4::from(42_u64);
        let one = Uint4::one();
        e %= &one;
        assert_eq!(e, Uint4::zero());
    }

    #[test]
    #[should_panic(expected = "division by zero")]
    fn rem_assign_panics_on_zero_divisor() {
        let mut a = Uint4::from(10_u64);
        let zero = Uint4::zero();
        a %= zero;
    }

    #[test]
    fn resize_method() {
        // Test resizing to same size
        let a = Uint4::from(0x12345678_u64);
        let resized_same = a.resize::<{ 4 * WORD_FACTOR }>();
        assert_eq!(resized_same, a);

        // Test resizing to larger size
        let b = Uint2::from(0x9ABCDEF0_u64);
        let resized_larger = b.resize::<{ 4 * WORD_FACTOR }>();
        assert_eq!(resized_larger.as_words()[0], b.as_words()[0]);
        assert_eq!(resized_larger.as_words()[1], b.as_words()[1]);
        assert_eq!(resized_larger.as_words()[2], 0);
        assert_eq!(resized_larger.as_words()[3], 0);

        // Test resizing to smaller size (truncation)
        let c = Uint4::from(0x1234567890ABCDEF_u64);
        let resized_smaller = c.resize::<{ 2 * WORD_FACTOR }>();
        assert_eq!(resized_smaller.as_words()[0], c.as_words()[0]);
        assert_eq!(resized_smaller.as_words()[1], c.as_words()[1]);
    }

    #[test]
    fn from_words() {
        // Test with single limb
        let words = [0x1234567890ABCDEF];
        let a = Uint1::from_words(words);
        assert_eq!(a.as_words()[0], words[0]);

        // Test with multiple limbs
        let words = [
            0x1234567890ABCDEF,
            0xFEDCBA9876543210,
            0x0F0F0F0F0F0F0F0F,
            0xF0F0F0F0F0F0F0F0,
        ];
        let b = Uint4::from_words(words);
        let b_words = b.as_words();
        for i in 0..4 {
            assert_eq!(b_words[i], words[i]);
        }
    }

    #[test]
    fn aggregate_operations() {
        let values: Vec<Uint4> = [1_u64, 2_u64, 3_u64].into_iter().map(Uint::from).collect();
        assert_eq!(values.iter().sum::<Uint4>(), Uint4::from(6_u64));
        assert_eq!(values.into_iter().sum::<Uint4>(), Uint4::from(6_u64));

        let values: Vec<Uint4> = [2_u64, 3_u64, 4_u64].into_iter().map(Uint::from).collect();
        assert_eq!(values.iter().product::<Uint4>(), Uint4::from(24_u64));
    }

    #[test]
    fn from_primitive() {
        // Test from_u8
        let a = Uint4::from_u8(42);
        assert_eq!(a, Uint4::from(42_u64));

        // Test from_u16
        let c = Uint4::from_u16(12345);
        assert_eq!(c, Uint4::from(12345_u64));

        // Test from_u32
        let e = Uint4::from_u32(1234567890);
        assert_eq!(e, Uint4::from(1234567890_u64));

        // Test from_u64
        let g = Uint4::from_u64(1234567890123456789);
        assert_eq!(g, Uint4::from(1234567890123456789_u64));

        // Test from_u128
        let i = Uint4::from_u128(1234567890123456789012345678901234567);
        assert_eq!(
            i.into_inner(),
            crypto_bigint::Uint::from(1234567890123456789012345678901234567_u128)
        );
    }

    #[test]
    fn from_primitive_edge_cases() {
        for value in [u32::MIN, u32::MAX] {
            let i = Uint1::from(value);
            let j = Uint2::from(value);
            assert_eq!(i.resize(), j);
        }

        for value in [u64::MIN, u64::MAX] {
            let i = Uint1::from(value);
            let j = Uint2::from(value);
            assert_eq!(i.resize(), j);
        }

        for value in [u128::MIN, u128::MAX] {
            let i = Uint2::from(value);
            let j = Uint::<3>::from(value);
            assert_eq!(i.resize(), j);
        }
    }

    #[should_panic]
    #[test]
    fn from_too_large_primitive() {
        // Test from_u128
        let _ = Uint1::from(u128::MAX);
    }

    #[test]
    fn edge_cases() {
        // Test operations with MAX values
        let max = Uint4::MAX;
        let one = Uint4::one();

        // MAX + 1 should overflow in checked_add
        assert!(max.checked_add(&one).is_none());

        // MAX - MAX = 0
        assert_eq!(max.checked_sub(&max).unwrap(), Uint4::zero());

        // Test operations with MIN values (0 for unsigned)
        let min = Uint4::ZERO;

        // MIN - 1 should overflow in checked_sub
        assert!(min.checked_sub(&one).is_none());

        // Test operations with large shifts
        let x = Uint4::from(1_u64);

        // Shift left by almost the bit limit
        let shifted = x << (Uint4::BITS - 1);
        let expected = {
            let mut expected_words = [0; { 4 * WORD_FACTOR }];
            let len = expected_words.len();
            expected_words[len - 1] = (1 as Word) << (Word::BITS - 1);
            Uint4::from_words(expected_words)
        };
        assert_eq!(shifted, expected);

        // Test with large powers that don't overflow
        let two = Uint4::from(2_u64);
        let large_power = two.pow(100); // 2^100 is large but fits in 256 bits

        // 2^100 should be divisible by 2^10 = 1024 with no remainder
        assert_eq!(large_power % Uint4::from(1024_u64), Uint4::zero());

        // 2^100 / 2 = 2^99
        let half_power = large_power >> 1;
        assert_eq!(half_power << 1, large_power);
    }

    #[test]
    fn assign_operations() {
        // Test AddAssign
        let mut a = Uint4::from(10_u64);
        a += Uint4::from(5_u64);
        assert_eq!(a, Uint4::from(15_u64));

        let mut b = Uint4::from(20_u64);
        b += &Uint4::from(3_u64);
        assert_eq!(b, Uint4::from(23_u64));

        // Test SubAssign
        let mut c = Uint4::from(10_u64);
        c -= Uint4::from(3_u64);
        assert_eq!(c, Uint4::from(7_u64));

        let mut d = Uint4::from(50_u64);
        d -= &Uint4::from(25_u64);
        assert_eq!(d, Uint4::from(25_u64));

        // Test MulAssign
        let mut e = Uint4::from(7_u64);
        e *= Uint4::from(6_u64);
        assert_eq!(e, Uint4::from(42_u64));

        let mut f = Uint4::from(3_u64);
        f *= &Uint4::from(4_u64);
        assert_eq!(f, Uint4::from(12_u64));

        let mut f = Uint1::from(2_u64);
        f <<= 2;
        assert_eq!(f, Uint1::from(8_u64)); // 2 << 2 = 8
        f <<= 61;
        assert_eq!(f, Uint1::ZERO);

        let mut f = Uint1::from(3_u64);
        f >>= 1;
        assert_eq!(f, Uint1::from(1_u64)); // 3 >> 1 = 1
        f >>= 1;
        assert_eq!(f, Uint1::ZERO);
    }

    #[test]
    fn formatting() {
        let a = Uint1::from(255_u64);
        let b = Uint1::MAX;

        // Test Debug
        assert_eq!(format!("{:?}", a), "Uint(0x00000000000000FF)");
        assert_eq!(format!("{:?}", b), "Uint(0xFFFFFFFFFFFFFFFF)");

        // Test Display
        assert_eq!(format!("{}", a), "00000000000000FF");
        assert_eq!(format!("{}", b), "FFFFFFFFFFFFFFFF");

        // Test LowerHex
        assert_eq!(format!("{:x}", a), "00000000000000ff");
        assert_eq!(format!("{:x}", b), "ffffffffffffffff");

        // Test UpperHex
        assert_eq!(format!("{:X}", a), "00000000000000FF");
        assert_eq!(format!("{:X}", b), "FFFFFFFFFFFFFFFF");
    }

    #[test]
    fn default_trait() {
        let default_val: Uint4 = Default::default();
        assert_eq!(default_val, Uint4::ZERO);
        assert!(default_val.is_zero());
    }

    #[test]
    fn constants() {
        // Test MAX
        assert!(Uint4::MAX > Uint4::ZERO);

        // Test BITS, BYTES, LIMBS
        assert_eq!(Uint4::BITS, 256);
        assert_eq!(Uint4::BYTES, 32);
        assert_eq!(Uint4::LIMBS, 4);

        assert_eq!(Uint2::BITS, 128);
        assert_eq!(Uint2::BYTES, 16);
        assert_eq!(Uint2::LIMBS, 2);
    }

    #[test]
    fn cmp() {
        let a = Uint4::from(10_u64);
        let b = Uint4::from(20_u64);
        let c = Uint4::from(10_u64);

        assert_eq!(a.cmp(&b), Ordering::Less);
        assert_eq!(b.cmp(&a), Ordering::Greater);
        assert_eq!(a.cmp(&c), Ordering::Equal);
    }

    #[test]
    fn cross_size_conversions() {
        // Test resize methods
        let a = Uint2::from(12345_u64);
        let b: Option<Uint4> = a.checked_resize();
        assert_eq!(b, Some(Uint::from(12345_u64)));
        let b: Uint4 = a.resize();
        assert_eq!(b, Uint::from(12345_u64));

        let a = Uint2::from_u128(u128::MAX);
        let b: Option<Uint1> = a.checked_resize();
        assert_eq!(b, None);
        let b: Uint1 = a.resize();
        assert_eq!(b, Uint::MAX);

        // Test From<&crypto_bigint::Uint<LIMBS>> for Uint<LIMBS2>
        let c = crypto_bigint::Uint::<{ 2 * WORD_FACTOR }>::from(67890_u64);
        let d: Uint4 = (&c).try_into().unwrap();
        assert_eq!(d, Uint4::from(67890_u64));

        // Test reference conversions from primitives
        let val = 42_u32;
        let e = Uint4::from(&val);
        assert_eq!(e, Uint4::from(42_u64));
    }

    #[test]
    fn constant_time_traits() {
        use crypto_bigint::{Choice, CtEq, CtGt, CtLt, CtSelect};

        let a = Uint4::from(10_u64);
        let b = Uint4::from(20_u64);
        let c = Uint4::from(10_u64);

        // Test CtEq
        assert_eq!(a.ct_eq(&c).to_u8(), 1);
        assert_eq!(a.ct_eq(&b).to_u8(), 0);

        // Test CtGt
        assert_eq!(b.ct_gt(&a).to_u8(), 1);
        assert_eq!(a.ct_gt(&b).to_u8(), 0);

        // Test CtLt
        assert_eq!(a.ct_lt(&b).to_u8(), 1);
        assert_eq!(b.ct_lt(&a).to_u8(), 0);

        // Test CtSelect
        let selected_true = a.ct_select(&b, Choice::from(0));
        assert_eq!(selected_true, a);

        let selected_false = a.ct_select(&b, Choice::from(1));
        assert_eq!(selected_false, b);
    }

    #[test]
    fn crypto_bigint_traits() {
        use crypto_bigint::Bounded;

        // Test Bounded trait
        assert_eq!(<Uint4 as Bounded>::BITS, 256);
        assert_eq!(<Uint4 as Bounded>::BYTES, 32);
    }

    #[test]
    fn from_str() {
        // Test parsing from string
        let a: Uint4 = "123".parse().unwrap();
        assert_eq!(a, Uint4::from(123_u64));

        let c: Uint4 = "0".parse().unwrap();
        assert_eq!(c, Uint4::zero());

        let d: Uint4 = "1".parse().unwrap();
        assert_eq!(d, Uint4::one());
        assert_eq!(u64::MAX.to_string().parse::<Uint1>().unwrap(), Uint1::MAX);

        assert_eq!("0xFF".parse::<Uint4>().unwrap(), Uint4::from(255_u64));

        // Test invalid cases
        assert!("abc".parse::<Uint4>().is_err());
        assert!("12.34".parse::<Uint4>().is_err());
        assert!("".parse::<Uint4>().is_err());
        assert!("-456".parse::<Uint4>().is_err()); // Negative not allowed for unsigned

        // Number doesn't fit Uint1
        assert!(
            ((u64::MAX as u128) + 1)
                .to_string()
                .parse::<Uint1>()
                .is_err()
        );
    }

    #[cfg(feature = "rand")]
    #[test]
    fn random_generation() {
        use rand::prelude::*;

        // Use a seeded RNG for reproducibility
        let mut rng = StdRng::seed_from_u64(1);

        // Test crypto_bigint::Random trait
        let random1: Uint4 = <Uint4 as crypto_bigint::Random>::random_from_rng(&mut rng);
        let random2: Uint4 = <Uint4 as crypto_bigint::Random>::random_from_rng(&mut rng);

        // Random values should be different
        assert_ne!(random1, random2);

        // Test Distribution trait
        let random3: Uint4 = rng.random();
        let random4: Uint4 = rng.random();

        assert_ne!(random3, random4);
    }

    #[test]
    fn wrapping_operations() {
        // WrappingAdd
        let max = Uint1::MAX;
        let one = Uint1::ONE;
        let wrapped_add = max.wrapping_add(&one);
        assert_eq!(wrapped_add, Uint1::ZERO); // MAX + 1 wraps to 0

        let wrapped_add2 = max.wrapping_add(&max);
        assert_eq!(wrapped_add2, Uint1::MAX - Uint1::ONE); // MAX + MAX wraps

        // WrappingSub
        let zero = Uint1::ZERO;
        let wrapped_sub = zero.wrapping_sub(&one);
        assert_eq!(wrapped_sub, Uint1::MAX); // 0 - 1 wraps to MAX

        let wrapped_sub2 = one.wrapping_sub(&Uint1::from(2_u64));
        assert_eq!(wrapped_sub2, Uint1::MAX); // 1 - 2 wraps to MAX

        // WrappingMul
        let large = Uint1::from(u64::MAX);
        let two = Uint1::from(2_u64);
        let wrapped_mul = large.wrapping_mul(&two);
        // u64::MAX * 2 = 2 * (2^64 - 1) = 2^65 - 2, which wraps to 2^64 - 2 = MAX - 1
        assert_eq!(wrapped_mul, Uint1::MAX - Uint1::ONE);

        // Non-overflowing cases should work normally
        let a = Uint4::from(10_u64);
        let b = Uint4::from(5_u64);
        assert_eq!(a.wrapping_add(&b), Uint4::from(15_u64));
        assert_eq!(a.wrapping_sub(&b), Uint4::from(5_u64));
        assert_eq!(a.wrapping_mul(&b), Uint4::from(50_u64));
    }

    #[test]
    #[cfg(feature = "zerocopy")]
    fn zerocopy() {
        ensure_type_implements_trait!(Uint4, zerocopy::KnownLayout);
    }
}
