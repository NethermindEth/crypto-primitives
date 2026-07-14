use super::*;
use crate::{Wrapper, boolean::Boolean, pow_via_repeated_squaring};
use alloc::boxed::Box;
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
use crypto_bigint::{
    BitOps, ConcatenatingMul, DivVartime, Integer, Limb, RandomBitsError, Resize, UintRef, Word,
};
use num_traits::{
    CheckedAdd, CheckedMul, CheckedRem, CheckedSub, One, Pow, WrappingAdd, WrappingMul,
    WrappingSub, Zero,
};
use pastey::paste;

#[cfg(feature = "rand")]
use rand::rand_core::TryRng;

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct BoxedUint(pub crypto_bigint::BoxedUint);

impl BoxedUint {
    /// Wraps a given value into this wrapper type
    #[inline(always)]
    pub const fn new(value: crypto_bigint::BoxedUint) -> Self {
        Self(value)
    }

    #[inline(always)]
    pub const fn new_ref(value: &crypto_bigint::BoxedUint) -> &Self {
        // Safety: BoxedUint is #[repr(transparent)] and is guaranteed to have the
        // same memory layout as crypto_bigint::BoxedUint
        unsafe { &*(value as *const crypto_bigint::BoxedUint as *const Self) }
    }

    #[inline(always)]
    pub const fn new_ref_mut(value: &mut crypto_bigint::BoxedUint) -> &mut Self {
        // Safety: BoxedUint is #[repr(transparent)] and is guaranteed to have the
        // same memory layout as crypto_bigint::BoxedUint
        unsafe { &mut *(value as *mut crypto_bigint::BoxedUint as *mut Self) }
    }

    /// See [crypto_bigint::BoxedUint::from_words]
    #[inline(always)]
    pub fn from_words(words: impl IntoIterator<Item = Word>) -> Self {
        Self(crypto_bigint::BoxedUint::from_words(words))
    }

    /// See [crypto_bigint::BoxedUint::to_words]
    #[inline]
    pub fn to_words(self) -> Box<[Word]> {
        self.0.to_words()
    }

    /// See [crypto_bigint::BoxedUint::as_words]
    pub fn as_words(&self) -> &[Word] {
        self.0.as_words()
    }

    /// See [crypto_bigint::BoxedUint::as_mut_words]
    pub fn as_mut_words(&mut self) -> &mut [Word] {
        self.0.as_mut_words()
    }

    /// See [crypto_bigint::BoxedUint::as_limbs]
    pub fn as_limbs(&self) -> &[Limb] {
        self.0.as_limbs()
    }

    /// See [crypto_bigint::BoxedUint::as_mut_limbs]
    pub fn as_mut_limbs(&mut self) -> &mut [Limb] {
        self.0.as_mut_limbs()
    }

    /// See [crypto_bigint::BoxedUint::to_limbs]
    pub fn to_limbs(self) -> Box<[Limb]> {
        self.0.to_limbs()
    }

    /// See [crypto_bigint::BoxedUint::try_resize]
    #[inline(always)]
    pub fn try_resize(&self, at_least_bits_precision: u32) -> Option<Self> {
        (&self.0).try_resize(at_least_bits_precision).map(Self)
    }

    /// See [crypto_bigint::BoxedUint::resize_unchecked] - using unchecked
    /// variant to be consistent with [crypto_bigint::Uint::resize]
    #[inline(always)]
    pub fn resize(&self, at_least_bits_precision: u32) -> Self {
        Self((&self.0).resize_unchecked(at_least_bits_precision))
    }

    /// See [crypto_bigint::BoxedUint::cmp_vartime]
    pub fn cmp_vartime(&self, rhs: &Self) -> Ordering {
        self.0.cmp_vartime(&rhs.0)
    }

    /// See [crypto_bigint::BoxedUint::from_be_hex]
    pub fn from_be_hex(hex: &str, bits_precision: u32) -> Option<Self> {
        crypto_bigint::BoxedUint::from_be_hex(hex, bits_precision)
            .into_option()
            .map(Self)
    }

    /// See [`crypto_bigint::BoxedUint::bits`]
    pub fn bits(&self) -> u32 {
        self.0.bits()
    }

    /// See [`crypto_bigint::BoxedUint::bits_precision`]
    pub fn bits_precision(&self) -> u32 {
        self.0.bits_precision()
    }

    /// See [`crypto_bigint::BoxedUint::bytes_precision`]
    pub fn bytes_precision(&self) -> usize {
        self.0.bytes_precision()
    }

    /// See [`crypto_bigint::BoxedUint::nlimbs`]
    pub fn nlimbs(&self) -> usize {
        self.0.nlimbs()
    }

    /// See [`crypto_bigint::BoxedUint::zero_with_precision`]
    pub fn zero_with_precision(at_least_bits_precision: u32) -> Self {
        Self(crypto_bigint::BoxedUint::zero_with_precision(
            at_least_bits_precision,
        ))
    }

    /// See [`crypto_bigint::BoxedUint::one_with_precision`]
    pub fn one_with_precision(at_least_bits_precision: u32) -> Self {
        Self(crypto_bigint::BoxedUint::one_with_precision(
            at_least_bits_precision,
        ))
    }

    /// See [`crypto_bigint::BoxedUint::concatenating_mul`]
    pub fn concatenating_mul(&self, rhs: &[Limb]) -> Self {
        let rhs = UintRef::new(rhs);
        self.0.concatenating_mul(rhs).into()
    }

    /// Get the least significant 64-bits.
    pub fn lowest_u64(&self) -> u64 {
        #[cfg(target_pointer_width = "32")]
        let res = {
            debug_assert!(self.nlimbs() >= 1);
            let mut ret = self.limbs[0].0 as u64;

            if self.nlimbs() >= 2 {
                ret |= (self.limbs[1].0 as u64) << 32;
            }

            ret
        };

        #[cfg(target_pointer_width = "64")]
        let res = self.as_limbs()[0].0;

        #[cfg(not(any(target_pointer_width = "32", target_pointer_width = "64")))]
        let res = panic!("Unsupported target pointer width");

        res
    }

    /// See [`crypto_bigint::BoxedUint::conditional_wrapping_neg_assign`]
    #[allow(clippy::cast_possible_truncation, clippy::arithmetic_side_effects)]
    pub fn conditional_wrapping_neg_assign(&mut self, choice: crypto_bigint::Choice) {
        use crypto_bigint::CtAssign;
        let mut carry = 1;

        for i in 0..self.nlimbs() {
            let r = crypto_bigint::WideWord::from(!self.as_limbs()[i].0) + carry;
            self.as_mut_limbs()[i].ct_assign(&Limb(r as Word), choice);
            carry = r >> Limb::BITS;
        }
    }

    /// See [`crypto_bigint::BoxedUint::borrowing_sub_assign`]
    #[inline(always)]
    #[allow(clippy::needless_range_loop)]
    pub fn borrowing_sub_assign(&mut self, rhs: &[Limb], borrow: Limb) -> Limb {
        assert!(rhs.len() <= self.nlimbs());

        let mut carry = borrow;
        let self_limbs = self.as_mut_limbs();

        for i in 0..self_limbs.len() {
            let &b = rhs.get(i).unwrap_or(&Limb::ZERO);
            (self_limbs[i], carry) = Limb::borrowing_sub(self_limbs[i], b, carry);
        }
        carry
    }

    /// See [`crypto_bigint::BoxedUint::sub_assign_mod_with_carry`]
    #[inline(always)]
    pub fn sub_assign_mod_with_carry(&mut self, carry: Limb, rhs: &BoxedUint, p: &BoxedUint) {
        debug_assert!(carry.0 <= 1);

        let borrow = self.borrowing_sub_assign(rhs.as_limbs(), Limb::ZERO);

        // The new `borrow = Word::MAX` iff `carry == 0` and `borrow == Word::MAX`.
        let mask = carry.wrapping_neg().not().bitand(borrow);

        // If underflow occurred on the final limb, borrow = 0xfff...fff, otherwise
        // borrow = 0x000...000. Thus, we use it as a mask to conditionally add the
        // modulus.
        let _ = UintRef::new_mut(self.as_mut_limbs())
            .conditional_add_assign(UintRef::new(p.as_limbs()), Limb::ZERO, !mask.is_zero())
            .lsb_to_choice();
    }
}

//
// Core traits
//

impl AsRef<UintRef> for BoxedUint {
    #[inline(always)]
    fn as_ref(&self) -> &UintRef {
        self.inner().as_ref()
    }
}

impl AsMut<UintRef> for BoxedUint {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut UintRef {
        self.inner_mut().as_mut()
    }
}

impl Debug for BoxedUint {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        Debug::fmt(&self.0, f)
    }
}

impl Display for BoxedUint {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        Display::fmt(&self.0, f)
    }
}

impl Default for BoxedUint {
    #[inline(always)]
    fn default() -> Self {
        Self::zero()
    }
}

impl LowerHex for BoxedUint {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        LowerHex::fmt(&self.0, f)
    }
}

impl UpperHex for BoxedUint {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        UpperHex::fmt(&self.0, f)
    }
}

impl Hash for BoxedUint {
    #[allow(clippy::arithmetic_side_effects)]
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Equality is by numeric value and ignores high zero limbs, so we must
        // hash only the significant limbs to keep `Hash` consistent with `Eq`.
        let limbs = self.0.as_limbs();
        let significant = limbs.iter().rposition(|l| l.0 != 0).map_or(0, |i| i + 1);
        limbs[..significant].hash(state)
    }
}

impl FromStr for BoxedUint {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (radix, s) = if let Some(s) = s.strip_prefix("0x") {
            (16, s)
        } else {
            (10, s)
        };
        let uint = crypto_bigint::BoxedUint::from_str_radix_vartime(s, radix).map_err(|_| ())?;
        Ok(Self(uint))
    }
}

//
// Zero and One traits
//

impl Zero for BoxedUint {
    #[inline(always)]
    fn zero() -> Self {
        Self(crypto_bigint::BoxedUint::zero())
    }

    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.0.is_zero().to_bool_vartime()
    }
}

impl One for BoxedUint {
    #[inline(always)]
    fn one() -> Self {
        Self(crypto_bigint::BoxedUint::one())
    }
}

//
// Basic arithmetic operations
//

macro_rules! impl_basic_and_wrapping_op {
    ($trait_name:tt, $trait_op:tt) => {
        paste! {
            impl $trait_name for BoxedUint {
                type Output = Self;

                #[inline(always)]
                fn $trait_op(self, rhs: Self) -> Self::Output {
                    self.$trait_op(&rhs)
                }
            }

            impl<'a> $trait_name<&'a Self> for BoxedUint {
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

            impl [<Wrapping $trait_name>] for BoxedUint {
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

impl Div for BoxedUint {
    type Output = Self;

    #[inline(always)]
    fn div(self, rhs: Self) -> Self::Output {
        let non_zero = crypto_bigint::NonZero::new(rhs.0).expect("division by zero");
        Self(self.0.div_vartime(&non_zero))
    }
}

impl<'a> Div<&'a Self> for BoxedUint {
    type Output = Self;

    fn div(self, rhs: &'a Self) -> Self::Output {
        self.div(rhs.clone())
    }
}

impl Rem for BoxedUint {
    type Output = Self;

    #[inline(always)]
    fn rem(self, rhs: Self) -> Self::Output {
        let non_zero = crypto_bigint::NonZero::new(rhs.0).expect("division by zero");
        Self(self.0.rem_vartime(&non_zero))
    }
}

impl<'a> Rem<&'a Self> for BoxedUint {
    type Output = Self;

    fn rem(self, rhs: &'a Self) -> Self::Output {
        self.rem(rhs.clone())
    }
}

impl Shl<u32> for BoxedUint {
    type Output = Self;

    #[inline(always)]
    fn shl(self, rhs: u32) -> Self::Output {
        Self(self.0.shl_vartime(rhs).expect("shl overflow"))
    }
}

impl Shr<u32> for BoxedUint {
    type Output = Self;

    #[inline(always)]
    fn shr(self, rhs: u32) -> Self::Output {
        Self(self.0.shr_vartime(rhs).expect("shl overflow"))
    }
}

impl Pow<u32> for BoxedUint {
    type Output = Self;

    fn pow(self, rhs: u32) -> Self::Output {
        let precision = self.bits_precision();
        pow_via_repeated_squaring!(self, rhs, Self::one_with_precision(precision))
    }
}

//
// Checked arithmetic operations
//

impl CheckedAdd for BoxedUint {
    fn checked_add(&self, other: &Self) -> Option<Self> {
        let (result, overflow) = self.0.carrying_add(&other.0, crypto_bigint::Limb::ZERO);
        if overflow.0 != 0 {
            None
        } else {
            Some(Self(result))
        }
    }
}

impl CheckedSub for BoxedUint {
    fn checked_sub(&self, other: &Self) -> Option<Self> {
        let (result, borrow) = self.0.borrowing_sub(&other.0, crypto_bigint::Limb::ZERO);
        if borrow.0 != 0 {
            None
        } else {
            Some(Self(result))
        }
    }
}

impl CheckedMul for BoxedUint {
    fn checked_mul(&self, other: &Self) -> Option<Self> {
        self.0.checked_mul(&other.0).into_option().map(Self)
    }
}

impl CheckedRem for BoxedUint {
    fn checked_rem(&self, other: &Self) -> Option<Self> {
        let non_zero = crypto_bigint::NonZero::new(other.0.clone()).into_option()?;
        Some(Self(self.inner().rem(&non_zero)))
    }
}

//
// Arithmetic assign operations
//

macro_rules! impl_assign_op {
    ($trait_name:tt, $trait_op:tt) => {
        impl $trait_name<Self> for BoxedUint {
            #[inline(always)]
            fn $trait_op(&mut self, rhs: Self) {
                self.$trait_op(&rhs);
            }
        }

        impl<'a> $trait_name<&'a Self> for BoxedUint {
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

impl RemAssign for BoxedUint {
    #[inline(always)]
    fn rem_assign(&mut self, rhs: Self) {
        self.rem_assign(&rhs);
    }
}

impl<'a> RemAssign<&'a Self> for BoxedUint {
    #![allow(clippy::arithmetic_side_effects)]
    fn rem_assign(&mut self, rhs: &'a Self) {
        let non_zero = crypto_bigint::NonZero::new(rhs.0.clone()).expect("division by zero");
        self.0 %= non_zero;
    }
}

impl ShlAssign<u32> for BoxedUint {
    #[inline(always)]
    fn shl_assign(&mut self, rhs: u32) {
        self.0.shl_assign(rhs);
    }
}

impl ShrAssign<u32> for BoxedUint {
    #[inline(always)]
    fn shr_assign(&mut self, rhs: u32) {
        self.0.shr_assign(rhs);
    }
}

//
// Aggregate operations
//

impl Sum for BoxedUint {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| {
            acc.checked_add(&x).expect("overflow in sum")
        })
    }
}

impl<'a> Sum<&'a Self> for BoxedUint {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| {
            acc.checked_add(x).expect("overflow in sum")
        })
    }
}

impl Product for BoxedUint {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| {
            acc.checked_mul(&x).expect("overflow in product")
        })
    }
}

impl<'a> Product<&'a Self> for BoxedUint {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| {
            acc.checked_mul(x).expect("overflow in product")
        })
    }
}

//
// Conversions
//

impl From<crypto_bigint::BoxedUint> for BoxedUint {
    #[inline(always)]
    fn from(value: crypto_bigint::BoxedUint) -> Self {
        Self(value)
    }
}

impl From<BoxedUint> for crypto_bigint::BoxedUint {
    #[inline(always)]
    fn from(value: BoxedUint) -> Self {
        value.0
    }
}

impl<'a> From<&'a crypto_bigint::BoxedUint> for &'a BoxedUint {
    #[inline(always)]
    fn from(value: &'a crypto_bigint::BoxedUint) -> Self {
        BoxedUint::new_ref(value)
    }
}

impl<'a> From<&'a BoxedUint> for &'a crypto_bigint::BoxedUint {
    #[inline(always)]
    fn from(value: &'a BoxedUint) -> Self {
        &value.0
    }
}

impl From<bool> for BoxedUint {
    #[inline(always)]
    fn from(value: bool) -> Self {
        Self(crypto_bigint::BoxedUint::from(u8::from(value)))
    }
}

impl From<Boolean> for BoxedUint {
    #[inline(always)]
    fn from(value: Boolean) -> Self {
        Self::from(value.into_inner())
    }
}

macro_rules! impl_from_primitive {
    ($($t:ty),+) => {
        $(
            impl From<$t> for BoxedUint {
                fn from(value: $t) -> Self {
                    Self(crypto_bigint::BoxedUint::from(value))
                }
            }

            impl<'a> From<&'a $t> for BoxedUint {
                #[inline(always)]
                fn from(value: &$t) -> Self {
                    Self::from(*value)
                }
            }
        )+
    };
}

impl_from_primitive!(u8, u16, u32, u64, u128);

impl<const LIMBS: usize> From<crypto_bigint::Uint<LIMBS>> for BoxedUint {
    #[inline(always)]
    fn from(value: crypto_bigint::Uint<LIMBS>) -> Self {
        Self(crypto_bigint::BoxedUint::from(value))
    }
}

impl<const LIMBS: usize> From<&crypto_bigint::Uint<LIMBS>> for BoxedUint {
    #[inline(always)]
    fn from(value: &crypto_bigint::Uint<LIMBS>) -> Self {
        Self(crypto_bigint::BoxedUint::from(value))
    }
}

//
// Wrapper
//

impl Wrapper for BoxedUint {
    type Inner = crypto_bigint::BoxedUint;

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

impl Semiring for BoxedUint {}

impl IntSemiring for BoxedUint {
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
impl crypto_bigint::RandomBits for BoxedUint {
    fn try_random_bits<R: TryRng + ?Sized>(
        rng: &mut R,
        bit_length: u32,
    ) -> Result<Self, RandomBitsError<R::Error>> {
        crypto_bigint::BoxedUint::try_random_bits(rng, bit_length).map(Self)
    }

    fn try_random_bits_with_precision<R: TryRng + ?Sized>(
        rng: &mut R,
        bit_length: u32,
        bits_precision: u32,
    ) -> Result<Self, RandomBitsError<R::Error>> {
        crypto_bigint::BoxedUint::try_random_bits_with_precision(rng, bit_length, bits_precision)
            .map(Self)
    }
}

//
// Serialization and Deserialization
//

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for BoxedUint {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        crypto_bigint::BoxedUint::deserialize(deserializer).map(Self)
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for BoxedUint {
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
impl zeroize::Zeroize for BoxedUint {
    fn zeroize(&mut self) {
        self.0.zeroize()
    }
}

//
// Traits from crypto_bigint
//

impl crypto_bigint::CtEq for BoxedUint {
    #[inline]
    fn ct_eq(&self, other: &Self) -> crypto_bigint::Choice {
        crypto_bigint::CtEq::ct_eq(&self.0, &other.0)
    }
}

impl crypto_bigint::CtGt for BoxedUint {
    #[inline]
    fn ct_gt(&self, other: &Self) -> crypto_bigint::Choice {
        crypto_bigint::CtGt::ct_gt(&self.0, &other.0)
    }
}

impl crypto_bigint::CtLt for BoxedUint {
    #[inline]
    fn ct_lt(&self, other: &Self) -> crypto_bigint::Choice {
        crypto_bigint::CtLt::ct_lt(&self.0, &other.0)
    }
}

impl crypto_bigint::CtSelect for BoxedUint {
    fn ct_select(&self, other: &Self, choice: crypto_bigint::Choice) -> Self {
        crypto_bigint::CtSelect::ct_select(&self.0, &other.0, choice).into()
    }
}

#[allow(
    clippy::arithmetic_side_effects,
    clippy::cast_lossless,
    clippy::identity_op,
    clippy::redundant_clone
)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::ensure_type_implements_trait;
    use alloc::{format, string::ToString, vec, vec::Vec};

    #[cfg(target_pointer_width = "64")]
    const WORD_FACTOR: usize = 1;
    #[cfg(target_pointer_width = "32")]
    const WORD_FACTOR: usize = 2;

    #[cfg(target_pointer_width = "64")]
    const WORD_BITS_FACTOR: u32 = 64;
    #[cfg(target_pointer_width = "32")]
    const WORD_BITS_FACTOR: u32 = 32;

    fn max(num_u64_words: usize) -> BoxedUint {
        BoxedUint::from_words(vec![Word::MAX; num_u64_words * WORD_FACTOR])
    }

    #[derive(Default)]
    struct ByteHasher(Vec<u8>);

    impl Hasher for ByteHasher {
        fn finish(&self) -> u64 {
            0
        }

        fn write(&mut self, bytes: &[u8]) {
            self.0.extend_from_slice(bytes);
        }
    }

    fn hash_of(x: &BoxedUint) -> Vec<u8> {
        let mut hasher = ByteHasher::default();
        x.hash(&mut hasher);
        hasher.0
    }

    #[test]
    fn ensure_traits() {
        ensure_type_implements_trait!(BoxedUint, Wrapper);
        ensure_type_implements_trait!(BoxedUint, FixedSemiring);
        ensure_type_implements_trait!(BoxedUint, IntSemiring);
        ensure_type_implements_trait!(BoxedUint, IntSemiringWithShifts);
    }

    #[test]
    fn basic_operations() {
        let a = BoxedUint::from(10_u64);
        let b = BoxedUint::from(5_u64);

        // Test addition
        assert_eq!(a.clone() + b.clone(), BoxedUint::from(15_u64));

        // Test subtraction
        assert_eq!(a.clone() - b.clone(), BoxedUint::from(5_u64));

        // Test multiplication
        assert_eq!(a.clone() * b.clone(), BoxedUint::from(50_u64));

        // Test remainder
        assert_eq!(a.clone() % b.clone(), BoxedUint::zero());

        // Test shl
        let x = BoxedUint::from(0x0001_u64);
        assert_eq!(x.clone() << 0, x.clone());
        assert_eq!(x.clone() << 1, 0x0002_u64.into());
        assert_eq!(x.clone() << 15, 0x8000_u64.into());

        // Test shr
        let x = BoxedUint::from(0x8000_u32);
        assert_eq!(x.clone() >> 0, x.clone());
        assert_eq!(x.clone() >> 1, 0x4000_u64.into());
        assert_eq!(x.clone() >> 15, 0x0001_u64.into());
    }

    #[test]
    #[should_panic(expected = "shl overflow")]
    fn shl_panics_on_overflow() {
        let x = BoxedUint::from(0x0001_u64);
        let _ = x << 64;
    }

    #[test]
    fn checked_operations() {
        let a = BoxedUint::from(10_u64);
        let b = BoxedUint::from(5_u64);

        assert_eq!(a.checked_add(&b), Some(BoxedUint::from(15_u64)));
        assert_eq!(a.checked_sub(&b), Some(BoxedUint::from(5_u64)));
        assert_eq!(a.checked_mul(&b), Some(BoxedUint::from(50_u64)));
        assert_eq!(a.checked_rem(&b), Some(BoxedUint::zero()));

        // Test underflow
        assert!(b.checked_sub(&a).is_none());
    }

    #[allow(clippy::op_ref)]
    #[test]
    fn reference_operations() {
        let a = BoxedUint::from(10_u64);
        let b = BoxedUint::from(5_u64);

        // Test reference-based addition
        let c = a.clone() + &b;
        assert_eq!(c, BoxedUint::from(15_u64));

        // Test reference-based subtraction
        let d = a.clone() - &b;
        assert_eq!(d, BoxedUint::from(5_u64));

        // Test reference-based multiplication
        let e = a.clone() * &b;
        assert_eq!(e, BoxedUint::from(50_u64));

        // Test reference-based remainder
        let f = a.clone() % &b;
        assert_eq!(f, BoxedUint::zero());
    }

    #[test]
    fn conversions() {
        // Test From<crypto_bigint::BoxedUint> for Uint
        let original = crypto_bigint::BoxedUint::from(123_u64);
        let wrapped: BoxedUint = original.clone().into();
        assert_eq!(wrapped.0, original);

        // Test From<BoxedUint> for crypto_bigint::BoxedUint
        let wrapped = BoxedUint::from(456_u64);
        let unwrapped: crypto_bigint::BoxedUint = wrapped.into();
        assert_eq!(unwrapped, crypto_bigint::BoxedUint::from(456_u64));

        // Test conversion methods
        let value = crypto_bigint::BoxedUint::from(789_u64);
        let wrapped = BoxedUint::new(value.clone());
        assert_eq!(wrapped.inner(), &value);
        assert_eq!(wrapped.into_inner(), value);

        assert_eq!(BoxedUint::from(true), BoxedUint::one());
        assert_eq!(BoxedUint::from(Boolean::TRUE), BoxedUint::one());
    }

    #[test]
    fn pow_operation() {
        // Test basic exponentiation
        let base = BoxedUint::from(2_u64);

        // 2^0 = 1
        assert_eq!(base.clone().pow(0), BoxedUint::one());

        // 2^1 = 2
        assert_eq!(base.clone().pow(1), base);

        // 2^3 = 8
        assert_eq!(base.clone().pow(3), BoxedUint::from(8_u64));

        // 2^10 = 1024
        assert_eq!(base.clone().pow(10), BoxedUint::from(1024_u64));

        // Test with different base
        let base = BoxedUint::from(3_u64);

        // 3^4 = 81
        assert_eq!(base.pow(4), BoxedUint::from(81_u64));

        // Test with base 1
        let base = BoxedUint::from(1_u64);
        assert_eq!(base.pow(1000), BoxedUint::from(1_u64));

        // Test with base 0
        let base = BoxedUint::from(0_u64);
        assert_eq!(base.clone().pow(0), BoxedUint::one()); // 0^0 = 1 by convention
        assert_eq!(base.clone().pow(10), BoxedUint::zero()); // 0^n = 0 for n > 0
    }

    #[test]
    fn rem_assign_operations() {
        // Test RemAssign with owned value
        let mut a = BoxedUint::from(17_u64);
        let b = BoxedUint::from(5_u64);
        a %= b;
        assert_eq!(a, BoxedUint::from(2_u64));

        // Test RemAssign with reference
        let mut c = BoxedUint::from(19_u64);
        let d = BoxedUint::from(6_u64);
        c %= &d;
        assert_eq!(c, BoxedUint::from(1_u64));

        // Test with divisor 1
        let mut e = BoxedUint::from(42_u64);
        let one = BoxedUint::one();
        e %= &one;
        assert_eq!(e, BoxedUint::zero());
    }

    #[test]
    #[should_panic(expected = "division by zero")]
    fn rem_assign_panics_on_zero_divisor() {
        let mut a = BoxedUint::from(10_u64);
        let zero = BoxedUint::zero();
        a %= zero;
    }

    #[test]
    fn resize_method() {
        // Test resizing to same size
        let a = BoxedUint::from(0x12345678_u64).resize(WORD_BITS_FACTOR);
        let resized_same = a.resize(4 * WORD_BITS_FACTOR);
        assert_eq!(resized_same, a);

        // Test resizing to larger size
        let b = BoxedUint::from(0x9ABCDEF0_u64).resize(2 * WORD_BITS_FACTOR);
        let resized_larger = b.resize(4 * WORD_BITS_FACTOR);
        assert_eq!(resized_larger.as_words()[0], b.as_words()[0]);
        assert_eq!(resized_larger.as_words()[1], b.as_words()[1]);
        assert_eq!(resized_larger.as_words()[2], 0);
        assert_eq!(resized_larger.as_words()[3], 0);

        // Test resizing to smaller size (truncation)
        let c = BoxedUint::from(0x1234567890ABCDEF1234567890ABCDEF_u128);
        let resized_smaller = c.resize(1 * WORD_BITS_FACTOR);
        for (w1, w2) in c.as_words().iter().zip(resized_smaller.as_words().iter()) {
            assert_eq!(w1, w2);
        }
        let resized_back = resized_smaller.resize(2 * WORD_BITS_FACTOR);
        for i in 0..WORD_FACTOR {
            assert_eq!(resized_back.as_words()[i], c.as_words()[i]);
        }
        for i in 0..WORD_FACTOR {
            assert_eq!(resized_back.as_words()[i + WORD_FACTOR], 0);
        }
    }

    #[test]
    fn from_words() {
        // Test with single limb
        let words = [0x1234567890ABCDEF];
        let a = BoxedUint::from_words(words).resize(4 * WORD_BITS_FACTOR);
        assert_eq!(a.as_words()[0], words[0]);

        // Test with multiple limbs
        let words = [
            0x1234567890ABCDEF,
            0xFEDCBA9876543210,
            0x0F0F0F0F0F0F0F0F,
            0xF0F0F0F0F0F0F0F0,
        ];
        let b = BoxedUint::from_words(words).resize(4 * WORD_BITS_FACTOR);
        let b_words = b.as_words();
        for i in 0..4 {
            assert_eq!(b_words[i], words[i]);
        }
    }

    #[test]
    fn aggregate_operations() {
        let values: Vec<BoxedUint> = [1_u64, 2_u64, 3_u64]
            .into_iter()
            .map(BoxedUint::from)
            .collect();
        assert_eq!(values.iter().sum::<BoxedUint>(), BoxedUint::from(6_u64));
        assert_eq!(
            values.into_iter().sum::<BoxedUint>(),
            BoxedUint::from(6_u64)
        );

        let values: Vec<BoxedUint> = [2_u64, 3_u64, 4_u64]
            .into_iter()
            .map(BoxedUint::from)
            .collect();
        assert_eq!(
            values.iter().product::<BoxedUint>(),
            BoxedUint::from(24_u64)
        );
    }

    #[test]
    fn from_primitive_edge_cases() {
        for value in [u32::MIN, u32::MAX] {
            let i = BoxedUint::from(value).resize(1 * WORD_BITS_FACTOR);
            let j = BoxedUint::from(value).resize(2 * WORD_BITS_FACTOR);
            assert_eq!(i.resize(2 * WORD_BITS_FACTOR), j);
        }

        for value in [u64::MIN, u64::MAX] {
            let i = BoxedUint::from(value).resize(1 * WORD_BITS_FACTOR);
            let j = BoxedUint::from(value).resize(2 * WORD_BITS_FACTOR);
            assert_eq!(i.resize(2 * WORD_BITS_FACTOR), j);
        }

        for value in [u128::MIN, u128::MAX] {
            let i = BoxedUint::from(value).resize(2 * WORD_BITS_FACTOR);
            let j = BoxedUint::from(value).resize(3 * WORD_BITS_FACTOR);
            assert_eq!(i.resize(3 * WORD_BITS_FACTOR), j);
        }
    }

    #[should_panic]
    #[test]
    fn from_too_large_primitive() {
        // Test from_u128
        let _ = BoxedUint::from(u128::MAX)
            .try_resize(1 * WORD_BITS_FACTOR)
            .unwrap();
    }

    #[test]
    fn edge_cases() {
        // Test operations with MAX values
        let max = max(4);
        let one = BoxedUint::one();

        // MAX + 1 should overflow in checked_add
        assert!(max.checked_add(&one).is_none());

        // MAX - MAX = 0
        assert_eq!(max.checked_sub(&max).unwrap(), BoxedUint::zero());

        // Test operations with MIN values (0 for unsigned)
        let min = BoxedUint::zero();

        // MIN - 1 should overflow in checked_sub
        assert!(min.checked_sub(&one).is_none());

        // Test operations with large shifts
        let x = BoxedUint::from(1_u64).resize(4 * WORD_BITS_FACTOR);

        // Shift left by almost the bit limit
        let shifted = x.clone() << (x.bits_precision() - 1);
        let expected = {
            let len = 4 * WORD_FACTOR;
            let mut expected_words = vec![0; len];
            expected_words[len - 1] = (1 as Word) << (Word::BITS - 1);
            BoxedUint::from_words(expected_words)
        };
        assert_eq!(shifted, expected);

        // Test with large powers that don't overflow
        let two = BoxedUint::from(2_u64).resize(256);
        let large_power = two.pow(100); // 2^100 is large but fits in 256 bits

        // 2^100 should be divisible by 2^10 = 1024 with no remainder
        assert_eq!(
            large_power.clone() % BoxedUint::from(1024_u64),
            BoxedUint::zero()
        );

        // 2^100 / 2 = 2^99
        let half_power = large_power.clone() >> 1;
        assert_eq!(half_power << 1, large_power);
    }

    #[test]
    fn assign_operations() {
        // Test AddAssign
        let mut a = BoxedUint::from(10_u64);
        a += BoxedUint::from(5_u64);
        assert_eq!(a, BoxedUint::from(15_u64));

        let mut b = BoxedUint::from(20_u64);
        b += &BoxedUint::from(3_u64);
        assert_eq!(b, BoxedUint::from(23_u64));

        // Test SubAssign
        let mut c = BoxedUint::from(10_u64);
        c -= BoxedUint::from(3_u64);
        assert_eq!(c, BoxedUint::from(7_u64));

        let mut d = BoxedUint::from(50_u64);
        d -= &BoxedUint::from(25_u64);
        assert_eq!(d, BoxedUint::from(25_u64));

        // Test MulAssign
        let mut e = BoxedUint::from(7_u64);
        e *= BoxedUint::from(6_u64);
        assert_eq!(e, BoxedUint::from(42_u64));

        let mut f = BoxedUint::from(3_u64);
        f *= &BoxedUint::from(4_u64);
        assert_eq!(f, BoxedUint::from(12_u64));

        let mut f = BoxedUint::from(2_u64).resize(1 * WORD_BITS_FACTOR);
        f <<= 2;
        assert_eq!(f, BoxedUint::from(8_u64).resize(1 * WORD_BITS_FACTOR)); // 2 << 2 = 8
        f <<= 61;
        assert_eq!(f, BoxedUint::zero());

        let mut f = BoxedUint::from(3_u64).resize(1 * WORD_BITS_FACTOR);
        f >>= 1;
        assert_eq!(f, BoxedUint::from(1_u64).resize(1 * WORD_BITS_FACTOR)); // 3 >> 1 = 1
        f >>= 1;
        assert_eq!(f, BoxedUint::zero());
    }

    #[test]
    fn formatting() {
        let a = BoxedUint::from(255_u64).resize(1 * WORD_BITS_FACTOR);
        let b = max(1);

        // Test Debug
        assert_eq!(format!("{:?}", a), "BoxedUint(0x00000000000000FF)");
        assert_eq!(format!("{:?}", b), "BoxedUint(0xFFFFFFFFFFFFFFFF)");

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
        let default_val: BoxedUint = Default::default();
        assert_eq!(default_val, BoxedUint::zero());
        assert!(default_val.is_zero());
    }

    #[test]
    fn hash_agrees_with_eq() {
        let a = BoxedUint::one();
        let b = BoxedUint::one_with_precision(4 * WORD_BITS_FACTOR);
        assert_eq!(a, b);
        assert_eq!(hash_of(&a), hash_of(&b));

        let c = BoxedUint::from(12345_u64);
        let d = c.resize(4 * WORD_BITS_FACTOR);
        assert_eq!(c, d);
        assert_eq!(hash_of(&c), hash_of(&d));

        let e = BoxedUint::from(2_u64).resize(4 * WORD_BITS_FACTOR).pow(0);
        assert_eq!(e, BoxedUint::one());
        assert_eq!(hash_of(&e), hash_of(&BoxedUint::one()));
    }

    #[test]
    fn cmp_vartime() {
        let a = BoxedUint::from(10_u64);
        let b = BoxedUint::from(20_u64);
        let c = BoxedUint::from(10_u64);

        assert_eq!(a.cmp_vartime(&b), Ordering::Less);
        assert_eq!(b.cmp_vartime(&a), Ordering::Greater);
        assert_eq!(a.cmp_vartime(&c), Ordering::Equal);
    }

    #[test]
    fn cross_size_conversions() {
        // Test resize methods
        let a = BoxedUint::from(12345_u64).resize(2 * WORD_BITS_FACTOR);
        let b: Option<BoxedUint> = a.try_resize(4 * WORD_BITS_FACTOR);
        assert_eq!(b, Some(BoxedUint::from(12345_u64)));
        let b: BoxedUint = a.resize(4 * WORD_BITS_FACTOR);
        assert_eq!(b, BoxedUint::from(12345_u64));

        let a = BoxedUint::from(u128::MAX).resize(2 * WORD_BITS_FACTOR);
        let b = a.try_resize(1 * WORD_BITS_FACTOR);
        assert_eq!(b, None);
        let b = a.resize(1 * WORD_BITS_FACTOR);
        assert_eq!(b, max(1));

        // Test From<&crypto_bigint::BoxedUint> for BoxedUint
        let c = crypto_bigint::BoxedUint::from(67890_u64).resize(2 * WORD_BITS_FACTOR);
        let d: BoxedUint = c.into();
        assert_eq!(d, BoxedUint::from(67890_u64));

        // Test reference conversions from primitives
        let val = 42_u32;
        let e = BoxedUint::from(&val);
        assert_eq!(e, BoxedUint::from(42_u64));
    }

    #[test]
    fn constant_time_traits() {
        use crypto_bigint::{Choice, CtEq, CtGt, CtLt, CtSelect};

        let a = BoxedUint::from(10_u64);
        let b = BoxedUint::from(20_u64);
        let c = BoxedUint::from(10_u64);

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
    fn from_str() {
        // Test parsing from string
        let a: BoxedUint = "123".parse().unwrap();
        assert_eq!(a, BoxedUint::from(123_u64));

        let c: BoxedUint = "0".parse().unwrap();
        assert_eq!(c, BoxedUint::zero());

        let d: BoxedUint = "1".parse().unwrap();
        assert_eq!(d, BoxedUint::one());
        assert_eq!(u64::MAX.to_string().parse::<BoxedUint>().unwrap(), max(1));

        assert_eq!(
            "0xFF".parse::<BoxedUint>().unwrap(),
            BoxedUint::from(255_u64)
        );

        // Test invalid cases
        assert!("abc".parse::<BoxedUint>().is_err());
        assert!("12.34".parse::<BoxedUint>().is_err());
        assert!("".parse::<BoxedUint>().is_err());
        assert!("-456".parse::<BoxedUint>().is_err()); // Negative not allowed for unsigned

        // Number doesn't fit 1-word
        assert_eq!(
            ((u64::MAX as u128) + 1)
                .to_string()
                .parse::<BoxedUint>()
                .unwrap()
                .bytes_precision(),
            2 * WORD_BITS_FACTOR as usize / 8
        );
    }

    #[cfg(feature = "rand")]
    #[test]
    fn random_generation() {
        use rand::prelude::*;

        // Test crypto_bigint::Random trait
        let mut rng = StdRng::seed_from_u64(1);
        let random1: BoxedUint = crypto_bigint::RandomBits::random_bits(&mut rng, 32);
        let random2: BoxedUint = crypto_bigint::RandomBits::random_bits(&mut rng, 32);
        assert_ne!(random1, random2);
    }

    #[test]
    fn wrapping_operations() {
        // WrappingAdd
        let max = max(1);
        let zero = BoxedUint::zero();
        let one = BoxedUint::one();
        let wrapped_add = max.wrapping_add(&one);
        assert_eq!(wrapped_add, BoxedUint::zero()); // MAX + 1 wraps to 0

        let wrapped_add2 = max.wrapping_add(&max);
        assert_eq!(wrapped_add2, max.clone() - &one); // MAX + MAX wraps

        // WrappingSub
        let wrapped_sub = zero.wrapping_sub(&one);
        assert_eq!(wrapped_sub, max); // 0 - 1 wraps to MAX

        let wrapped_sub2 = one.wrapping_sub(&BoxedUint::from(2_u64));
        assert_eq!(wrapped_sub2, max); // 1 - 2 wraps to MAX

        // WrappingMul
        let large = max.clone();
        let two = BoxedUint::from(2_u64);
        let wrapped_mul = large.wrapping_mul(&two);
        // u64::MAX * 2 = 2 * (2^64 - 1) = 2^65 - 2, which wraps to 2^64 - 2 = MAX - 1
        assert_eq!(wrapped_mul, max.clone() - &one);

        // Non-overflowing cases should work normally
        let a = BoxedUint::from(10_u64);
        let b = BoxedUint::from(5_u64);
        assert_eq!(a.wrapping_add(&b), BoxedUint::from(15_u64));
        assert_eq!(a.wrapping_sub(&b), BoxedUint::from(5_u64));
        assert_eq!(a.wrapping_mul(&b), BoxedUint::from(50_u64));
    }
}
