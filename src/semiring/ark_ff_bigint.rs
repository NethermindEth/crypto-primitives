use super::*;
use crate::{boolean::Boolean, impl_pow_via_repeated_squaring};
use alloc::{format, vec::Vec};
use ark_ff::{BigInt as ArkBigInt, BigInteger as ArkBigInteger};
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};
use core::{
    fmt::{Debug, Display, Formatter, Result as FmtResult, UpperHex},
    hash::{Hash, Hasher},
    iter::{Product, Sum},
    ops::{
        Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Mul,
        MulAssign, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign,
    },
    str::FromStr,
};
use num_traits::{CheckedAdd, CheckedMul, CheckedSub, ConstOne, ConstZero, Num, One, Pow, Zero};
#[cfg(feature = "rand")]
use rand::{distr::StandardUniform, prelude::*};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct BigInt<const N: usize>(ArkBigInt<N>);

impl<const N: usize> BigInt<N> {
    /// Size of the underlying limbs in bits
    #[allow(clippy::cast_possible_truncation)]
    pub const BITS: u32 = Self::BYTES as u32 * 8;
    /// Size of the underlying limbs in bytes
    pub const BYTES: usize = Self::NUM_LIMBS * 8;
    /// Number of u64 limbs in the underlying representation
    pub const LIMBS: usize = N;

    /// Wraps a given value into this wrapper type
    #[inline(always)]
    pub const fn new(value: ArkBigInt<N>) -> Self {
        Self(value)
    }

    #[inline(always)]
    pub const fn new_ref(value: &ArkBigInt<N>) -> &Self {
        // Safety: BigInt is #[repr(transparent)] and is guaranteed to have the
        // same memory layout as ArkBigInt
        unsafe { &*(value as *const ArkBigInt<N> as *const Self) }
    }

    #[inline(always)]
    pub const fn new_ref_mut(value: &mut ArkBigInt<N>) -> &mut Self {
        // Safety: BigInt is #[repr(transparent)] and is guaranteed to have the
        // same memory layout as ArkBigInt
        unsafe { &mut *(value as *mut ArkBigInt<N> as *mut Self) }
    }

    /// Get the reference to the wrapped value
    #[inline(always)]
    pub const fn inner(&self) -> &ArkBigInt<N> {
        &self.0
    }

    /// Get the wrapped value, consuming self
    #[inline(always)]
    pub const fn into_inner(self) -> ArkBigInt<N> {
        self.0
    }

    /// See [ArkBigInteger::new]
    #[inline(always)]
    pub const fn from_limbs(arr: [u64; N]) -> Self {
        Self(ArkBigInt::new(arr))
    }

    pub const fn to_limbs(self) -> [u64; N] {
        self.0.0
    }

    pub const fn as_limbs(&self) -> &[u64; N] {
        &self.0.0
    }

    pub const fn as_mut_limbs(&mut self) -> &mut [u64; N] {
        &mut self.0.0
    }

    /// See [ArkBigInteger::from_bits_le]
    pub fn from_bits_le(bits: &[bool]) -> Self {
        Self(ArkBigInt::<N>::from_bits_le(bits))
    }

    /// See [ArkBigInteger::from_bits_be]
    pub fn from_bits_be(bits: &[bool]) -> Self {
        Self(ArkBigInt::<N>::from_bits_be(bits))
    }

    fn checked_add_assign_helper(&mut self, other: &Self) -> Result<(), ()> {
        let overflow = self.0.add_with_carry(&other.0);
        if overflow { Err(()) } else { Ok(()) }
    }

    fn checked_sub_assign_helper(&mut self, other: &Self) -> Result<(), ()> {
        let overflow = self.0.sub_with_borrow(&other.0);
        if overflow { Err(()) } else { Ok(()) }
    }
}

//
// Core traits
//

impl<const N: usize> AsRef<[u64]> for BigInt<N> {
    #[inline]
    fn as_ref(&self) -> &[u64] {
        self.0.as_ref()
    }
}

impl<const N: usize> AsMut<[u64]> for BigInt<N> {
    #[inline]
    fn as_mut(&mut self) -> &mut [u64] {
        self.0.as_mut()
    }
}

impl<const N: usize> Debug for BigInt<N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        Debug::fmt(&self.0, f)
    }
}

impl<const N: usize> Display for BigInt<N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        Display::fmt(&self.0, f)
    }
}

impl<const N: usize> Default for BigInt<N> {
    #[inline(always)]
    fn default() -> Self {
        Self::ZERO
    }
}

impl<const N: usize> UpperHex for BigInt<N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        UpperHex::fmt(&self.0, f)
    }
}

impl<const N: usize> Hash for BigInt<N> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state)
    }
}

impl<const N: usize> FromStr for BigInt<N> {
    type Err = <ArkBigInt<N> as FromStr>::Err;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (radix, s) = if let Some(s) = s.strip_prefix("0x") {
            (16, s)
        } else {
            (10, s)
        };
        let uint = num_bigint::BigUint::from_str_radix(s, radix).map_err(|_| ())?;
        ArkBigInt::try_from(uint).map(Self)
    }
}

//
// Zero and One traits
//

impl<const N: usize> Zero for BigInt<N> {
    #[inline(always)]
    fn zero() -> Self {
        Self::ZERO
    }

    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl<const N: usize> One for BigInt<N> {
    #[inline(always)]
    fn one() -> Self {
        Self::ONE
    }
}

impl<const N: usize> ConstZero for BigInt<N> {
    const ZERO: Self = Self(ArkBigInt::zero());
}

impl<const N: usize> ConstOne for BigInt<N> {
    const ONE: Self = Self(ArkBigInt::one());
}

//
// Basic arithmetic operations
//

macro_rules! impl_basic_op_forward_to_assign {
    ($trait:ident, $method:ident, $assign_method:ident) => {
        impl<const N: usize> $trait for BigInt<N> {
            type Output = BigInt<N>;

            #[inline(always)]
            fn $method(self, rhs: BigInt<N>) -> Self::Output {
                self.$method(&rhs)
            }
        }

        impl<const N: usize> $trait<&Self> for BigInt<N> {
            type Output = BigInt<N>;

            #[inline(always)]
            fn $method(mut self, rhs: &BigInt<N>) -> Self::Output {
                self.$assign_method(rhs);
                self
            }
        }

        impl<const N: usize> $trait for &BigInt<N> {
            type Output = BigInt<N>;

            #[inline(always)]
            fn $method(self, rhs: &BigInt<N>) -> Self::Output {
                self.clone().$method(rhs)
            }
        }

        impl<const N: usize> $trait<BigInt<N>> for &BigInt<N> {
            type Output = BigInt<N>;

            #[inline(always)]
            fn $method(self, rhs: BigInt<N>) -> Self::Output {
                self.clone().$method(&rhs)
            }
        }
    };
}

impl_basic_op_forward_to_assign!(Add, add, add_assign);
impl_basic_op_forward_to_assign!(Sub, sub, sub_assign);
impl_basic_op_forward_to_assign!(Mul, mul, mul_assign);

impl<const N: usize> Shl<u32> for BigInt<N> {
    type Output = Self;

    #[inline(always)]
    fn shl(self, rhs: u32) -> Self::Output {
        Self(self.0.shl(rhs))
    }
}

impl<const N: usize> Shr<u32> for BigInt<N> {
    type Output = Self;

    #[inline(always)]
    fn shr(self, rhs: u32) -> Self::Output {
        Self(self.0.shr(rhs))
    }
}

impl<const N: usize> Pow<u32> for BigInt<N> {
    type Output = Self;

    impl_pow_via_repeated_squaring!();
}

//
// Checked arithmetic operations
//

impl<const N: usize> CheckedAdd for BigInt<N> {
    fn checked_add(&self, other: &Self) -> Option<Self> {
        let mut res = *self;
        res.checked_add_assign_helper(other).ok()?;
        Some(res)
    }
}

impl<const N: usize> CheckedSub for BigInt<N> {
    fn checked_sub(&self, other: &Self) -> Option<Self> {
        let mut res = *self;
        res.checked_sub_assign_helper(other).ok()?;
        Some(res)
    }
}

impl<const N: usize> CheckedMul for BigInt<N> {
    fn checked_mul(&self, other: &Self) -> Option<Self> {
        // Use widening_mul which returns (lo, hi)
        let (lo, hi) = self.0.mul(&other.0);
        if hi.is_zero() { Some(Self(lo)) } else { None }
    }
}

//
// Arithmetic assign operations
//

macro_rules! impl_op_assign_boilerplate {
    ($trait:ident, $method:ident) => {
        impl<const N: usize> $trait for BigInt<N> {
            #[inline(always)]
            fn $method(&mut self, rhs: Self) {
                self.$method(&rhs);
            }
        }
    };
}

impl_op_assign_boilerplate!(AddAssign, add_assign);
impl_op_assign_boilerplate!(SubAssign, sub_assign);
impl_op_assign_boilerplate!(MulAssign, mul_assign);

impl<const N: usize> AddAssign<&Self> for BigInt<N> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: &Self) {
        self.checked_add_assign_helper(rhs)
            .expect("addition overflow")
    }
}

impl<const N: usize> SubAssign<&Self> for BigInt<N> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: &Self) {
        self.checked_sub_assign_helper(rhs)
            .expect("subtraction overflow")
    }
}

impl<const N: usize> MulAssign<&Self> for BigInt<N> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: &Self) {
        *self = self.checked_mul(rhs).expect("subtraction overflow")
    }
}

impl<const N: usize> ShlAssign<u32> for BigInt<N> {
    #[inline(always)]
    fn shl_assign(&mut self, rhs: u32) {
        self.0.shl_assign(rhs);
    }
}

impl<const N: usize> ShrAssign<u32> for BigInt<N> {
    #[inline(always)]
    fn shr_assign(&mut self, rhs: u32) {
        self.0.shr_assign(rhs);
    }
}

//
// Bitwise operations
//

macro_rules! impl_bitwise_op {
    ($trait:ident, $method:ident, $assign_trait:ident, $assign_method:ident) => {
        impl<const N: usize> $assign_trait for BigInt<N> {
            #[inline(always)]
            fn $assign_method(&mut self, rhs: Self) {
                self.0.$assign_method(rhs.0);
            }
        }

        impl<const N: usize> $assign_trait<&Self> for BigInt<N> {
            #[inline(always)]
            fn $assign_method(&mut self, rhs: &Self) {
                self.0.$assign_method(&rhs.0);
            }
        }

        impl<const N: usize> $trait for BigInt<N> {
            type Output = BigInt<N>;

            #[inline(always)]
            fn $method(mut self, rhs: BigInt<N>) -> Self::Output {
                self.$assign_method(rhs);
                self
            }
        }

        impl<const N: usize> $trait<&Self> for BigInt<N> {
            type Output = BigInt<N>;

            #[inline(always)]
            fn $method(mut self, rhs: &BigInt<N>) -> Self::Output {
                self.$assign_method(rhs);
                self
            }
        }
    };
}

impl_bitwise_op!(BitAnd, bitand, BitAndAssign, bitand_assign);
impl_bitwise_op!(BitOr, bitor, BitOrAssign, bitor_assign);
impl_bitwise_op!(BitXor, bitxor, BitXorAssign, bitxor_assign);

//
// Aggregate operations
//

impl<const N: usize> Sum for BigInt<N> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| {
            acc.checked_add(&x).expect("overflow in sum")
        })
    }
}

impl<'a, const N: usize> Sum<&'a Self> for BigInt<N> {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| {
            acc.checked_add(x).expect("overflow in sum")
        })
    }
}

impl<const N: usize> Product for BigInt<N> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| {
            acc.checked_mul(&x).expect("overflow in product")
        })
    }
}

impl<'a, const N: usize> Product<&'a Self> for BigInt<N> {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| {
            acc.checked_mul(x).expect("overflow in product")
        })
    }
}

//
// Conversions
//

impl<const N: usize> From<ArkBigInt<N>> for BigInt<N> {
    #[inline(always)]
    fn from(value: ArkBigInt<N>) -> Self {
        Self(value)
    }
}

impl<const N: usize> From<BigInt<N>> for ArkBigInt<N> {
    #[inline(always)]
    fn from(value: BigInt<N>) -> Self {
        value.0
    }
}

impl<const N: usize> From<bool> for BigInt<N> {
    #[inline(always)]
    fn from(value: bool) -> Self {
        Self(ArkBigInt::<N>::from(u8::from(value)))
    }
}

impl<const N: usize> From<Boolean> for BigInt<N> {
    #[inline(always)]
    fn from(value: Boolean) -> Self {
        Self::from(value.into_inner())
    }
}

macro_rules! impl_from_primitive {
    ($($t:ty),+) => {
        $(
            impl<const N: usize> From<$t> for BigInt<N> {
                fn from(value: $t) -> Self {
                    Self(ArkBigInt::<N>::from(value))
                }
            }

            impl<'a, const N: usize> From<&'a $t> for BigInt<N> {
                #[inline(always)]
                fn from(value: &$t) -> Self {
                    Self::from(*value)
                }
            }
        )+
    };
}

impl_from_primitive!(u8, u16, u32, u64);

impl<const N: usize> From<BigInt<N>> for num_bigint::BigUint {
    #[inline]
    fn from(value: BigInt<N>) -> num_bigint::BigUint {
        num_bigint::BigUint::from(value.0)
    }
}

impl<const N: usize> TryFrom<num_bigint::BigUint> for BigInt<N> {
    type Error = ();

    #[inline]
    fn try_from(value: num_bigint::BigUint) -> Result<Self, Self::Error> {
        ArkBigInt::<N>::try_from(value).map(Self)
    }
}

//
// Semiring
//

impl<const N: usize> Semiring for BigInt<N> {}

impl<const N: usize> ConstSemiring for BigInt<N> {
    const MAX: Self = Self(ArkBigInt([u64::MAX; N]));
    const MIN: Self = Self::ZERO;
}

impl<const N: usize> IntSemiring for BigInt<N> {
    fn is_odd(&self) -> bool {
        self.0.is_odd()
    }

    fn is_even(&self) -> bool {
        self.0.is_even()
    }
}

//
// RNG
//

impl<const N: usize> ark_std::rand::distributions::Distribution<BigInt<N>>
    for ark_std::rand::distributions::Standard
{
    fn sample<R: ark_std::rand::Rng + ?Sized>(&self, rng: &mut R) -> BigInt<N> {
        BigInt(rng.sample(ark_std::rand::distributions::Standard))
    }
}

#[cfg(feature = "rand")]
impl<const N: usize> Distribution<BigInt<N>> for StandardUniform {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> BigInt<N> {
        BigInt(ArkBigInt(rng.random()))
    }
}

//
// Serialization and Deserialization
//

#[cfg(feature = "serde")]
impl<'de, const N: usize> serde::Deserialize<'de> for BigInt<N> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // Serde does not support deserializing arrays of arbitrary length, work around
        // that with Vec
        use serde::de::Error;
        let vec = Vec::<u64>::deserialize(deserializer)?;
        let arr = vec
            .try_into()
            .map_err(|vec: Vec<_>| D::Error::invalid_length(vec.len(), &format!("{N}").as_str()))?;
        Ok(Self(ArkBigInt(arr)))
    }
}

#[cfg(feature = "serde")]
impl<const N: usize> serde::Serialize for BigInt<N> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.0.0.serialize(serializer)
    }
}

impl<const N: usize> CanonicalSerialize for BigInt<N> {
    #[inline]
    fn serialize_with_mode<W: ark_serialize::Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.0.serialize_with_mode(writer, compress)
    }

    #[inline]
    fn serialized_size(&self, compress: Compress) -> usize {
        self.0.serialized_size(compress)
    }
}

impl<const N: usize> Valid for BigInt<N> {
    #[inline]
    fn check(&self) -> Result<(), SerializationError> {
        self.0.check()
    }
}

impl<const N: usize> CanonicalDeserialize for BigInt<N> {
    #[inline]
    fn deserialize_with_mode<R: ark_serialize::Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        ArkBigInt::<N>::deserialize_with_mode(reader, compress, validate).map(Self)
    }
}

//
// Zeroize
//

// ark-ff unconditionally requires zeroize
impl<const N: usize> zeroize::DefaultIsZeroes for BigInt<N> {}

//
// Traits from ark_ff
//

impl<const N: usize> ArkBigInteger for BigInt<N> {
    const NUM_LIMBS: usize = N;

    #[inline]
    fn add_with_carry(&mut self, other: &Self) -> bool {
        self.0.add_with_carry(&other.0)
    }

    #[inline]
    fn sub_with_borrow(&mut self, other: &Self) -> bool {
        self.0.sub_with_borrow(&other.0)
    }

    #[inline]
    fn mul2(&mut self) -> bool {
        self.0.mul2()
    }

    #[allow(deprecated)]
    #[inline]
    fn muln(&mut self, amt: u32) {
        self.0.muln(amt)
    }

    #[inline]
    fn mul_low(&self, other: &Self) -> Self {
        Self(self.0.mul_low(&other.0))
    }

    #[inline]
    fn mul_high(&self, other: &Self) -> Self {
        Self(self.0.mul_high(&other.0))
    }

    #[inline]
    fn mul(&self, other: &Self) -> (Self, Self) {
        let (lo, hi) = self.0.mul(&other.0);
        (Self(lo), Self(hi))
    }

    #[inline]
    fn div2(&mut self) {
        self.0.div2()
    }

    #[allow(deprecated)]
    #[inline]
    fn divn(&mut self, amt: u32) {
        self.0.divn(amt)
    }

    #[inline]
    fn is_odd(&self) -> bool {
        self.0.is_odd()
    }

    #[inline]
    fn is_even(&self) -> bool {
        self.0.is_even()
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }

    #[inline]
    fn num_bits(&self) -> u32 {
        self.0.num_bits()
    }

    #[inline]
    fn get_bit(&self, i: usize) -> bool {
        self.0.get_bit(i)
    }

    #[inline]
    fn from_bits_be(bits: &[bool]) -> Self {
        Self(ArkBigInt::from_bits_be(bits))
    }

    #[inline]
    fn from_bits_le(bits: &[bool]) -> Self {
        Self(ArkBigInt::from_bits_le(bits))
    }

    #[inline]
    fn to_bits_be(&self) -> Vec<bool> {
        self.0.to_bits_be()
    }

    #[inline]
    fn to_bits_le(&self) -> Vec<bool> {
        self.0.to_bits_le()
    }

    #[inline]
    fn to_bytes_be(&self) -> Vec<u8> {
        self.0.to_bytes_be()
    }

    #[inline]
    fn to_bytes_le(&self) -> Vec<u8> {
        self.0.to_bytes_le()
    }

    #[inline]
    fn find_wnaf(&self, w: usize) -> Option<Vec<i64>> {
        self.0.find_wnaf(w)
    }
}

#[allow(clippy::arithmetic_side_effects, clippy::cast_lossless)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::ensure_type_implements_trait;
    use alloc::{format, string::ToString, vec::Vec};
    use ark_std::rand::Rng;
    use core::cmp::Ordering;

    type BigInt1 = BigInt<1>;
    type BigInt2 = BigInt<2>;
    type BigInt4 = BigInt<4>;

    #[test]
    fn ensure_blanket_traits() {
        ensure_type_implements_trait!(BigInt4, ConstIntSemiring);
        ensure_type_implements_trait!(BigInt4, IntSemiringWithShifts);
    }

    #[test]
    fn basic_operations() {
        let a = BigInt4::from(10_u64);
        let b = BigInt4::from(5_u64);

        // Test addition
        assert_eq!(a + b, BigInt4::from(15_u64));

        // Test subtraction
        assert_eq!(a - b, BigInt4::from(5_u64));

        // Test multiplication
        assert_eq!(a * b, BigInt4::from(50_u64));

        // Test shl
        let x = BigInt1::from(0x0001_u64);
        assert_eq!(x << 0, x);
        assert_eq!(x << 1, 0x0002_u64.into());
        assert_eq!(x << 15, 0x8000_u64.into());

        // Test shr
        let x = BigInt4::from(0x8000_u32);
        assert_eq!(x >> 0, x);
        assert_eq!(x >> 1, 0x4000_u64.into());
        assert_eq!(x >> 15, 0x0001_u64.into());
    }

    #[test]
    fn shl_does_not_panic_on_overflow() {
        // ark_ff::BigInt semantics is different from primitive types, it does not panic
        // on shl overflow.
        let x = BigInt1::from(0x0001_u64);
        let y = x << 64;
        assert_eq!(y, Zero::zero());
    }

    #[test]
    fn checked_operations() {
        let a = BigInt4::from(10_u64);
        let b = BigInt4::from(5_u64);

        assert_eq!(a.checked_add(&b), Some(BigInt4::from(15_u64)));
        assert_eq!(a.checked_sub(&b), Some(BigInt4::from(5_u64)));
        assert_eq!(a.checked_mul(&b), Some(BigInt4::from(50_u64)));

        // Test underflow
        assert!(b.checked_sub(&a).is_none());

        // MIN and MAX
        assert_eq!(BigInt4::MAX.checked_add(&One::one()), None);
        assert_eq!(BigInt4::MIN.checked_sub(&One::one()), None);
    }

    #[allow(clippy::op_ref)]
    #[test]
    fn reference_operations() {
        let a = BigInt4::from(10_u64);
        let b = BigInt4::from(5_u64);

        // Test reference-based addition
        let c = a + &b;
        assert_eq!(c, BigInt4::from(15_u64));

        // Test reference-based subtraction
        let d = a - &b;
        assert_eq!(d, BigInt4::from(5_u64));

        // Test reference-based multiplication
        let e = a * &b;
        assert_eq!(e, BigInt4::from(50_u64));
    }

    #[test]
    fn conversions() {
        // Test From<ArkBigInt> for BigInt
        let original = ArkBigInt::from(123_u64);
        let wrapped: BigInt4 = original.into();
        assert_eq!(wrapped.0, original);

        // Test From<BigInt> for ArkBigInt
        let wrapped = BigInt4::from(456_u64);
        let unwrapped: ArkBigInt<4> = wrapped.into();
        assert_eq!(unwrapped, ArkBigInt::from(456_u64));

        // Test conversion methods
        let value = ArkBigInt::from(789_u64);
        let wrapped = BigInt4::new(value);
        assert_eq!(wrapped.inner(), &value);
        assert_eq!(wrapped.into_inner(), value);

        assert_eq!(BigInt4::from(true), BigInt4::ONE);
        assert_eq!(BigInt4::from(Boolean::TRUE), BigInt4::ONE);
    }

    #[test]
    fn pow_operation() {
        // Test basic exponentiation
        let base = BigInt4::from(2_u64);

        // 2^0 = 1
        assert_eq!(base.pow(0), BigInt4::one());

        // 2^1 = 2
        assert_eq!(base.pow(1), base);

        // 2^3 = 8
        assert_eq!(base.pow(3), BigInt4::from(8_u64));

        // 2^10 = 1024
        assert_eq!(base.pow(10), BigInt4::from(1024_u64));

        // Test with different base
        let base = BigInt4::from(3_u64);

        // 3^4 = 81
        assert_eq!(base.pow(4), BigInt4::from(81_u64));

        // Test with base 1
        let base = BigInt4::from(1_u64);
        assert_eq!(base.pow(1000), BigInt4::from(1_u64));

        // Test with base 0
        let base = BigInt4::from(0_u64);
        assert_eq!(base.pow(0), BigInt4::one()); // 0^0 = 1 by convention
        assert_eq!(base.pow(10), BigInt4::zero()); // 0^n = 0 for n > 0
    }

    #[test]
    fn from_limbs() {
        // Test with single limb
        let limbs = [0x1234567890ABCDEF];
        let a = BigInt1::from_limbs(limbs);
        assert_eq!(a.to_limbs()[0], limbs[0]);

        // Test with multiple limbs
        let limbs = [
            0x1234567890ABCDEF,
            0xFEDCBA9876543210,
            0x0F0F0F0F0F0F0F0F,
            0xF0F0F0F0F0F0F0F0,
        ];
        let b = BigInt4::from_limbs(limbs);
        let b_limbs = b.to_limbs();
        for i in 0..4 {
            assert_eq!(b_limbs[i], limbs[i]);
        }
    }

    #[test]
    fn aggregate_operations() {
        let values: Vec<BigInt4> = [1_u64, 2_u64, 3_u64]
            .into_iter()
            .map(BigInt::from)
            .collect();
        assert_eq!(values.iter().sum::<BigInt4>(), BigInt4::from(6_u64));
        assert_eq!(values.into_iter().sum::<BigInt4>(), BigInt4::from(6_u64));

        let values: Vec<BigInt4> = [2_u64, 3_u64, 4_u64]
            .into_iter()
            .map(BigInt::from)
            .collect();
        assert_eq!(values.iter().product::<BigInt4>(), BigInt4::from(24_u64));
    }

    #[test]
    fn edge_cases() {
        // Test operations with MAX values
        let max = BigInt4::MAX;
        let one = BigInt4::one();

        // MAX + 1 should overflow in checked_add
        assert!(max.checked_add(&one).is_none());

        // MAX - MAX = 0
        assert_eq!(max.checked_sub(&max).unwrap(), BigInt4::zero());

        // Test operations with MIN values (0 for unsigned)
        let min = BigInt4::ZERO;

        // MIN - 1 should overflow in checked_sub
        assert!(min.checked_sub(&one).is_none());

        // Test operations with large shifts
        let x = BigInt4::from(1_u64);

        // Shift left by almost the bit limit
        let shifted = x << (BigInt4::BITS - 1);
        let expected = {
            let mut expected_limbs = [0; 4];
            expected_limbs[3] = 1 << 63;
            BigInt4::from_limbs(expected_limbs)
        };
        assert_eq!(shifted, expected);

        // Test with large powers that don't overflow
        let two = BigInt4::from(2_u64);
        let large_power = two.pow(100); // 2^100 is large but fits in 256 bits

        // 2^100 / 2 = 2^99
        let half_power = large_power >> 1;
        assert_eq!(half_power << 1, large_power);
    }

    #[test]
    fn assign_operations() {
        // Test AddAssign
        let mut a = BigInt4::from(10_u64);
        a += BigInt4::from(5_u64);
        assert_eq!(a, BigInt4::from(15_u64));

        let mut b = BigInt4::from(20_u64);
        b += &BigInt4::from(3_u64);
        assert_eq!(b, BigInt4::from(23_u64));

        // Test SubAssign
        let mut c = BigInt4::from(10_u64);
        c -= BigInt4::from(3_u64);
        assert_eq!(c, BigInt4::from(7_u64));

        let mut d = BigInt4::from(50_u64);
        d -= &BigInt4::from(25_u64);
        assert_eq!(d, BigInt4::from(25_u64));

        // Test MulAssign
        let mut e = BigInt4::from(7_u64);
        e *= BigInt4::from(6_u64);
        assert_eq!(e, BigInt4::from(42_u64));

        let mut f = BigInt4::from(3_u64);
        f *= &BigInt4::from(4_u64);
        assert_eq!(f, BigInt4::from(12_u64));

        let mut f = BigInt1::from(2_u64);
        f <<= 2;
        assert_eq!(f, BigInt1::from(8_u64)); // 2 << 2 = 8
        f <<= 61;
        assert_eq!(f, BigInt1::ZERO);

        let mut f = BigInt1::from(3_u64);
        f >>= 1;
        assert_eq!(f, BigInt1::from(1_u64)); // 3 >> 1 = 1
        f >>= 1;
        assert_eq!(f, BigInt1::ZERO);
    }

    #[test]
    fn formatting() {
        let a = BigInt1::from(255_u64);
        let b = BigInt1::MAX;

        // Test Display
        assert_eq!(format!("{}", a), "255");
        assert_eq!(format!("{}", b), format!("{}", u64::MAX));

        // Test UpperHex
        assert_eq!(format!("{:X}", a), "00000000000000FF");
        assert_eq!(format!("{:X}", b), "FFFFFFFFFFFFFFFF");
    }

    #[test]
    fn default_trait() {
        let default_val: BigInt4 = Default::default();
        assert_eq!(default_val, BigInt4::ZERO);
        assert!(Zero::is_zero(&default_val));
    }

    #[test]
    fn constants() {
        // Test MAX
        assert!(BigInt4::MAX > BigInt4::ZERO);

        // Test BITS, BYTES, LIMBS
        assert_eq!(BigInt4::BITS, 256);
        assert_eq!(BigInt4::BYTES, 32);
        assert_eq!(BigInt4::LIMBS, 4);

        assert_eq!(BigInt2::BITS, 128);
        assert_eq!(BigInt2::BYTES, 16);
        assert_eq!(BigInt2::LIMBS, 2);
    }

    #[test]
    fn cmp() {
        let a = BigInt4::from(10_u64);
        let b = BigInt4::from(20_u64);
        let c = BigInt4::from(10_u64);

        assert_eq!(a.cmp(&b), Ordering::Less);
        assert_eq!(b.cmp(&a), Ordering::Greater);
        assert_eq!(a.cmp(&c), Ordering::Equal);
    }

    #[test]
    fn from_str() {
        // Test parsing from string
        let a: BigInt4 = "123".parse().unwrap();
        assert_eq!(a, BigInt4::from(123_u64));

        let c: BigInt4 = "0".parse().unwrap();
        assert_eq!(c, BigInt4::zero());

        let d: BigInt4 = "1".parse().unwrap();
        assert_eq!(d, BigInt4::one());
        assert_eq!(
            u64::MAX.to_string().parse::<BigInt1>().unwrap(),
            BigInt1::MAX
        );

        assert_eq!("0xFF".parse::<BigInt4>().unwrap(), BigInt4::from(255_u64));

        // Test invalid cases
        assert!("abc".parse::<BigInt4>().is_err());
        assert!("12.34".parse::<BigInt4>().is_err());
        assert!("".parse::<BigInt4>().is_err());
        assert!("-456".parse::<BigInt4>().is_err()); // Negative not allowed for unsigned

        // Number doesn't fit BigInt1
        assert!(
            ((u64::MAX as u128) + 1)
                .to_string()
                .parse::<BigInt1>()
                .is_err()
        );
    }

    #[cfg(feature = "rand")]
    #[test]
    fn random_generation() {
        use rand::prelude::*;
        // Use a seeded RNG for reproducibility

        // Test Distribution trait
        let mut rng = StdRng::seed_from_u64(1);
        let random1: BigInt4 = rng.random();
        let random2: BigInt4 = rng.random();
        assert_ne!(random1, random2);

        // Test ark_std::rand trait
        let mut ark_rng = ark_std::rand::rngs::mock::StepRng::new(1, 1);
        let random1: BigInt4 = ark_rng.r#gen();
        let random2: BigInt4 = ark_rng.r#gen();
        assert_ne!(random1, random2);
    }
}
