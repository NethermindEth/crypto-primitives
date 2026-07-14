use super::*;
use crate::{IntRing, Wrapper, boolean::Boolean, crypto_bigint_int::Int, crypto_bigint_uint::Uint};
use core::fmt::{Display, Formatter, Result as FmtResult};
use crypto_bigint::{
    BitOps, Limb, NonZero, Odd, Word,
    modular::{FixedMontyForm, FixedMontyParams},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MontyField<const LIMBS: usize> {
    pub params: FixedMontyParams<LIMBS>,
    pub modulus_minus_one_div_two: Uint<LIMBS>,
}

impl<const LIMBS: usize> MontyField<LIMBS> {
    pub const LIMBS: usize = LIMBS;

    /// Creates a new [`MontyField`] from [`FixedMontyParams`].
    #[inline(always)]
    #[allow(clippy::arithmetic_side_effects)] // False alert
    pub const fn wrap(params: FixedMontyParams<LIMBS>) -> Self {
        let modulus = params.modulus().get_copy();
        let two = crypto_bigint::Uint::<LIMBS>::from_u8(2_u8)
            .to_nz()
            .to_inner_unchecked();
        let (tmp, _) = modulus.borrowing_sub(&crypto_bigint::Uint::ONE, Limb(0_u8 as Word));
        let modulus_minus_one_div_two = Uint::new(tmp.div_exact(&two).to_inner_unchecked());
        Self {
            params,
            modulus_minus_one_div_two,
        }
    }

    /// Unwrap the [`FixedMontyForm`] into a field config and a field element.
    pub const fn unwrap_monty(el: FixedMontyForm<LIMBS>) -> (Self, Uint<LIMBS>) {
        (Self::wrap(*el.params()), Uint::new(el.to_montgomery()))
    }

    /// Reassemble the `crypto-bigint`'s [`FixedMontyForm`] from a raw
    /// Montgomery value.
    #[inline(always)]
    pub const fn form(&self, el: &MontyFieldElement<LIMBS>) -> FixedMontyForm<LIMBS> {
        FixedMontyForm::from_montgomery(el.0.0, &self.params)
    }
}

/// A wrapper around [`Uint`] to prevent accidentally calling math operations
/// on it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct MontyFieldElement<const LIMBS: usize>(pub Uint<LIMBS>);

impl<const LIMBS: usize> From<Uint<LIMBS>> for MontyFieldElement<LIMBS> {
    #[inline(always)]
    fn from(value: Uint<LIMBS>) -> Self {
        Self(value)
    }
}

impl<const LIMBS: usize> From<crypto_bigint::Uint<LIMBS>> for MontyFieldElement<LIMBS> {
    #[inline(always)]
    fn from(value: crypto_bigint::Uint<LIMBS>) -> Self {
        Self(Uint::new(value))
    }
}

//
// Wrapper
//

impl<const LIMBS: usize> Wrapper for MontyFieldElement<LIMBS> {
    type Inner = Uint<LIMBS>;

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
// Core traits
//

impl<const LIMBS: usize> Display for MontyField<LIMBS> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "F_{}", self.params.modulus())
    }
}

//
// Field operations
//

impl<const LIMBS: usize> FieldConfig for MontyField<LIMBS> {
    type Element = MontyFieldElement<LIMBS>;

    fn is_zero(&self, value: &Self::Element) -> bool {
        value.0.inner().is_zero_vartime()
    }

    fn zero(&self) -> Self::Element {
        crypto_bigint::Uint::ZERO.into()
    }

    fn one(&self) -> Self::Element {
        (*self.params.one()).into()
    }

    fn neg(&self, x: &Self::Element) -> Self::Element {
        x.0.inner()
            .neg_mod(self.params.modulus().as_nz_ref())
            .into()
    }

    fn add(&self, x: &Self::Element, y: &Self::Element) -> Self::Element {
        x.0.inner()
            .add_mod(y.0.inner(), self.params.modulus().as_nz_ref())
            .into()
    }

    fn sub(&self, x: &Self::Element, y: &Self::Element) -> Self::Element {
        x.0.inner()
            .sub_mod(y.0.inner(), self.params.modulus().as_nz_ref())
            .into()
    }

    fn inv(&self, x: &Self::Element) -> Option<Self::Element> {
        // Follows FixedMontyForm::invert_vartime, but avoids the params copies
        // of assembling forms: invert the Montgomery representation directly,
        // then bring the plain inverse back into the Montgomery domain.
        // (xR)⁻¹ = x⁻¹R⁻¹, and two Montgomery multiplications by R² yield the
        // Montgomery form x⁻¹R.
        let modulus = self.params.modulus();
        let inverted: Option<crypto_bigint::Uint<LIMBS>> =
            x.0.inner().invert_odd_mod_vartime(modulus).into();
        let inverted = inverted?;

        let r2 = self.params.r2();
        let x_inv = crypto_bigint_helpers::mul::monty_mul(&inverted, r2, modulus.as_ref());
        Some(crypto_bigint_helpers::mul::monty_mul(&x_inv, r2, modulus.as_ref()).into())
    }

    fn mul(&self, x: &Self::Element, y: &Self::Element) -> Self::Element {
        crypto_bigint_helpers::mul::monty_mul(
            x.0.inner(),
            y.0.inner(),
            self.params.modulus().as_ref(),
        )
        .into()
    }

    fn pow(&self, x: &Self::Element, y: &Self::Integer) -> Self::Element {
        crypto_bigint_helpers::pow::pow_bounded_exp(
            x.0.inner(),
            y.inner().as_limbs(),
            y.inner().bits_precision(),
            *self.params.one(),
            self.params.modulus().as_ref(),
            self.params.mod_neg_inv(),
        )
        .into()
    }

    fn pow_u32(&self, x: &Self::Element, y: u32) -> Self::Element {
        crypto_bigint_helpers::pow::pow_u32(
            x.0.inner(),
            y,
            *self.params.one(),
            self.params.modulus().as_ref(),
            self.params.mod_neg_inv(),
        )
        .into()
    }

    // Assign operations use the default implementations, which delegate to the
    // operations above (`Uint` arithmetic is not in-place anyway)
}

//
// Conversions
//

macro_rules! impl_from_unsigned {
    ($($t:ty),* $(,)?) => {
        $(
            impl<const LIMBS: usize> ProjectElementDynamic<$t> for MontyField<LIMBS> {
                fn project(&self, value: &$t) -> Self::Element {
                    let abs: crypto_bigint::Uint<LIMBS> = (*value).into();
                    self.project(&Uint::new(abs))
                }
            }
        )*
    };
}

macro_rules! impl_from_signed {
    ($($t:ty),* $(,)?) => {
        $(
            impl<const LIMBS: usize> ProjectElementDynamic<$t> for MontyField<LIMBS> {
                fn project(&self, value: &$t) -> Self::Element {
                    let magnitude: crypto_bigint::Uint<LIMBS> = value.abs_diff(0).into();
                    let magnitude = self.project(&Uint::new(magnitude));
                    if value.is_negative() { self.neg(&magnitude) } else { magnitude }
                }
            }
        )*
    };
}

impl_from_unsigned!(u8, u16, u32, u64, u128);
impl_from_signed!(i8, i16, i32, i64, i128);

impl<const LIMBS: usize> ProjectElementDynamic<bool> for MontyField<LIMBS> {
    fn project(&self, value: &bool) -> Self::Element {
        if *value { self.one() } else { self.zero() }
    }
}

impl<const LIMBS: usize> ProjectElementDynamic<Boolean> for MontyField<LIMBS> {
    #[inline(always)]
    fn project(&self, value: &Boolean) -> Self::Element {
        self.project(value.inner())
    }
}

impl<const LIMBS: usize, const LIMBS2: usize> ProjectElementDynamic<Int<LIMBS2>>
    for MontyField<LIMBS>
{
    fn project(&self, value: &Int<LIMBS2>) -> Self::Element {
        let abs = self.project(&Uint::new(value.inner().abs()));
        if value.is_negative() {
            self.neg(&abs)
        } else {
            abs
        }
    }
}

impl<const LIMBS: usize, const LIMBS2: usize> ProjectElementDynamic<Uint<LIMBS2>>
    for MontyField<LIMBS>
{
    /// Convert the given integer into the Montgomery domain.
    fn project(&self, value: &Uint<LIMBS2>) -> Self::Element {
        let value: crypto_bigint::Uint<LIMBS> = if LIMBS >= LIMBS2 {
            value.inner().resize()
        } else {
            // The value is wider than the modulus: reduce it first, so that
            // the resize below cannot truncate.
            value
                .inner()
                .rem(&NonZero::<crypto_bigint::Uint<LIMBS2>>::new_unwrap(
                    self.params.modulus().as_ref().resize::<LIMBS2>(),
                ))
                .resize()
        };

        // Convert into the Montgomery domain
        crypto_bigint_helpers::mul::monty_mul(
            &value,
            self.params.r2(),
            self.params.modulus().as_ref(),
        )
        .into()
    }
}

//
// Field stuff
//

impl<const LIMBS: usize> BaseFieldConfig for MontyField<LIMBS> {
    fn new(modulus: &Self::Integer) -> Result<Self, FieldError> {
        let Some(modulus) = Odd::new(*modulus.inner()).into_option() else {
            return Err(FieldError::InvalidModulus);
        };
        Ok(Self::wrap(FixedMontyParams::new(modulus)))
    }

    #[inline(always)]
    fn modulus(&self) -> Self::Integer {
        Uint::new(self.params.modulus().get())
    }

    #[inline(always)]
    fn modulus_minus_one_div_two(&self) -> Self::Integer {
        self.modulus_minus_one_div_two
    }
}

impl<const LIMBS: usize> WithAssociatedInteger for MontyField<LIMBS> {
    type Integer = Uint<LIMBS>;
}

impl<const LIMBS: usize> LiftElementDynamic<<Self as WithAssociatedInteger>::Integer>
    for MontyField<LIMBS>
{
    /// Retrieves the integer currently encoded in this [`MontyField`],
    /// guaranteed to be reduced.
    #[inline(always)]
    fn lift(&self, value: &Self::Element) -> Self::Integer {
        let mut out = crypto_bigint::Uint::<LIMBS>::ZERO;
        crypto_bigint_helpers::monty_retrieve_inner(
            value.0.inner().as_limbs(),
            out.as_mut_limbs(),
            self.params.modulus().as_ref().as_limbs(),
            self.params.mod_neg_inv(),
        );
        Uint::new(out)
    }
}

// TODO: Do we want to zeroize the modulus?

//
// Predefined fields of various sizes for convenience
//

pub type F64 = MontyField<{ crypto_bigint::U64::LIMBS }>;
pub type F128 = MontyField<{ 2 * WORD_FACTOR }>;
pub type F192 = MontyField<{ 3 * WORD_FACTOR }>;
pub type F256 = MontyField<{ 4 * WORD_FACTOR }>;
pub type F320 = MontyField<{ 5 * WORD_FACTOR }>;
pub type F384 = MontyField<{ 6 * WORD_FACTOR }>;
pub type F448 = MontyField<{ 7 * WORD_FACTOR }>;
pub type F512 = MontyField<{ 8 * WORD_FACTOR }>;
pub type F576 = MontyField<{ 9 * WORD_FACTOR }>;
pub type F640 = MontyField<{ 10 * WORD_FACTOR }>;
pub type F704 = MontyField<{ 11 * WORD_FACTOR }>;
pub type F768 = MontyField<{ 12 * WORD_FACTOR }>;
pub type F832 = MontyField<{ 13 * WORD_FACTOR }>;
pub type F896 = MontyField<{ 14 * WORD_FACTOR }>;
pub type F960 = MontyField<{ 15 * WORD_FACTOR }>;
pub type F1024 = MontyField<{ 16 * WORD_FACTOR }>;
pub type F1280 = MontyField<{ 20 * WORD_FACTOR }>;
pub type F1536 = MontyField<{ 24 * WORD_FACTOR }>;
pub type F1792 = MontyField<{ 28 * WORD_FACTOR }>;
pub type F2048 = MontyField<{ 32 * WORD_FACTOR }>;
pub type F3072 = MontyField<{ 48 * WORD_FACTOR }>;
pub type F4096 = MontyField<{ 64 * WORD_FACTOR }>;
pub type F6144 = MontyField<{ 96 * WORD_FACTOR }>;
pub type F8192 = MontyField<{ 128 * WORD_FACTOR }>;
pub type F16384 = MontyField<{ 256 * WORD_FACTOR }>;
pub type F32768 = MontyField<{ 512 * WORD_FACTOR }>;

#[allow(
    clippy::arithmetic_side_effects,
    clippy::cast_lossless,
    clippy::redundant_clone,
    clippy::clone_on_copy
)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::ensure_type_implements_trait;
    use alloc::vec;
    use core::cmp::Ordering;
    use crypto_bigint::U64;

    const LIMBS: usize = 4;
    type F = F256;

    #[test]
    fn ensure_traits() {
        ensure_type_implements_trait!(F, BaseFieldConfig);
    }

    //
    // Test helpers
    //

    fn make_f() -> F {
        // Using a 256-bit prime 2^256 - 2^32 - 977 (secp256k1 field prime)
        let modulus = crypto_bigint::Uint::<LIMBS>::from_be_hex(
            "fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f",
        );
        let modulus = Odd::new(modulus).expect("modulus should be odd");
        F::wrap(FixedMontyParams::new(modulus))
    }

    #[test]
    fn new_with_cfg_correct() {
        let f = make_f();

        // `modulus + 1` should reduce to one.
        let x = Uint::<LIMBS>::from_be_hex(
            "fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc30",
        );
        assert_eq!(f.project(&x), f.one());

        // `modulus` itself should reduce to zero.
        let modulus = f.modulus();
        assert_eq!(f.project(&modulus), f.zero());

        // Lifting to integer and projecting back yields the original element.
        for x in [
            f.zero(),
            f.one(),
            f.project(&2_u64),
            f.project(&123456789_u64),
        ] {
            assert_eq!(f.project(&f.lift(&x)), x);
        }
    }

    #[test]
    fn zero_one_basics() {
        let f = make_f();
        let z = f.zero();
        assert!(f.is_zero(&z));
        let o = f.one();
        assert!(!f.is_zero(&o));
        assert_ne!(z, o);
    }

    #[test]
    fn basic_operations() {
        let f = make_f();

        // Negation
        let a = f.project(&9);
        let neg_a = f.neg(&a);
        assert_eq!(f.add(&a, &neg_a), f.zero());

        let a = f.project(&10);
        let b = f.project(&5);

        // Addition
        let c = f.add(&a, &b);
        assert_eq!(c, f.project(&15));

        // Subtraction
        let d = f.sub(&a, &b);
        assert_eq!(d, f.project(&5));

        // Multiplication
        let e = f.mul(&a, &b);
        assert_eq!(e, f.project(&50));

        // Division
        let num = f.project(&11);
        let den = f.project(&5);
        let q = f.div(&num, &den);
        assert_eq!(f.mul(&q, &den), num);
    }

    #[test]
    fn basic_operations_overflow() {
        let f = make_f();

        let mod_minus_one = Uint::new(f.params.modulus().get() - crypto_bigint::Uint::ONE);
        let mod_minus_one = f.project(&mod_minus_one);

        // Negation
        let res = f.neg(&mod_minus_one);
        assert_eq!(res, f.one());

        // Addition
        let res = f.add(&mod_minus_one, &f.one());
        assert_eq!(res, f.zero());

        // Subtraction
        let res = f.sub(&f.zero(), &f.one());
        assert_eq!(res, mod_minus_one);

        // Multiplication
        let res = f.mul(&mod_minus_one, &f.project(&2));
        assert_eq!(res, f.sub(&mod_minus_one, &f.one()));

        let res = f.mul(&mod_minus_one, &mod_minus_one);
        assert_eq!(res, f.one());

        // Division
        let res = f.div(&f.one(), &mod_minus_one);
        assert_eq!(res, mod_minus_one);
    }

    #[test]
    fn add_wrapping() {
        let f = make_f();

        let a = f.project(&-100);
        let b = f.project(&105);
        let c = f.add(&a, &b);
        let d = f.project(&5);
        assert_eq!(c, d);
    }

    #[test]
    fn from_bool() {
        let f = make_f();

        assert_eq!(f.project(&true), f.one());
        assert_eq!(f.project(&false), f.zero());
    }

    #[test]
    fn from_unsigned_and_signed() {
        const LIMBS: usize = U64::LIMBS;
        type F = F64;

        // Using a 64-bit prime
        let f = F::new(&Uint::from(10064419296686275259_u64)).unwrap();
        macro_rules! to_field {
            ($x:expr) => {
                f.project(&$x)
            };
        }
        let zero = f.zero();
        let one = f.one();
        assert_eq!(to_field!(0), zero);
        assert_eq!(to_field!(1), one);
        assert_eq!(f.add(&to_field!(-1), &one), zero);
        assert_eq!(f.add(&to_field!(-5), &f.project(&5)), zero);

        // u64 maximum value (hand-calculated)
        assert_eq!(
            to_field!(u64::MAX),
            to_field!(Uint::<LIMBS>::from_be_hex("7453fff9266d8544"))
        );

        // i64 maximum value (hand-calculated)
        assert_eq!(
            to_field!(i64::MAX),
            to_field!(Uint::<LIMBS>::from_be_hex("7fffffffffffffff"))
        );

        // i64 minimum value (hand-calculated)
        assert_eq!(
            to_field!(i64::MIN),
            to_field!(Uint::<LIMBS>::from_be_hex("0bac0006d9927abb"))
        );

        // Verify property: i64::MIN + |i64::MIN| = 0
        let i64_min_abs = to_field!(i64::MIN.unsigned_abs());
        assert_eq!(f.add(&to_field!(i64::MIN), &i64_min_abs), zero);
    }

    #[test]
    fn from_uint_and_int() {
        const LIMBS: usize = U64::LIMBS;
        type F = F64;

        // Using a 64-bit prime
        let f = F::new(&Uint::from(10064419296686275259_u64)).unwrap();
        macro_rules! to_field {
            ($x:expr) => {
                f.project(&$x)
            };
        }

        let u: Uint<LIMBS> = Uint::from(123_u64);
        assert_eq!(to_field!(u), to_field!(123_u64));

        let i: Int<LIMBS> = Int::from(123_i64);
        assert_eq!(to_field!(i), to_field!(123_u64));

        assert_eq!(to_field!(Uint::<LIMBS>::from(0_u64)), f.zero());

        // Negative Int encodes to the additive inverse
        let i: Int<LIMBS> = Int::from(-123_i64);
        assert_eq!(f.add(&to_field!(i), &to_field!(123_u64)), f.zero());

        // A Uint wider than the modulus is reduced before encoding:
        // 2^64 mod m = (u64::MAX mod m) + 1 (hand-calculated)
        let wide = Uint::<{ 2 * LIMBS }>::from_be_hex("00000000000000010000000000000000");
        assert_eq!(
            to_field!(wide),
            to_field!(Uint::<LIMBS>::from_be_hex("7453fff9266d8545"))
        );
    }

    #[test]
    fn assign_operations() {
        let f = make_f();

        // Addition
        let mut a = f.project(&5);
        f.add_assign(&mut a, &f.project(&6));
        assert_eq!(a, f.project(&11));

        // Subtraction
        let mut a = f.project(&20);
        f.sub_assign(&mut a, &f.project(&7));
        assert_eq!(a, f.project(&13));

        // Multiplication
        let mut a = f.project(&11);
        f.mul_assign(&mut a, &f.project(&3));
        assert_eq!(a, f.project(&33));

        // Division
        let mut a = f.project(&20);
        let b = f.project(&4);
        f.div_assign(&mut a, &b);
        assert_eq!(f.mul(&a, &b), f.project(&20));
    }

    #[test]
    #[should_panic(expected = "Division by zero")]
    fn div_by_zero_panics() {
        let f = make_f();

        let a = f.project(&7);
        let zero = f.zero();
        let _ = f.div(&a, &zero);
    }

    #[test]
    fn pow_operation() {
        let f = make_f();

        // Check both `pow` flavors: a `u32` exponent and the same exponent
        // as a `Self::Integer`.
        macro_rules! assert_pow {
            ($base:expr, $exp:expr, $expected:expr) => {{
                let exp: u32 = $exp;
                assert_eq!(f.pow_u32(&$base, exp), $expected);
                assert_eq!(f.pow(&$base, &Uint::from(exp)), $expected);
            }};
        }

        // Test basic exponentiation
        let base = f.project(&2);

        // 2^0 = 1
        assert_pow!(base, 0, f.one());

        // 2^1 = 2
        assert_pow!(base, 1, base);

        // 2^3 = 8
        assert_pow!(base, 3, f.project(&8));

        // 2^10 = 1024
        assert_pow!(base, 10, f.project(&1024));

        // Test with different base
        let base = f.project(&3);

        // 3^4 = 81
        assert_pow!(base, 4, f.project(&81));

        // Test with base 1
        let base = f.one();
        assert_pow!(base, 1000, f.one());

        // Test with base 0
        let base = f.zero();
        assert_pow!(base, 0, f.one()); // 0^0 = 1 by convention
        assert_pow!(base, 10, f.zero()); // 0^n = 0 for n > 0
    }

    #[test]
    fn inv_operation() {
        let f = make_f();

        let a = f.project(&5);
        let inv_a = f.inv(&a).unwrap();
        assert_eq!(f.mul(&a, &inv_a), f.one());

        // Test that zero has no inverse
        let zero = f.zero();
        assert!(f.inv(&zero).is_none());
    }

    #[test]
    fn checked_div() {
        let f = make_f();

        let a = f.project(&10);
        let b = f.project(&5);
        let zero = f.zero();

        // Normal division
        let c = f.checked_div(&a, &b).unwrap();
        assert_eq!(f.mul(&c, &b), a);

        // Division by zero
        assert!(f.checked_div(&a, &zero).is_none());
    }

    #[test]
    fn aggregate_operations() {
        let f = make_f();

        // Sum
        let values = vec![f.project(&1), f.project(&2), f.project(&3)];
        let sum = f.sum_refs(values.iter());
        assert_eq!(sum, f.project(&6));

        let sum2 = f.sum(values.into_iter());
        assert_eq!(sum2, f.project(&6));

        // Product
        let values = vec![f.project(&2), f.project(&3), f.project(&4)];
        let product = f.product_refs(values.iter());
        assert_eq!(product, f.project(&24));

        let product2 = f.product(values.into_iter());
        assert_eq!(product2, f.project(&24));

        // Empty iterators yield the respective identity elements
        assert_eq!(f.sum(core::iter::empty()), f.zero());
        assert_eq!(f.product(core::iter::empty()), f.one());
    }

    #[test]
    fn conversions() {
        let f = make_f();

        // Test ProjectElement for Uint
        let u = Uint::<LIMBS>::from(123_u64);
        let a = f.project(&u);
        assert_eq!(a, f.project(&123_u64));
    }

    #[test]
    fn from_primitive() {
        let f = make_f();

        // Unsigned types
        assert_eq!(f.project(&42_u8), f.project(&42_u64));
        assert_eq!(f.project(&12345_u16), f.project(&12345_u64));
        assert_eq!(f.project(&1234567890_u32), f.project(&1234567890_u64));
        assert_eq!(
            f.project(&1234567890123456789_u64),
            f.project(&1234567890123456789_u128)
        );

        // Signed types
        assert_eq!(f.project(&-42_i8), f.project(&-42_i64));
        assert_eq!(f.project(&-12345_i16), f.project(&-12345_i64));
        assert_eq!(f.project(&-1234567_i32), f.project(&-1234567_i64));
        assert_eq!(
            f.project(&-1234567890123456789_i64),
            f.project(&-1234567890123456789_i128)
        );
    }

    #[test]
    fn clone_works() {
        let f = make_f();

        let a = f.project(&42);
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn equality_and_ordering() {
        let f = make_f();

        let a = f.project(&10);
        let b = f.project(&10);
        let c = f.project(&20);

        // Test equality
        assert_eq!(a, b);
        assert_ne!(a, c);

        // Test ordering
        // Note: ordering is based on Montgomery representation, not value
        // We just test consistency of ordering
        assert_eq!(a.partial_cmp(&b), Some(Ordering::Equal));
        assert_ne!(a.partial_cmp(&c), Some(Ordering::Equal));
        assert!(a <= b);
        assert!(a >= b);

        // Verify transitivity of ordering
        let d = f.project(&30);
        if a < c && c < d {
            assert!(a < d);
        }
    }

    #[test]
    fn hash_trait() {
        use core::hash::{Hash, Hasher};

        // Simple hasher for testing
        struct TestHasher {
            state: u64,
        }

        impl Hasher for TestHasher {
            fn finish(&self) -> u64 {
                self.state
            }

            fn write(&mut self, bytes: &[u8]) {
                for &byte in bytes {
                    self.state = self.state.wrapping_mul(31).wrapping_add(u64::from(byte));
                }
            }
        }

        let f = make_f();

        let a = f.project(&42);
        let b = f.project(&42);

        let mut hasher_a = TestHasher { state: 0 };
        a.hash(&mut hasher_a);
        let hash_a = hasher_a.finish();

        let mut hasher_b = TestHasher { state: 0 };
        b.hash(&mut hasher_b);
        let hash_b = hasher_b.finish();

        // Equal values should have equal hashes
        assert_eq!(hash_a, hash_b);
    }

    // MontyField-specific tests

    #[test]
    fn pow_matches_crypto_bigint() {
        let f = make_f();

        for (base, exp) in [
            (0_u64, 0_u32),
            (0, 7),
            (1, 1000),
            (2, 10),
            (3, 251),
            (123456789, 1),
            (123456789, u32::MAX),
        ] {
            let x = f.project(&base);
            let expected = FixedMontyForm::from_montgomery(*x.0.inner(), &f.params)
                .pow(&crypto_bigint::Uint::<1>::from(exp));
            assert_eq!(
                f.pow_u32(&x, exp).0.inner(),
                expected.as_montgomery(),
                "{base}^{exp} diverges from FixedMontyForm::pow"
            );
            assert_eq!(
                f.pow(&x, &Uint::from(exp)).0.inner(),
                expected.as_montgomery(),
                "{base}^{exp} (integer exponent) diverges from FixedMontyForm::pow"
            );
        }

        // An exponent wider than u32 (integer-exponent `pow` only)
        let x = f.project(&123456789_u64);
        let exp = u128::MAX;
        let expected = FixedMontyForm::from_montgomery(*x.0.inner(), &f.params)
            .pow(&crypto_bigint::Uint::<LIMBS>::from(exp));
        assert_eq!(
            f.pow(&x, &Uint::from(exp)).0.inner(),
            expected.as_montgomery(),
            "u128::MAX exponent diverges from FixedMontyForm::pow"
        );
    }

    #[test]
    fn prime_field_methods() {
        let f = make_f();

        // Test that we can get modulus
        let modulus = f.modulus();
        assert_eq!(
            modulus,
            Uint::new(crypto_bigint::Uint::<LIMBS>::from_be_hex(
                "fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f"
            ))
        );

        // Test modulus_minus_one_div_two
        let m_minus_1_div_2 = f.modulus_minus_one_div_two();
        assert_eq!(
            m_minus_1_div_2,
            Uint::new(
                (crypto_bigint::Uint::<LIMBS>::from_be_hex(
                    "fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f"
                ) - crypto_bigint::Uint::ONE)
                    / NonZero::new(crypto_bigint::Uint::<LIMBS>::from(2u64)).unwrap()
            )
        );

        // Test zero and one
        let z = f.zero();
        assert!(f.is_zero(&z));
        let o = f.one();
        assert!(!f.is_zero(&o));
    }

    #[test]
    fn new_works() {
        let modulus = Uint::new(crypto_bigint::Uint::<LIMBS>::from_be_hex(
            "fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f",
        ));
        let f = F::new(&modulus).expect("Should create config");

        // Create a field element using this config
        let a = f.project(&123_u64);
        assert_eq!(a, f.project(&123));
    }

    #[test]
    fn new_rejects_even_modulus() {
        let even_modulus = Uint::<LIMBS>::from(42_u64);
        let result = F::new(&even_modulus);
        assert!(result.is_err());
    }
}
