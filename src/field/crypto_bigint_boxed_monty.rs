use super::*;
use crate::{
    IntSemiring, IntSemiringConfig, LiftElementWithConfig, Wrapper, boolean::Boolean,
    crypto_bigint_boxed_uint::BoxedUint, crypto_bigint_int::Int, crypto_bigint_uint::Uint,
    helpers::crypto_bigint as helpers,
};
use alloc::borrow::Cow;
use core::fmt::{Debug, Display, Formatter, Result as FmtResult};
use crypto_bigint::{
    MontyForm, Odd,
    modular::{BoxedMontyForm, BoxedMontyParams},
};
use num_traits::{One, Signed};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BoxedMontyField {
    pub params: BoxedMontyParams,
    pub modulus_minus_one_div_two: BoxedUint,
}

impl BoxedMontyField {
    /// Creates a new [`BoxedMontyField`] from [`BoxedMontyParams`].
    #[inline(always)]
    #[allow(clippy::arithmetic_side_effects)] // False alert
    pub fn wrap(params: BoxedMontyParams) -> Self {
        let two = BoxedUint::from(2_u8).into_inner();
        let two = two.as_nz_vartime().expect("Can't fail");
        let modulus_minus_one_div_two =
            (params.modulus().as_ref() - BoxedUint::one().into_inner()).div(two);
        let modulus_minus_one_div_two = BoxedUint::new(modulus_minus_one_div_two);
        Self {
            params,
            modulus_minus_one_div_two,
        }
    }

    /// Unwrap the [`BoxedMontyForm`] into a field config and a field element.
    #[inline(always)]
    pub fn unwrap_monty(el: BoxedMontyForm) -> (Self, BoxedUint) {
        let params = el.params().clone();
        (Self::wrap(params), BoxedUint::new(el.into_montgomery()))
    }

    /// Reassemble the `crypto-bigint`'s [`BoxedMontyForm`] from a raw
    /// Montgomery value.
    #[inline(always)]
    pub fn form(&self, el: &BoxedMontyFieldElement) -> BoxedMontyForm {
        BoxedMontyForm::from_montgomery(el.0.inner().clone(), &self.params)
    }

    #[inline(always)]
    pub fn bits_precision(&self) -> u32 {
        self.params.bits_precision()
    }
}

/// A wrapper around [`BoxedUint`] to prevent accidentally calling math
/// operations on it.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct BoxedMontyFieldElement(pub BoxedUint);

impl BoxedMontyFieldElement {
    #[inline(always)]
    pub fn bits_precision(&self) -> u32 {
        self.0.bits_precision()
    }
}

impl From<BoxedUint> for BoxedMontyFieldElement {
    #[inline(always)]
    fn from(value: BoxedUint) -> Self {
        Self(value)
    }
}

impl From<&BoxedUint> for &BoxedMontyFieldElement {
    #[inline(always)]
    fn from(value: &BoxedUint) -> Self {
        // Safety: BoxedMontyFieldElement is #[repr(transparent)] and is guaranteed to
        // have the same memory layout.
        unsafe { &*(value as *const BoxedUint as *const BoxedMontyFieldElement) }
    }
}

impl From<crypto_bigint::BoxedUint> for BoxedMontyFieldElement {
    #[inline(always)]
    fn from(value: crypto_bigint::BoxedUint) -> Self {
        Self(BoxedUint::new(value))
    }
}

//
// Wrapper
//

impl Wrapper for BoxedMontyFieldElement {
    type Inner = BoxedUint;

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

impl Display for BoxedMontyField {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "F_{}", self.params.modulus())
    }
}

//
// Field operations
//

impl SetConfig for BoxedMontyField {
    type Element = BoxedMontyFieldElement;
}

impl SemiringConfig for BoxedMontyField {
    fn is_zero(&self, value: &Self::Element) -> bool {
        value.0.is_zero()
    }

    fn zero(&self) -> Self::Element {
        BoxedUint::zero_with_precision(self.params.bits_precision()).into()
    }

    fn one(&self) -> Self::Element {
        self.params.as_ref().one().clone().into()
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

    /// Montgomery multiplication `x*y/R mod m`, fully reduced.
    fn mul(&self, x: &Self::Element, y: &Self::Element) -> Self::Element {
        // Note: CIOS multiplication is worse for this
        let modulus = BoxedUint::new_ref(self.params.modulus().as_ref());
        let mut out = BoxedUint::zero_with_precision(self.bits_precision());
        let carry = helpers::mul::monty_mul_limbs(
            x.0.as_limbs(),
            y.0.as_limbs(),
            out.as_mut_limbs(),
            modulus.as_limbs(),
            self.params.as_ref().mod_neg_inv(),
        );

        out.sub_assign_mod_with_carry(carry, modulus, modulus);
        out.into()
    }

    fn pow_u32(&self, x: &Self::Element, y: u32) -> Self::Element {
        helpers::pow::pow_u32(
            &x.0,
            y,
            BoxedUint::new(self.params.as_ref().one().clone()),
            BoxedUint::new_ref(self.params.modulus().as_ref()),
            self.params.as_ref().mod_neg_inv(),
        )
        .into()
    }

    // Modular arithmetic cannot overflow, so the checked variants always
    // succeed.

    fn checked_add(&self, x: &Self::Element, y: &Self::Element) -> Option<Self::Element> {
        Some(self.add(x, y))
    }

    fn checked_sub(&self, x: &Self::Element, y: &Self::Element) -> Option<Self::Element> {
        Some(self.sub(x, y))
    }

    fn checked_mul(&self, x: &Self::Element, y: &Self::Element) -> Option<Self::Element> {
        Some(self.mul(x, y))
    }

    fn checked_pow_u32(&self, x: &Self::Element, y: u32) -> Option<Self::Element> {
        Some(self.pow_u32(x, y))
    }

    fn add_assign(&self, x: &mut Self::Element, y: &Self::Element) {
        x.0.inner_mut()
            .add_mod_assign(y.0.inner(), self.params.modulus().as_nz_ref());
    }

    fn sub_assign(&self, x: &mut Self::Element, y: &Self::Element) {
        let modulus_ref = BoxedUint::new_ref(self.params.modulus().as_ref());
        x.0.sub_assign_mod_with_carry(crypto_bigint::Limb::ZERO, &y.0, modulus_ref);
    }

    // Original mul_assign's implementation delegates to mul, so we stick to that
}

impl RingConfig for BoxedMontyField {
    fn neg(&self, x: &Self::Element) -> Self::Element {
        x.0.inner()
            .neg_mod(self.params.modulus().as_nz_ref())
            .into()
    }

    fn checked_neg(&self, x: &Self::Element) -> Option<Self::Element> {
        Some(self.neg(x))
    }
}

impl IntSemiringConfig for BoxedMontyField {
    // Parity is a property of the represented residue, so it requires leaving
    // the Montgomery domain.

    fn is_odd(&self, x: &Self::Element) -> bool {
        IntSemiring::is_odd(&self.lift(x))
    }

    fn is_even(&self, x: &Self::Element) -> bool {
        IntSemiring::is_even(&self.lift(x))
    }
}

impl WithExtensionDegree for BoxedMontyField {
    #[inline(always)]
    fn extension_degree() -> u64 {
        1
    }
}

impl FieldConfig for BoxedMontyField {
    fn inv(&self, x: &Self::Element) -> Option<Self::Element> {
        // Follows BoxedMontyForm::invert_vartime
        let inverted: Option<crypto_bigint::BoxedUint> =
            x.0.inner()
                .invert_odd_mod_vartime(self.params.modulus())
                .into();
        let inverted = BoxedUint::new(inverted?).into();

        let r2 = BoxedUint::new_ref(self.params.as_ref().r2()).into();
        let x_inv = self.mul(&inverted, r2);
        Some(self.mul(&x_inv, r2))
    }

    fn pow(&self, x: &Self::Element, y: &Self::Integer) -> Self::Element {
        helpers::pow::pow_bounded_exp(
            &x.0,
            y.as_limbs(),
            y.bits_precision(),
            BoxedUint::new(self.params.as_ref().one().clone()),
            BoxedUint::new_ref(self.params.modulus().as_ref()),
            self.params.as_ref().mod_neg_inv(),
        )
        .into()
    }
}

//
// Conversions
//

macro_rules! impl_from_unsigned {
    ($($t:ty),* $(,)?) => {
        $(
            impl ProjectElementWithConfig<$t> for BoxedMontyField {
                fn project(&self, value: &$t) -> Self::Element {
                    let abs: BoxedUint = value.into();
                    self.project(&abs.resize(self.modulus().bits_precision()))
                }
            }
        )*
    };
}

macro_rules! impl_from_signed {
    ($($t:ty),* $(,)?) => {
        $(
            impl ProjectElementWithConfig<$t> for BoxedMontyField {
                fn project(&self, value: &$t) -> Self::Element {
                    let magnitude = BoxedUint::from(value.abs_diff(0)).resize(self.modulus().bits_precision());
                    let magnitude = self.project(&magnitude);
                    if value.is_negative() { self.neg(&magnitude) } else { magnitude }
                }
            }
        )*
    };
}

impl_from_unsigned!(u8, u16, u32, u64, u128);
impl_from_signed!(i8, i16, i32, i64, i128);

impl ProjectElementWithConfig<bool> for BoxedMontyField {
    fn project(&self, value: &bool) -> Self::Element {
        if *value { self.one() } else { self.zero() }
    }
}

impl ProjectElementWithConfig<Boolean> for BoxedMontyField {
    #[inline(always)]
    fn project(&self, value: &Boolean) -> Self::Element {
        self.project(value.inner())
    }
}

impl<const LIMBS: usize> ProjectElementWithConfig<Int<LIMBS>> for BoxedMontyField {
    #[allow(clippy::arithmetic_side_effects)] // False alert
    fn project(&self, value: &Int<LIMBS>) -> Self::Element {
        let abs: BoxedUint = value.inner().abs().into();
        let abs = self.project(&abs.resize(self.modulus().bits_precision()));
        if value.is_negative() {
            self.neg(&abs)
        } else {
            abs
        }
    }
}

impl ProjectElementWithConfig<BoxedUint> for BoxedMontyField {
    /// Convert the given integer into the Montgomery domain.
    fn project(&self, value: &BoxedUint) -> Self::Element {
        use crypto_bigint::Resize;

        let value = if value.bits_precision() < self.bits_precision() {
            Cow::Owned(value.resize(self.bits_precision()))
        } else if value.bits_precision() > self.bits_precision() {
            // The value is wider than the modulus: reduce it first, so that
            // the resize below cannot truncate.
            let modulus = self
                .params
                .modulus()
                .as_ref()
                .resize(value.bits_precision());
            let modulus = modulus.into_nz().expect("modulus should be non-zero");
            let rem = value.0.rem_vartime(&modulus);
            Cow::Owned(BoxedUint::new(rem).resize(self.bits_precision()))
        } else {
            Cow::Borrowed(value)
        };

        self.mul(
            value.as_ref().into(),
            BoxedUint::new_ref(self.params.as_ref().r2()).into(),
        )
    }
}

impl<const LIMBS: usize> ProjectElementWithConfig<Uint<LIMBS>> for BoxedMontyField {
    #[inline(always)]
    fn project(&self, value: &Uint<LIMBS>) -> Self::Element {
        self.project(&BoxedUint::from(&value.0))
    }
}

impl<const LIMBS: usize> ProjectElementWithConfig<crypto_bigint::Uint<LIMBS>> for BoxedMontyField {
    #[inline(always)]
    fn project(&self, value: &crypto_bigint::Uint<LIMBS>) -> Self::Element {
        self.project(Uint::new_ref(value))
    }
}

//
// Field stuff
//

impl BaseFieldConfig for BoxedMontyField {
    fn new(modulus: &Self::Integer) -> Result<Self, FieldError> {
        let Some(modulus) = Odd::new(modulus.clone().into_inner()).into_option() else {
            return Err(FieldError::InvalidModulus);
        };
        Ok(Self::wrap(BoxedMontyParams::new(modulus)))
    }

    fn modulus(&self) -> Self::Integer {
        BoxedUint::new(self.params.modulus().clone().get())
    }

    #[allow(clippy::arithmetic_side_effects)] // False alert
    fn modulus_minus_one_div_two(&self) -> Self::Integer {
        self.modulus_minus_one_div_two.clone()
    }
}

impl WithAssociatedInteger for BoxedMontyField {
    type Integer = BoxedUint;
}

impl LiftElementWithConfig<<Self as WithAssociatedInteger>::Integer> for BoxedMontyField {
    /// Retrieves the integer currently encoded in this [`BoxedMontyField`],
    /// guaranteed to be reduced.
    #[inline(always)]
    fn lift(&self, value: &Self::Element) -> <Self as WithAssociatedInteger>::Integer {
        let mut out = BoxedUint::zero_with_precision(value.bits_precision());
        helpers::monty_retrieve_inner(
            value.0.as_limbs(),
            out.as_mut_limbs(),
            self.modulus().as_limbs(),
            self.params.as_ref().mod_neg_inv(),
        );
        out
    }
}

// TODO: Do we want to zeroize the modulus?

#[allow(
    clippy::arithmetic_side_effects,
    clippy::cast_lossless,
    clippy::redundant_clone
)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::ensure_type_implements_trait;
    use alloc::vec;
    use core::cmp::Ordering;

    type F = BoxedMontyField;

    #[test]
    fn ensure_traits() {
        ensure_type_implements_trait!(F, BaseFieldConfig);
    }

    //
    // Test helpers
    //

    fn make_f() -> F {
        // Using a 256-bit prime 2^256 - 2^32 - 977 (secp256k1 field prime)
        let modulus = BoxedUint::from_be_hex(
            "fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f",
            256,
        )
        .unwrap();
        let modulus = Odd::new(modulus.into_inner()).expect("modulus should be odd");
        F::wrap(BoxedMontyParams::new(modulus))
    }

    #[test]
    fn encoding() {
        let f = make_f();

        // `modulus + 1` should reduce to one.
        let x = BoxedUint::from_be_hex(
            "fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc30",
            256,
        )
        .unwrap();
        assert_eq!(f.project(&x), f.one());

        // `modulus` itself should reduce to zero.
        let modulus = F::modulus(&f);
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

        let mod_minus_one = f.modulus() - BoxedUint::one();
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
        let f = {
            // Using a 64-bit prime 10064419296686275259
            let modulus = BoxedUint::from_be_hex("8bac0006d9927abb", 64).unwrap();
            let modulus = Odd::new(modulus.into_inner()).expect("modulus should be odd");
            F::wrap(BoxedMontyParams::new(modulus))
        };
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
            to_field!(BoxedUint::from_be_hex("7453fff9266d8544", 64).unwrap())
        );

        // i64 maximum value (hand-calculated)
        assert_eq!(
            to_field!(i64::MAX),
            to_field!(BoxedUint::from_be_hex("7fffffffffffffff", 64).unwrap())
        );

        // i64 minimum value (hand-calculated)
        assert_eq!(
            to_field!(i64::MIN),
            to_field!(BoxedUint::from_be_hex("0bac0006d9927abb", 64).unwrap())
        );

        // Verify property: i64::MIN + |i64::MIN| = 0
        let i64_min_abs = to_field!(i64::MIN.unsigned_abs());
        assert_eq!(f.add(&to_field!(i64::MIN), &i64_min_abs), zero);
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
                assert_eq!(f.pow(&$base, &BoxedUint::from(exp)), $expected);
            }};
        }

        // Test basic exponentiation
        let base = f.project(&2);

        // 2^0 = 1
        assert_pow!(base, 0, f.one());

        // 2^1 = 2
        assert_pow!(base, 1, base.clone());

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
            let expected = BoxedMontyForm::from_montgomery(x.0.inner().clone(), &f.params)
                .pow(&crypto_bigint::BoxedUint::from(u64::from(exp)));
            assert_eq!(
                f.pow_u32(&x, exp).0.inner(),
                expected.as_montgomery(),
                "{base}^{exp} diverges from BoxedMontyForm::pow"
            );
            assert_eq!(
                f.pow(&x, &BoxedUint::from(exp)).0.inner(),
                expected.as_montgomery(),
                "{base}^{exp} (integer exponent) diverges from BoxedMontyForm::pow"
            );
        }

        // An exponent wider than u32 (integer-exponent `pow` only)
        let x = f.project(&123456789_u64);
        let exp = u128::MAX;
        let expected = BoxedMontyForm::from_montgomery(x.0.inner().clone(), &f.params)
            .pow(&crypto_bigint::BoxedUint::from(exp));
        assert_eq!(
            f.pow(&x, &BoxedUint::from(exp)).0.inner(),
            expected.as_montgomery(),
            "u128::MAX exponent diverges from BoxedMontyForm::pow"
        );
    }

    /// Regression test: the final AMM value of the exponentiation can land in
    /// `[modulus, 2*modulus)`, where exactly one final conditional subtraction
    /// is needed. `almost_montgomery_reduce` used to apply a stale comparison
    /// to both subtractions, subtracting the modulus twice and wrapping.
    #[test]
    fn pow_amm_final_reduction() {
        let f = {
            // Using a 64-bit prime 10064419296686275259
            let modulus = BoxedUint::from_be_hex("8bac0006d9927abb", 64).unwrap();
            let modulus = Odd::new(modulus.into_inner()).expect("modulus should be odd");
            F::wrap(BoxedMontyParams::new(modulus))
        };

        // Small cases with hand-checkable results
        for (base, exp, result) in [(2_u64, 8_u32, 256_u64), (2, 9, 512), (3, 5, 243)] {
            assert_eq!(
                f.pow_u32(&f.project(&base), exp),
                f.project(&result),
                "{base}^{exp} != {result}"
            );
        }

        // Larger cases, verified against the library
        for (base, exp) in [(12345_u64, 67_u32), (2, 64), (123456789, u32::MAX)] {
            let x = f.project(&base);
            let expected = BoxedMontyForm::from_montgomery(x.0.inner().clone(), &f.params)
                .pow(&crypto_bigint::BoxedUint::from(u64::from(exp)));
            assert_eq!(
                f.pow_u32(&x, exp).0.inner(),
                expected.as_montgomery(),
                "{base}^{exp} diverges from BoxedMontyForm::pow"
            );
        }
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

        // Test ProjectElement for BoxedUint
        let u = BoxedUint::from(123_u64).resize(f.modulus().bits_precision());
        let a = f.project(&u);
        assert_eq!(a, f.project(&123_u64));

        // A BoxedUint narrower than the field is zero-extended...
        let narrow = BoxedUint::from(123_u64); // 64-bit precision
        assert_eq!(f.project(&narrow), f.project(&123_u64));

        // ...including when it arrives via the fixed-width Uint projections.
        assert_eq!(f.project(&Uint::<1>::from(123_u64)), f.project(&123_u64));

        // A BoxedUint wider than the field is reduced, not truncated:
        // 2^256 mod (2^256 - 2^32 - 977) = 2^32 + 977 (truncation would give 0)
        let wide = BoxedUint::from_be_hex(
            "0000000000000000000000000000000000000000000000000000000000000001\
             0000000000000000000000000000000000000000000000000000000000000000",
            512,
        )
        .unwrap();
        assert_eq!(f.project(&wide), f.project(&0x1_000003D1_u64));
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

    // BoxedMontyField-specific tests

    #[test]
    fn prime_field_methods() {
        let f = make_f();

        // Test that we can get modulus
        let modulus = f.modulus();
        assert!(modulus.bits_precision() > 0);

        // Test modulus_minus_one_div_two
        let m_minus_1_div_2 = f.modulus_minus_one_div_two();
        assert!(m_minus_1_div_2.bits_precision() > 0);

        // Test zero and one
        let z = f.zero();
        assert!(f.is_zero(&z));
        let o = f.one();
        assert!(!f.is_zero(&o));
    }

    #[test]
    fn new_works() {
        let modulus = BoxedUint::from_be_hex(
            "fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f",
            256,
        )
        .unwrap();
        let f = F::new(&modulus).expect("Should create config");

        // Create a field element using this config
        let a = f.project(&123_u64);
        assert_eq!(a, f.project(&123));
    }

    #[test]
    fn new_rejects_even_modulus() {
        let even_modulus = BoxedUint::from(42_u64);
        let result = F::new(&even_modulus);
        assert!(result.is_err());
    }
}
