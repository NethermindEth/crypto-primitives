#[cfg(feature = "ark_ff")]
pub mod ark_ff_field;
#[cfg(feature = "ark_ff")]
pub mod ark_ff_fp;
#[cfg(feature = "crypto_bigint")]
pub mod crypto_bigint_boxed_monty;
#[cfg(feature = "crypto_bigint")]
pub mod crypto_bigint_const_monty;
#[cfg(feature = "crypto_bigint")]
pub(crate) mod crypto_bigint_helpers;
#[cfg(feature = "crypto_bigint")]
pub mod crypto_bigint_monty;
pub mod f2;

use crate::{
    ConstRing, ConstSemiring, FixedRing, FixedSemiring, IntRing, IntSemiring, Semiring, ring::Ring,
};
use core::{
    fmt::Debug,
    marker::PhantomData,
    ops::{Div, DivAssign, Neg},
};
use num_traits::{Inv, Pow, Zero};
use pastey::paste;
use thiserror::Error;

#[cfg(target_pointer_width = "64")]
pub const WORD_FACTOR: usize = 1;
#[cfg(target_pointer_width = "32")]
pub const WORD_FACTOR: usize = 2;

//
// FixedField (static and const)
//

/// Element of a field (F) - a group where addition and multiplication are
/// defined with their respective inverse operations.
pub trait FixedField:
    FixedRing
    + WithAssociatedInteger
    + Pow<u32, Output=Self>
    + Inv<Output = Option<Self>>
    // Arithmetic operations consuming rhs
    + Pow<Self::Integer, Output=Self>
    + Div<Output=Self>
    + DivAssign
    // Arithmetic operations with rhs reference
    + for<'a> Pow<&'a Self::Integer, Output=Self>
    + for<'a> Div<&'a Self, Output=Self>
    + for<'a> DivAssign<&'a Self>
    // Conversion
    + From<u64>
    + From<u128>
    + From<Self::Integer>
    + for<'a> From<&'a Self::Integer>
    + 'static
{
    /// Type of the underlying representation of this structure.
    type Inner: Debug + Eq + Clone + Sync + Send;

    /// Get a reference to the wrapped value.
    fn inner(&self) -> &Self::Inner;

    /// Get a mutable reference to the wrapped value.
    fn inner_mut(&mut self) -> &mut Self::Inner;

    /// Get the wrapped value, consuming self.
    fn into_inner(self) -> Self::Inner;

    /// Creates a new instance of this structure from a representation
    /// known to be valid - should consume exactly the value returned by
    /// `inner()`. Ideally, this should not check the validity of the
    /// element, but it's acceptable to perform a check if it can't be
    /// avoided.
    fn new_unchecked(inner: Self::Inner) -> Self;
}

/// Base (non-extension) prime field with elements being self-sufficient, but
/// whose metadata like modulus is not necessarily known at compile-time.
pub trait FixedBaseField: FixedField + LiftToIntegerStatic {
    fn modulus() -> Self::Integer;

    fn modulus_minus_one_div_two() -> Self::Integer;
}

/// Base (non-extension) prime field whose modulus and other metadata are
/// constant values known at compile time.
pub trait ConstBaseField: FixedField + LiftToIntegerStatic + ConstRing {
    const MODULUS: Self::Integer;
    const MODULUS_MINUS_ONE_DIV_TWO: Self::Integer;
}

impl<T: ConstBaseField> FixedBaseField for T {
    fn modulus() -> Self::Integer {
        Self::MODULUS
    }

    fn modulus_minus_one_div_two() -> Self::Integer {
        Self::MODULUS_MINUS_ONE_DIV_TWO
    }
}

impl<F: FixedField + IntSemiring> IntRing for F {
    #[inline(always)]
    fn checked_abs(&self) -> Option<Self> {
        Some(self.clone())
    }

    #[inline(always)]
    fn is_positive(&self) -> bool {
        !self.is_zero()
    }

    #[inline(always)]
    fn is_negative(&self) -> bool {
        false
    }
}

macro_rules! delegate_to_ref_binary {
    ($(#[$attr:meta])* $op:ident) => {
        delegate_to_ref_binary!($(#[$attr])* $op(&Self::Element));
    };
    ($(#[$attr:meta])* $op:ident($rhs_type:ty)) => {
        paste! {
            $(#[$attr])*
            fn [<$op _assign>](&self, x: &mut Self::Element, y: $rhs_type) {
                *x = self.$op(x, y);
            }
        }
    };
}

//
// FieldConfig (both static and dynamic)
//

pub trait FieldConfigOps: WithAssociatedInteger {
    type Element: Debug + Eq + Clone + Send + Sync + 'static;

    fn is_zero(&self, value: &Self::Element) -> bool;

    fn zero(&self) -> Self::Element;

    fn one(&self) -> Self::Element;

    //
    // Operations on refs
    //

    /// -x
    fn neg(&self, x: &Self::Element) -> Self::Element;

    /// x + y
    fn add(&self, x: &Self::Element, y: &Self::Element) -> Self::Element;

    /// x - y
    fn sub(&self, x: &Self::Element, y: &Self::Element) -> Self::Element;

    /// 1/x
    fn inv(&self, x: &Self::Element) -> Option<Self::Element>;

    /// x * y
    fn mul(&self, x: &Self::Element, y: &Self::Element) -> Self::Element;

    /// x / y
    fn div(&self, x: &Self::Element, y: &Self::Element) -> Self::Element {
        self.checked_div(x, y).expect("Division by zero")
    }

    /// x / y
    ///
    /// (Note: Field operations do not overflow)
    #[inline(always)]
    fn checked_div(&self, x: &Self::Element, y: &Self::Element) -> Option<Self::Element> {
        Some(self.mul(x, &self.inv(y)?))
    }

    /// x ** y
    fn pow(&self, x: &Self::Element, y: &Self::Integer) -> Self::Element;

    /// x ** y
    fn pow_u32(&self, x: &Self::Element, y: u32) -> Self::Element;

    //
    // Operations on mutable refs
    //

    // x = -x;
    fn neg_assign(&self, x: &mut Self::Element) {
        *x = self.neg(x);
    }

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
        /// x /= y
        div
    }

    delegate_to_ref_binary! {
        /// x **= y
        pow(&Self::Integer)
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

/// Element of an integer field modulo prime number (F_p).
/// Prime modulus might be dynamic and can be determined at runtime.
///
/// When performing arithmetic operations, the modulus of both operands must be
/// the same, otherwise operations should panic in debug mode.
///
/// Constant prime fields are considered a special case of dynamic prime fields.
pub trait FieldConfig:
    Sized + WithAssociatedInteger + FieldConfigOps + ProjectElement<<Self as WithAssociatedInteger>::Integer>
{
    fn new(modulus: &Self::Integer) -> Result<Self, FieldError>;

    fn modulus(&self) -> Self::Integer;

    fn modulus_minus_one_div_two(&self) -> Self::Integer;
}

/// [`FieldConfig`] implementation for fixed fields.
/// It delegates operations to the field, and `new` checks if the modulus
/// matches the static one.
#[derive(Clone, Copy, Default, Debug)]
pub struct FixedFieldConfig<F: FixedBaseField>(PhantomData<F>);

impl<F> FieldConfigOps for FixedFieldConfig<F>
where
    F: FixedBaseField + LiftToIntegerStatic,
{
    type Element = F;

    #[inline(always)]
    fn is_zero(&self, value: &Self::Element) -> bool {
        value.is_zero()
    }

    #[inline(always)]
    fn zero(&self) -> Self::Element {
        F::zero()
    }

    #[inline(always)]
    fn one(&self) -> Self::Element {
        F::one()
    }

    #[inline(always)]
    fn neg(&self, x: &Self::Element) -> Self::Element {
        x.clone().neg()
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
    fn inv(&self, x: &Self::Element) -> Option<Self::Element> {
        x.clone().inv()
    }

    #[inline(always)]
    fn mul(&self, x: &Self::Element, y: &Self::Element) -> Self::Element {
        x.clone() * y
    }

    #[inline(always)]
    fn pow(&self, x: &Self::Element, y: &F::Integer) -> Self::Element {
        x.clone().pow(y)
    }

    #[inline(always)]
    fn pow_u32(&self, x: &Self::Element, y: u32) -> Self::Element {
        x.clone().pow(y)
    }
}

impl<F> FieldConfig for FixedFieldConfig<F>
where
    F: FixedBaseField + LiftToIntegerStatic,
{
    fn new(modulus: &Self::Integer) -> Result<Self, FieldError> {
        if *modulus == F::modulus() {
            Ok(Self::default())
        } else {
            Err(FieldError::InvalidModulus)
        }
    }

    #[inline(always)]
    fn modulus(&self) -> Self::Integer {
        F::modulus()
    }

    #[inline(always)]
    fn modulus_minus_one_div_two(&self) -> Self::Integer {
        F::modulus_minus_one_div_two()
    }
}

impl<F: FixedBaseField> WithAssociatedInteger for FixedFieldConfig<F> {
    type Integer = F::Integer;
}

impl<F> LiftToIntegerDynamic for FixedFieldConfig<F>
where
    F: FixedBaseField + LiftToIntegerStatic,
{
    #[inline(always)]
    fn lift_to_integer(&self, value: &Self::Element) -> Self::Integer {
        LiftToIntegerStatic::lift_to_integer(value)
    }
}

impl<F> ProjectElement<F::Integer> for FixedFieldConfig<F>
where
    F: FixedBaseField + LiftToIntegerStatic,
{
    fn project(&self, value: &F::Integer) -> Self::Element {
        F::from(value)
    }
}

//
// LiftToInteger
//

pub trait WithAssociatedInteger {
    /// A semiring integer type that's used to represent the value of this field
    /// when lifted to integers. Also used to represent modulus in prime
    /// fields.
    type Integer: Semiring;
}

pub trait LiftToIntegerStatic: WithAssociatedInteger {
    /// Lift the field element to integer semiring using a natural approach.
    ///
    /// Can be projected back to the field using
    /// [`ProjectElement::encode`] to get the same field element.
    fn lift_to_integer(&self) -> Self::Integer;
}

pub trait LiftToIntegerDynamic: WithAssociatedInteger + FieldConfigOps {
    /// Lift the field element to integer semiring using a natural approach.
    ///
    /// Can be projected back to the field using
    /// [`FromWithConfig::from_with_cfg`] to get the same field element.
    fn lift_to_integer(&self, value: &Self::Element) -> Self::Integer;
}

//
// ProjectElement
//

/// Converts a given value to a field element of a current field.
pub trait ProjectElement<T>: FieldConfigOps {
    fn project(&self, value: &T) -> Self::Element;
}

/// Trivial implementation for fixed fields and types that implement `From<T>`.
impl<F, T> ProjectElement<T> for F
where
    F: FieldConfigOps + FixedBaseField,
    F::Element: for<'a> From<&'a T>,
{
    fn project(&self, value: &T) -> F::Element {
        F::Element::from(value)
    }
}

/// The trait combines all `ProjectElement<u*>` and `ProjectElement<i*>` into
/// one umbrella trait. Handy when one needs conversion functions for different
/// primitive int types.
pub trait ProjectPrimitiveIntegers:
    ProjectElement<u8>
    + ProjectElement<u16>
    + ProjectElement<u32>
    + ProjectElement<u64>
    + ProjectElement<u128>
    + ProjectElement<i8>
    + ProjectElement<i16>
    + ProjectElement<i32>
    + ProjectElement<i64>
    + ProjectElement<i128>
{
}

/// Blanket implementation.
impl<
    T: ProjectElement<u8>
        + ProjectElement<u16>
        + ProjectElement<u32>
        + ProjectElement<u64>
        + ProjectElement<u128>
        + ProjectElement<i8>
        + ProjectElement<i16>
        + ProjectElement<i32>
        + ProjectElement<i64>
        + ProjectElement<i128>,
> ProjectPrimitiveIntegers for T
{
}

//
// Errors
//

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum FieldError {
    #[error("Invalid field modulus")]
    InvalidModulus,
}
