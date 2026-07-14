//! This module defines **fields** - sets where addition and
//! multiplication are defined with their respective inverse operations.
//! Supports both static and dynamic fields (via general [`FieldConfig`]).
//!
//! Currently, this only defines **base** (non-extension) prime fields, whose
//! modulus is an integer, but can be seamlessly extended to support extension
//! fields in the future as well.
//!
//! - [`FixedField`] and [`ConstField`] are fields where each element is
//!   self-sufficient. Instances carry around all metadata necessary to perform
//!   operations, so they can be used normally as e.g. `a + b`. Every
//!   [`ConstField`] is a [`FixedField`] as well.
//!
//! - [`FieldConfig`] is a field configuration that carries the metadata needed
//!   to perform operations that the elements themselves do not carry, e.g. a
//!   modulus only available at runtime. It's used as `f.add(&a, &b)`. Bridge
//!   between this and [`FixedField`] is [`FixedFieldConfig`].
//!
//! - [`WithAssociatedInteger`] is a trait that defines the associated integer
//!   type for a field, which is used for exponents and order, as well as the
//!   modulus for base fields.
//!
//! - [`LiftElementDynamic`] and [`ProjectElementDynamic`] define how to lift a
//!   field element to a chosen type and project it back, for a general
//!   [`FieldConfig`]; base fields lift to their associated integer type (see
//!   [`BaseFieldConfig`]).
//!
//! - Lift/project counterpart for fixed fields is asymmetric - lifting is done
//!   via [`LiftElementStatic`] (by reference, avoiding a copy of the element)
//!   while projection is done with [`From`] (both by reference and by value),
//!   e.g. `F::from(f.lift()) == f`.
//!
//! - Base field variants of fields are [`FixedBaseField`], [`ConstBaseField`]
//!   and [`BaseFieldConfig`]. They additionally allow lifting elements to the
//!   associated integer type (and, on the config side, projecting from it).
//!   They also define an integer `modulus`.

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

use crate::{ConstRing, ConstSemiring, FixedRing, IntRing, IntSemiring, Semiring, ring::Ring};
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

/// Element of a field (F) - a set where addition and multiplication are
/// defined with their respective inverse operations.
///
/// This is a general trait for all fields, base and extension.
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
}

/// [`FixedField`] with a bunch of values known at compile time.
pub trait ConstField: FixedField + ConstRing {}
impl<F: FixedField + ConstRing> ConstField for F {}

/// Base (non-extension) prime field with elements being self-sufficient, but
/// whose metadata like modulus is not necessarily known at compile-time.
pub trait FixedBaseField: FixedField + LiftElementStatic<Self::Integer> {
    fn modulus() -> Self::Integer;

    /// (mod - 1) / 2
    fn modulus_minus_one_div_two() -> Self::Integer;
}

/// Base (non-extension) prime field whose modulus and other metadata are
/// constant values known at compile time.
pub trait ConstBaseField: ConstField + LiftElementStatic<Self::Integer> {
    const MODULUS: Self::Integer;

    /// (mod - 1) / 2
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

pub trait FieldConfig: WithAssociatedInteger {
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

    /// x / y, [`None`] if `y == 0`
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
/// the same, otherwise outcome is undefined.
///
/// Fixed/constant prime fields are considered a special case of dynamic prime
/// fields, and the bridge struct is [`FixedFieldConfig`].
///
/// For base fields (and only for them), [`WithAssociatedInteger::Integer`]
/// additionally serves as the modulus type and the target of `LiftElement*`.
pub trait BaseFieldConfig:
    Sized
    + FieldConfig
    + LiftElementDynamic<<Self as WithAssociatedInteger>::Integer>
    + ProjectElementDynamic<<Self as WithAssociatedInteger>::Integer>
{
    fn new(modulus: &Self::Integer) -> Result<Self, FieldError>;

    fn modulus(&self) -> Self::Integer;

    /// (mod - 1) / 2
    fn modulus_minus_one_div_two(&self) -> Self::Integer;
}

/// [`BaseFieldConfig`] implementation for fixed fields.
/// It delegates operations to the field, and `new` checks if the modulus
/// matches the static one.
///
/// Use [`Default::default`] to get a usable instance.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct FixedFieldConfig<F: FixedField>(PhantomData<F>);

impl<F: FixedField> FieldConfig for FixedFieldConfig<F> {
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

impl<F: FixedBaseField> BaseFieldConfig for FixedFieldConfig<F> {
    /// Usually it makes more sense to use [`Default::default`] to obtain an
    /// instance instead.
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

impl<F: FixedField> WithAssociatedInteger for FixedFieldConfig<F> {
    type Integer = F::Integer;
}

impl<F: FixedBaseField> LiftElementDynamic<F::Integer> for FixedFieldConfig<F> {
    #[inline(always)]
    fn lift(&self, value: &F) -> F::Integer {
        LiftElementStatic::lift(value)
    }
}

impl<F: FixedBaseField> ProjectElementDynamic<F::Integer> for FixedFieldConfig<F> {
    #[inline(always)]
    fn project(&self, value: &F::Integer) -> F {
        F::from(value)
    }
}

//
// WithAssociatedInteger
//

pub trait WithAssociatedInteger {
    /// The exponent/order domain of this field: an integer semiring type wide
    /// enough to hold the exponents the field cares about (up to its order).
    type Integer: Semiring;
}

//
// LiftElement
//

/// Lifts the field element to a specified type, applicable for fixed fields
/// where this can be done on the element itself.
pub trait LiftElementStatic<T> {
    /// Lift the field element to a specified type using a natural approach.
    ///
    /// Can be projected back to the field using [`From`] to get the same
    /// field element.
    fn lift(&self) -> T;
}

/// Lifts the field element to a specified type, applicable for
/// general/dynamic fields where this requires a [`FieldConfig`] to work.
pub trait LiftElementDynamic<T>: FieldConfig {
    /// Lift the field element to a specified type using a natural approach.
    ///
    /// Can be projected back to the field using
    /// [`ProjectElementDynamic::project`] to get the same field element.
    fn lift(&self, value: &Self::Element) -> T;
}

//
// ProjectElement
//

/// Converts a given value to a field element of a current field.
///
/// Static counterpart of this trait is just [`From`].
pub trait ProjectElementDynamic<T>: FieldConfig {
    fn project(&self, value: &T) -> Self::Element;
}

/// Trivial implementation for fixed fields and types that implement `From<T>`.
impl<F, T> ProjectElementDynamic<T> for F
where
    F: FieldConfig + FixedBaseField,
    F::Element: for<'a> From<&'a T>,
{
    fn project(&self, value: &T) -> F::Element {
        F::Element::from(value)
    }
}

/// The trait combines all `ProjectElement<u*>` and `ProjectElement<i*>` into
/// one umbrella trait. Handy when one needs conversion functions for different
/// primitive int types.
pub trait ProjectPrimitiveIntegersDynamic:
    ProjectElementDynamic<u8>
    + ProjectElementDynamic<u16>
    + ProjectElementDynamic<u32>
    + ProjectElementDynamic<u64>
    + ProjectElementDynamic<u128>
    + ProjectElementDynamic<i8>
    + ProjectElementDynamic<i16>
    + ProjectElementDynamic<i32>
    + ProjectElementDynamic<i64>
    + ProjectElementDynamic<i128>
{
}

/// Blanket implementation.
impl<
    T: ProjectElementDynamic<u8>
        + ProjectElementDynamic<u16>
        + ProjectElementDynamic<u32>
        + ProjectElementDynamic<u64>
        + ProjectElementDynamic<u128>
        + ProjectElementDynamic<i8>
        + ProjectElementDynamic<i16>
        + ProjectElementDynamic<i32>
        + ProjectElementDynamic<i64>
        + ProjectElementDynamic<i128>,
> ProjectPrimitiveIntegersDynamic for T
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
