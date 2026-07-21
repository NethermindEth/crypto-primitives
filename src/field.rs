//! This module defines **fields** - sets where addition and
//! multiplication are defined with their respective inverse operations.
//!
//! See the [crate-level documentation](crate) for the overall element/config
//! structure.
//!
//! Currently, this only defines **base** (non-extension) prime fields, whose
//! modulus is an integer, but can be seamlessly extended to support extension
//! fields in the future as well.
//!
//! - [`WithAssociatedInteger`] defines the associated integer type for a field,
//!   which is used for exponents and order, as well as the modulus for base
//!   fields.
//!
//! - [`WithExtensionDegree`] defines the field's degree over its prime subfield
//!   (1 for base fields).
//!
//! - Base field variants of fields are [`BaseField`], [`ConstBaseField`] and
//!   [`BaseFieldConfig`] that define an integer `modulus`. They additionally
//!   allow lifting elements to the associated integer type (and, on the config
//!   side, projecting from it).
//!
//! - [`LiftElementWithConfig`] and [`ProjectElementWithConfig`] define how to
//!   lift a field element to a chosen type and project it back to the field.
//!
//! - Lift/project counterpart for self-sufficient elements is asymmetric -
//!   lifting is done via [`LiftElement`] (by reference, avoiding a copy of the
//!   element) while projection is done with [`From`] (both by reference and by
//!   value), e.g. `F::from(f.lift()) == f`.

#[cfg(feature = "ark_ff")]
pub mod ark_ff_field;
#[cfg(feature = "ark_ff")]
pub mod ark_ff_fp;
#[cfg(feature = "crypto_bigint")]
pub mod crypto_bigint_boxed_monty;
#[cfg(feature = "crypto_bigint")]
pub mod crypto_bigint_const_monty;
#[cfg(feature = "crypto_bigint")]
pub mod crypto_bigint_monty;
pub mod f2;

use crate::{
    ConstRing, FixedConfig, IntSemiring, Ring, RingConfig, SemiringConfig, SetConfig, SetElement,
    helpers::{define_blanket_trait, delegate_to_ref_binary},
};
use core::{
    fmt::Debug,
    ops::{Div, DivAssign, Neg},
};
use num_traits::{Bounded, Inv, Pow, Zero};
use pastey::paste;
use thiserror::Error;

//
// Field (static and const)
//

define_blanket_trait! {
    /// See [module-level documentation](crate::field).
    ///
    /// This is a general trait for all fields, [base](BaseField) and extension.
    pub trait Field:
        SetElement
        + Ring
        + WithAssociatedInteger
        + WithExtensionDegree
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
}

/// The field's degree over its prime subfield; 1 for base fields.
pub trait WithExtensionDegree {
    // Note that extension degree does not require config.
    // However, we still have to manually implement it for BaseFieldConfigs to avoid
    // conflicting blanket implementation with BaseField.

    /// Extension degree of the field.
    fn extension_degree() -> u64;
}

define_blanket_trait! {
    /// [`Field`] with a bunch of values known at compile time.
    pub trait ConstField: Field + ConstRing
}

/// Base (non-extension) prime field with elements being self-sufficient, but
/// whose metadata like modulus is not necessarily known at compile-time.
pub trait BaseField:
    Field + Bounded + LiftElement<<Self as WithAssociatedInteger>::Integer>
{
    fn modulus() -> Self::Integer;

    /// (mod - 1) / 2
    fn modulus_minus_one_div_two() -> Self::Integer;
}

impl<F: BaseField> WithExtensionDegree for F {
    fn extension_degree() -> u64 {
        1
    }
}

/// Base (non-extension) prime field whose modulus and other metadata are
/// constant values known at compile time.
pub trait ConstBaseField:
    ConstField + Bounded + LiftElement<<Self as WithAssociatedInteger>::Integer>
{
    const MODULUS: Self::Integer;

    /// (mod - 1) / 2
    const MODULUS_MINUS_ONE_DIV_TWO: Self::Integer;
}

impl<T: ConstBaseField> BaseField for T {
    fn modulus() -> Self::Integer {
        Self::MODULUS
    }

    /// (mod - 1) / 2
    fn modulus_minus_one_div_two() -> Self::Integer {
        Self::MODULUS_MINUS_ONE_DIV_TWO
    }
}

//
// FieldConfig (both static and dynamic)
//

/// See [module-level documentation](crate::field).
///
/// This is a general trait for all fields, [base](BaseFieldConfig) and
/// extension.
pub trait FieldConfig: RingConfig + WithAssociatedInteger + WithExtensionDegree {
    //
    // Operations on refs
    //

    /// 1/x
    fn inv(&self, x: &Self::Element) -> Option<Self::Element>;

    /// x / y
    fn div(&self, x: &Self::Element, y: &Self::Element) -> Self::Element {
        self.checked_div(x, y).expect("Division by zero")
    }

    /// x ** y
    fn pow(&self, x: &Self::Element, y: &Self::Integer) -> Self::Element;

    //
    // Checked operations on refs
    //

    /// x / y, [`None`] if `y == 0`
    #[inline(always)]
    fn checked_div(&self, x: &Self::Element, y: &Self::Element) -> Option<Self::Element> {
        Some(self.mul(x, &self.inv(y)?))
    }

    // NOTE: `pow` cannot fail

    //
    // Operations on mutable refs
    //

    delegate_to_ref_binary! {
        /// x /= y
        div
    }

    delegate_to_ref_binary! {
        /// x **= y
        pow(&Self::Integer)
    }
}

/// Configuration of an integer field modulo prime number (F_p).
/// Prime modulus might be dynamic and can be determined at runtime.
///
/// When performing arithmetic operations, the modulus of both operands must be
/// the same, otherwise outcome is undefined.
///
/// For base fields (and only for them), [`WithAssociatedInteger::Integer`]
/// additionally serves as the modulus type and the target of `LiftElement*`.
pub trait BaseFieldConfig:
    Sized
    + FieldConfig
    + LiftElementWithConfig<<Self as WithAssociatedInteger>::Integer>
    + ProjectElementWithConfig<<Self as WithAssociatedInteger>::Integer>
{
    fn new(modulus: &Self::Integer) -> Result<Self, FieldError>;

    fn modulus(&self) -> Self::Integer;

    /// (mod - 1) / 2
    fn modulus_minus_one_div_two(&self) -> Self::Integer;
}

impl<F: Field> FieldConfig for FixedConfig<F> {
    #[inline(always)]
    fn inv(&self, x: &Self::Element) -> Option<Self::Element> {
        x.clone().inv()
    }

    #[inline(always)]
    fn pow(&self, x: &Self::Element, y: &F::Integer) -> Self::Element {
        x.clone().pow(y)
    }
}

impl<F: BaseField> BaseFieldConfig for FixedConfig<F> {
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

impl<F> WithAssociatedInteger for FixedConfig<F>
where
    F: SetElement + WithAssociatedInteger,
{
    type Integer = F::Integer;
}

impl<F> LiftElementWithConfig<F::Integer> for FixedConfig<F>
where
    F: Field + WithAssociatedInteger + LiftElement<F::Integer>,
{
    #[inline(always)]
    fn lift(&self, value: &F) -> F::Integer {
        LiftElement::lift(value)
    }
}

impl<F: Field> WithExtensionDegree for FixedConfig<F> {
    #[inline(always)]
    fn extension_degree() -> u64 {
        F::extension_degree()
    }
}

impl<F, T> ProjectElementWithConfig<T> for FixedConfig<F>
where
    F: Field + for<'a> From<&'a T>,
{
    #[inline(always)]
    fn project(&self, value: &T) -> F {
        F::from(value)
    }
}

//
// WithAssociatedInteger
//

pub trait WithAssociatedInteger {
    /// The exponent/order domain of this field: an integer semiring type wide
    /// enough to hold the exponents the field cares about (up to its order).
    type Integer: IntSemiring;
}

//
// LiftElement
//

/// Lifts the field element to a specified type, applicable for self-sufficient
/// fields where this can be done on the element itself.
pub trait LiftElement<T> {
    /// Lift the field element to a specified type using a natural approach.
    ///
    /// Can be projected back to the field using [`From`] to get the same
    /// field element.
    fn lift(&self) -> T;
}

/// Lifts the field element to a specified type, applicable for
/// general/dynamic fields where this requires a [`SetConfig`] to work.
pub trait LiftElementWithConfig<T>: SetConfig {
    /// Lift the field element to a specified type using a natural approach.
    ///
    /// Can be projected back to the field using
    /// [`ProjectElementWithConfig::project`] to get the same field element.
    fn lift(&self, value: &Self::Element) -> T;
}

//
// ProjectElement
//

/// Converts a given value to an element of the current field.
///
/// Static counterpart of this trait is just [`From`].
pub trait ProjectElementWithConfig<T>: SetConfig {
    fn project(&self, value: &T) -> Self::Element;
}

define_blanket_trait! {
    /// The trait combines all `ProjectElementWithConfig<u*>` and
    /// `ProjectElementWithConfig<i*>` into one umbrella trait. Handy when one needs
    /// conversion functions for different primitive int types.
    pub trait ProjectPrimitiveIntegersWithConfig:
        ProjectElementWithConfig<u8>
        + ProjectElementWithConfig<u16>
        + ProjectElementWithConfig<u32>
        + ProjectElementWithConfig<u64>
        + ProjectElementWithConfig<u128>
        + ProjectElementWithConfig<i8>
        + ProjectElementWithConfig<i16>
        + ProjectElementWithConfig<i32>
        + ProjectElementWithConfig<i64>
        + ProjectElementWithConfig<i128>
}

//
// Errors
//

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum FieldError {
    #[error("Invalid field modulus")]
    InvalidModulus,
}
