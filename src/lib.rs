//! This crate defines an algebraic hierarchy of **sets**, **semirings**,
//! **rings** and **fields**. Supports both static and dynamic structures (via
//! the general `*Config` traits: [`SetConfig`], [`SemiringConfig`],
//! [`RingConfig`] and [`FieldConfig`]).
//!
//! - [`Semiring`], [`Ring`] and [`Field`] (along with their `Const*` variants)
//!   are structures where each element is self-sufficient. Instances carry
//!   around all metadata necessary to perform operations, so they can be used
//!   normally as e.g. `a + b`. Every `Const*` structure implements its
//!   non-const counterpart as well. These traits have blanket implementations.
//!
//! - [`SemiringConfig`], [`RingConfig`] and [`FieldConfig`] are configurations
//!   that carry the metadata needed to perform operations that the elements
//!   themselves do not carry, e.g. a modulus only available at runtime. They're
//!   used as `f.add(&a, &b)`.
//!
//! - Bridge between these and structures with self-sufficient elements is
//!   [`FixedConfig`].

#![no_std]
extern crate alloc;

pub mod field;
pub(crate) mod helpers;
pub mod matrix;
pub mod ring;
pub mod semiring;

pub use field::*;
pub use matrix::*;
pub use ring::*;
pub use semiring::*;

use crate::helpers::define_blanket_trait;
use core::{fmt::Debug, marker::PhantomData};

/// A thin wrapper around some underlying representation.
pub trait Wrapper {
    /// Type of the underlying representation of this structure.
    type Inner: Debug + Clone + Eq + Send + Sync;

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

define_blanket_trait! {
    /// Base type we demand of all of our set elements - values that would be passed
    /// around.
    pub trait SetElement: Debug + Clone + Eq + Send + Sync + 'static
}

pub trait SetConfig: Debug + Clone + Eq + Send + Sync {
    type Element: SetElement;
}

pub trait ParseStrConfig: SetConfig {
    fn parse_str(&self, str: &str) -> Option<Self::Element>;
}

/// Bridge between set configs and fixed sets with self-sufficient elements that
/// delegates config operations to the set itself.
///
/// Use [`Default::default`] to get a usable instance.
/// For fields, `new` checks if the modulus matches the static one.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FixedConfig<T: SetElement>(PhantomData<T>);

impl<T: SetElement> FixedConfig<T> {
    pub const fn const_default() -> Self {
        FixedConfig(PhantomData)
    }
}

impl<T: SetElement> SetConfig for FixedConfig<T> {
    type Element = T;
}
