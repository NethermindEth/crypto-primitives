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

use core::fmt::Debug;

/// A thin wrapper around some underlying representation.
pub trait Wrapper {
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
