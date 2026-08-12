//! Canonical byte serialization, intended for Fiat-Shamir transcripts.
//!
//! The encoding is a function of the mathematical value alone. It does not
//! depend on the backend that holds the value, on the target endianness, or on
//! the storage precision. Byte order is little-endian throughout.
//!
//! Do not use `serde` or `zerocopy` to feed a transcript. `zerocopy` exposes
//! the memory layout, and `serde` is a storage format that is only guaranteed
//! to round-trip through the same type.
//!
//! - [`CanonicalBytes`] serializes self-sufficient elements.
//! - [`CanonicalBytesWithConfig`] serializes elements that need a config, such
//!   as a field with a runtime modulus.
//! - [`FromUniformBytes`] and [`FromUniformBytesWithConfig`] map squeezed
//!   transcript output to an element. They are a different map, and they are
//!   not injective. Never use them to read a prover message.
//!
//! # Framing
//!
//! The transcript layer owns message framing and domain labels. This module
//! only guarantees that an encoding is unambiguous: either the width is
//! constant for the type (see [`FixedCanonicalBytes`]) or for the config, or
//! the encoding carries its own length.

use crate::{BaseField, BaseFieldConfig, ProjectElementWithConfig, SetConfig};
use alloc::vec::Vec;
use thiserror::Error;

/// Bytes drawn beyond the modulus width when sampling an element from
/// transcript output. Bounds the sampling bias by `2^-128`.
pub const UNIFORM_BYTES_MARGIN: usize = 16;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CanonicalBytesError {
    #[error("expected {expected} bytes, got {actual}")]
    InvalidLength { expected: usize, actual: usize },
    #[error("value is not reduced modulo the modulus")]
    NonCanonical,
    #[error("value does not fit into {width} bytes")]
    Overflow { width: usize },
}

/// Converts a bit count to the number of bytes that holds it.
#[inline]
fn bits_to_bytes(bits: u32) -> usize {
    usize::try_from(bits).unwrap_or(usize::MAX).div_ceil(8)
}

//
// Element-side traits
//

/// Canonical byte encoding for elements that carry everything they need.
///
/// Two values are equal if and only if their encodings are equal.
pub trait CanonicalBytes: Sized {
    /// Number of bytes [`Self::write_canonical`] appends for `self`.
    fn canonical_byte_len(&self) -> usize;

    /// Append the canonical encoding of `self` to `out`.
    fn write_canonical(&self, out: &mut Vec<u8>);

    /// Read one value from its complete canonical encoding.
    ///
    /// Rejects a wrong length and an encoding that is not canonical.
    fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, CanonicalBytesError>;

    fn to_canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.canonical_byte_len());
        self.write_canonical(&mut out);
        out
    }
}

/// [`CanonicalBytes`] with a width that is the same for every value.
///
/// A constant width makes the encoding prefix-free, so a transcript can absorb
/// such values back to back without a length prefix.
pub trait FixedCanonicalBytes: CanonicalBytes {
    /// Width shared by every value of this type.
    fn fixed_canonical_byte_len() -> usize;
}

/// Maps uniform transcript output to an element.
pub trait FromUniformBytes: Sized {
    /// Bytes that [`Self::from_uniform_bytes`] needs.
    fn uniform_byte_len() -> usize;

    /// Maps uniform bytes to a near-uniform element. Not injective.
    ///
    /// Shorter input is accepted but increases the bias.
    fn from_uniform_bytes(bytes: &[u8]) -> Self;
}

//
// Config-side traits
//

/// Canonical byte encoding for elements that need a config.
///
/// The config pins the modulus, so the width is constant for one config
/// instance, and the encoding is prefix-free for that instance.
#[allow(
    clippy::wrong_self_convention,
    reason = "kept symmetric with the element-side trait, as `lift` already is"
)]
pub trait CanonicalBytesWithConfig: SetConfig {
    /// Number of bytes [`Self::write_canonical`] appends for any element.
    fn canonical_byte_len(&self) -> usize;

    fn write_canonical(&self, value: &Self::Element, out: &mut Vec<u8>);

    fn from_canonical_bytes(&self, bytes: &[u8]) -> Result<Self::Element, CanonicalBytesError>;

    fn to_canonical_bytes(&self, value: &Self::Element) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.canonical_byte_len());
        self.write_canonical(value, &mut out);
        out
    }
}

/// Config-side counterpart of [`FromUniformBytes`].
#[allow(
    clippy::wrong_self_convention,
    reason = "kept symmetric with the element-side trait, as `lift` already is"
)]
pub trait FromUniformBytesWithConfig: SetConfig {
    fn uniform_byte_len(&self) -> usize;

    fn from_uniform_bytes(&self, bytes: &[u8]) -> Self::Element;
}

//
// Integer layer
//

/// Little-endian byte access for the integer types that back our sets.
///
/// `bit_len` lives here rather than on [`IntSemiring`](crate::IntSemiring)
/// because a bit length is ambiguous for a signed type, and only a modulus
/// ever needs one.
pub trait CanonicalIntBytes: Sized {
    /// Number of bits needed to hold this value. Zero for zero.
    fn bit_len(&self) -> u32;

    /// Append exactly `width` little-endian bytes.
    ///
    /// Fails if the value needs more than `width` bytes.
    fn write_le(&self, width: usize, out: &mut Vec<u8>) -> Result<(), CanonicalBytesError>;

    /// Read a little-endian value from `bytes`.
    ///
    /// `like` supplies the storage shape for heap-allocated integers. Fixed
    /// width integers ignore it.
    fn read_le_like(bytes: &[u8], like: &Self) -> Result<Self, CanonicalBytesError>;
}

/// Bytes a base field element occupies: `ceil(modulus_bits / 8)`.
///
/// The width follows the modulus, never the limb count or a runtime precision.
#[inline]
pub fn canonical_width<I: CanonicalIntBytes>(modulus: &I) -> usize {
    bits_to_bytes(modulus.bit_len())
}

//
// Shared base field logic
//

pub fn write_base_field<F>(value: &F, out: &mut Vec<u8>)
where
    F: BaseField,
    F::Integer: CanonicalIntBytes,
{
    let width = canonical_width(&F::modulus());
    // A lifted element is reduced, so it always fits.
    let _ = value.lift().write_le(width, out);
}

pub fn read_base_field<F>(bytes: &[u8]) -> Result<F, CanonicalBytesError>
where
    F: BaseField,
    F::Integer: CanonicalIntBytes,
{
    let modulus = F::modulus();
    let width = canonical_width(&modulus);
    if bytes.len() != width {
        return Err(CanonicalBytesError::InvalidLength {
            expected: width,
            actual: bytes.len(),
        });
    }
    let value = F::Integer::read_le_like(bytes, &modulus)?;
    if value >= modulus {
        return Err(CanonicalBytesError::NonCanonical);
    }
    Ok(F::from(&value))
}

pub fn write_base_field_with_config<C>(cfg: &C, value: &C::Element, out: &mut Vec<u8>)
where
    C: BaseFieldConfig,
    C::Integer: CanonicalIntBytes,
{
    let width = canonical_width(&cfg.modulus());
    let _ = cfg.lift(value).write_le(width, out);
}

pub fn read_base_field_with_config<C>(
    cfg: &C,
    bytes: &[u8],
) -> Result<C::Element, CanonicalBytesError>
where
    C: BaseFieldConfig,
    C::Integer: CanonicalIntBytes,
{
    let modulus = cfg.modulus();
    let width = canonical_width(&modulus);
    if bytes.len() != width {
        return Err(CanonicalBytesError::InvalidLength {
            expected: width,
            actual: bytes.len(),
        });
    }
    let value = C::Integer::read_le_like(bytes, &modulus)?;
    if value >= modulus {
        return Err(CanonicalBytesError::NonCanonical);
    }
    Ok(cfg.project(&value))
}

//
// Shared uniform sampling logic
//

/// Reduces `bytes` into the field with Horner's rule over 64-bit chunks.
///
/// Working inside the field avoids wide integer arithmetic, so this needs
/// nothing beyond the field operations themselves.
pub fn base_field_from_uniform_bytes<F>(bytes: &[u8]) -> F
where
    F: BaseField,
{
    let shift = F::from(1_u128 << 64);
    let mut acc = F::from(0_u64);
    // Only the most significant chunk can be short, and it is consumed first.
    for chunk in bytes.chunks(8).rev() {
        let mut buf = [0_u8; 8];
        buf[..chunk.len()].copy_from_slice(chunk);
        acc = acc * &shift + F::from(u64::from_le_bytes(buf));
    }
    acc
}

/// Config-side counterpart of [`base_field_from_uniform_bytes`].
pub fn base_field_from_uniform_bytes_with_config<C>(cfg: &C, bytes: &[u8]) -> C::Element
where
    C: BaseFieldConfig + ProjectElementWithConfig<u64> + ProjectElementWithConfig<u128>,
{
    let shift = cfg.project(&(1_u128 << 64));
    let mut acc = cfg.project(&0_u64);
    for chunk in bytes.chunks(8).rev() {
        let mut buf = [0_u8; 8];
        buf[..chunk.len()].copy_from_slice(chunk);
        let limb = cfg.project(&u64::from_le_bytes(buf));
        acc = cfg.add(&cfg.mul(&acc, &shift), &limb);
    }
    acc
}

/// Bytes needed to sample a base field element with bias at most `2^-128`.
#[inline]
pub fn uniform_width<I: CanonicalIntBytes>(modulus: &I) -> usize {
    canonical_width(modulus).saturating_add(UNIFORM_BYTES_MARGIN)
}

//
// Bridge for fixed configs
//

impl<F> CanonicalBytesWithConfig for crate::FixedConfig<F>
where
    F: crate::SetElement + FixedCanonicalBytes,
{
    fn canonical_byte_len(&self) -> usize {
        F::fixed_canonical_byte_len()
    }

    fn write_canonical(&self, value: &Self::Element, out: &mut Vec<u8>) {
        value.write_canonical(out);
    }

    fn from_canonical_bytes(&self, bytes: &[u8]) -> Result<Self::Element, CanonicalBytesError> {
        F::from_canonical_bytes(bytes)
    }
}

//
// Primitive integers
//

macro_rules! impl_primitive_int_bytes {
    ($($t:ty),+ $(,)?) => {
        $(
            impl CanonicalIntBytes for $t {
                #[inline]
                fn bit_len(&self) -> u32 {
                    <$t>::BITS.saturating_sub(self.leading_zeros())
                }

                fn write_le(
                    &self,
                    width: usize,
                    out: &mut Vec<u8>,
                ) -> Result<(), CanonicalBytesError> {
                    let bytes = self.to_le_bytes();
                    if bits_to_bytes(self.bit_len()) > width {
                        return Err(CanonicalBytesError::Overflow { width });
                    }
                    let taken = bytes.len().min(width);
                    out.extend_from_slice(&bytes[..taken]);
                    out.resize(out.len().saturating_add(width.saturating_sub(taken)), 0);
                    Ok(())
                }

                fn read_le_like(bytes: &[u8], _like: &Self) -> Result<Self, CanonicalBytesError> {
                    let width = size_of::<$t>();
                    let mut buf = [0_u8; size_of::<$t>()];
                    let taken = bytes.len().min(width);
                    if bytes[taken..].iter().any(|byte| *byte != 0) {
                        return Err(CanonicalBytesError::Overflow { width });
                    }
                    buf[..taken].copy_from_slice(&bytes[..taken]);
                    Ok(<$t>::from_le_bytes(buf))
                }
            }

            impl CanonicalBytes for $t {
                #[inline]
                fn canonical_byte_len(&self) -> usize {
                    size_of::<$t>()
                }

                fn write_canonical(&self, out: &mut Vec<u8>) {
                    out.extend_from_slice(&self.to_le_bytes());
                }

                fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, CanonicalBytesError> {
                    let expected = size_of::<$t>();
                    if bytes.len() != expected {
                        return Err(CanonicalBytesError::InvalidLength {
                            expected,
                            actual: bytes.len(),
                        });
                    }
                    Self::read_le_like(bytes, &0)
                }
            }

            impl FixedCanonicalBytes for $t {
                #[inline]
                fn fixed_canonical_byte_len() -> usize {
                    size_of::<$t>()
                }
            }
        )+
    };
}

impl_primitive_int_bytes!(u8, u16, u32, u64, u128);

//
// Cross-backend agreement
//

#[cfg(all(test, feature = "crypto_bigint"))]
mod backend_tests {
    use super::*;
    use crate::{
        BaseFieldConfig, ConstBaseField, LiftElement, LiftElementWithConfig,
        crypto_bigint_boxed_monty::BoxedMontyField, crypto_bigint_boxed_uint::BoxedUint,
        crypto_bigint_const_monty::ConstMontyField, crypto_bigint_monty::MontyField,
        crypto_bigint_uint::Uint,
    };
    use alloc::vec;
    use crypto_bigint::{Resize, U256, const_monty_params};

    /// secp256k1 field prime, 2^256 - 2^32 - 977. 256 bits, so 32 bytes.
    const MODULUS_HEX: &str = "fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f";
    const WIDTH: usize = 32;

    const_monty_params!(ModP, U256, MODULUS_HEX);
    type ConstF = ConstMontyField<ModP, { U256::LIMBS }>;

    fn monty() -> MontyField<{ U256::LIMBS }> {
        MontyField::new(&Uint::new(U256::from_be_hex(MODULUS_HEX))).expect("valid modulus")
    }

    fn boxed(precision: u32) -> BoxedMontyField {
        let modulus = BoxedUint::new(
            crypto_bigint::BoxedUint::from_be_hex(MODULUS_HEX, 256).expect("valid hex"),
        );
        let modulus = BoxedUint::new(modulus.0.resize_unchecked(precision));
        BoxedMontyField::new(&modulus).expect("valid modulus")
    }

    /// The bytes of `value` under each backend that can hold this modulus.
    fn all_encodings(value: u64) -> vec::Vec<vec::Vec<u8>> {
        let const_bytes = ConstF::from(value).to_canonical_bytes();

        let m = monty();
        let monty_bytes = m.to_canonical_bytes(&m.project(&value));

        let b256 = boxed(256);
        let boxed_bytes = b256.to_canonical_bytes(&b256.project(&value));

        let b320 = boxed(320);
        let boxed_wide_bytes = b320.to_canonical_bytes(&b320.project(&value));

        let mut all = vec![const_bytes, monty_bytes, boxed_bytes, boxed_wide_bytes];
        all.extend(ark_encodings(value));
        all
    }

    #[cfg(not(feature = "ark_ff"))]
    fn ark_encodings(_value: u64) -> vec::Vec<vec::Vec<u8>> {
        vec::Vec::new()
    }

    /// The same modulus through the two arkworks backends.
    #[cfg(feature = "ark_ff")]
    fn ark_encodings(value: u64) -> vec::Vec<vec::Vec<u8>> {
        use crate::{ark_ff_field::ArkField, ark_ff_fp::Fp};
        use ark_ff::{Fp256, MontBackend, MontConfig};

        #[derive(MontConfig)]
        #[modulus = "115792089237316195423570985008687907853269984665640564039457584007908834671663"]
        #[generator = "3"]
        pub struct TestConfig;

        type ArkF = ArkField<Fp256<MontBackend<TestConfig, 4>>>;
        type FpF = Fp<MontBackend<TestConfig, 4>, 4>;

        vec![
            ArkF::from(value).to_canonical_bytes(),
            FpF::from(value).to_canonical_bytes(),
        ]
    }

    #[test]
    fn backends_agree_byte_for_byte() {
        // Guard against the comparison going vacuous if a backend drops out.
        let expected_backends = if cfg!(feature = "ark_ff") { 6 } else { 4 };
        assert_eq!(all_encodings(1).len(), expected_backends);

        for value in [0_u64, 1, 2, 3, 255, 256, u64::MAX] {
            let encodings = all_encodings(value);
            for bytes in &encodings {
                assert_eq!(bytes.len(), WIDTH, "width must follow the modulus");
                assert_eq!(
                    bytes, &encodings[0],
                    "backends disagree on the encoding of {value}"
                );
            }
        }
    }

    #[test]
    fn boxed_width_ignores_precision() {
        // 320-bit precision holds the same 256-bit modulus. The width must not
        // follow the allocation.
        assert_eq!(boxed(256).canonical_byte_len(), WIDTH);
        assert_eq!(boxed(320).canonical_byte_len(), WIDTH);
    }

    #[test]
    fn known_answer_vectors() {
        let three = all_encodings(3);
        let mut expected = [0_u8; WIDTH];
        expected[0] = 3;
        assert_eq!(three[0].as_slice(), &expected);

        let zero = all_encodings(0);
        assert_eq!(zero[0].as_slice(), &[0_u8; WIDTH]);

        // p - 1 ends in 0x2e, little-endian, and the top bytes are 0xff.
        let minus_one = ConstF::from(0_u64) - ConstF::from(1_u64);
        let bytes = minus_one.to_canonical_bytes();
        assert_eq!(bytes[0], 0x2e);
        assert_eq!(bytes[WIDTH - 1], 0xff);
    }

    #[test]
    fn round_trips() {
        for value in [0_u64, 1, 7, u64::MAX] {
            let element = ConstF::from(value);
            let bytes = element.to_canonical_bytes();
            assert_eq!(ConstF::from_canonical_bytes(&bytes), Ok(element));

            let m = monty();
            let element = m.project(&value);
            let bytes = m.to_canonical_bytes(&element);
            assert_eq!(m.from_canonical_bytes(&bytes), Ok(element));

            let b = boxed(320);
            let element = b.project(&value);
            let bytes = b.to_canonical_bytes(&element);
            assert_eq!(b.from_canonical_bytes(&bytes), Ok(element));
        }
    }

    #[test]
    fn rejects_the_modulus_itself() {
        let modulus = <ConstF as ConstBaseField>::MODULUS;
        let mut bytes = vec::Vec::new();
        modulus.write_le(WIDTH, &mut bytes).expect("fits");
        assert_eq!(
            ConstF::from_canonical_bytes(&bytes),
            Err(CanonicalBytesError::NonCanonical)
        );

        let m = monty();
        assert_eq!(
            m.from_canonical_bytes(&bytes),
            Err(CanonicalBytesError::NonCanonical)
        );
    }

    #[test]
    fn rejects_all_ones() {
        let bytes = [0xff_u8; WIDTH];
        assert_eq!(
            ConstF::from_canonical_bytes(&bytes),
            Err(CanonicalBytesError::NonCanonical)
        );
    }

    #[test]
    fn rejects_wrong_length() {
        assert_eq!(
            ConstF::from_canonical_bytes(&[0; WIDTH - 1]),
            Err(CanonicalBytesError::InvalidLength {
                expected: WIDTH,
                actual: WIDTH - 1
            })
        );
        let m = monty();
        assert!(m.from_canonical_bytes(&[0; WIDTH + 1]).is_err());
    }

    #[test]
    fn boxed_uint_encoding_is_unambiguous() {
        // Two 64-bit values back to back, against one 128-bit value holding the
        // same bytes. Without the length prefix both would hash alike.
        let a = BoxedUint::new(crypto_bigint::BoxedUint::from(7_u64));
        let b = BoxedUint::new(crypto_bigint::BoxedUint::from(5_u64));
        let mut pair = a.to_canonical_bytes();
        pair.extend_from_slice(&b.to_canonical_bytes());

        let wide = BoxedUint::new(
            crypto_bigint::BoxedUint::from(7_u64)
                .resize_unchecked(128)
                .shl_vartime(64)
                .expect("shift fits")
                | crypto_bigint::BoxedUint::from(5_u64).resize_unchecked(128),
        );
        assert_ne!(pair, wide.to_canonical_bytes());
    }

    #[test]
    fn boxed_uint_ignores_precision() {
        let narrow = BoxedUint::new(crypto_bigint::BoxedUint::from(5_u64));
        let wide = BoxedUint::new(crypto_bigint::BoxedUint::from(5_u64).resize_unchecked(512));
        assert_eq!(narrow.to_canonical_bytes(), wide.to_canonical_bytes());
        assert_eq!(
            BoxedUint::from_canonical_bytes(&narrow.to_canonical_bytes()),
            Ok(narrow)
        );
    }

    #[test]
    fn uniform_sampling_reduces_the_input() {
        // A short input is below the modulus, so it must map to itself.
        for value in [0_u8, 1, 5, 200] {
            assert_eq!(
                ConstF::from_uniform_bytes(&[value]),
                ConstF::from(u64::from(value))
            );
        }
        // 2^64 little-endian, which is not below the modulus but is exact.
        let mut bytes = [0_u8; 9];
        bytes[8] = 1;
        let expected = ConstF::from(1_u64 << 32) * ConstF::from(1_u64 << 32);
        assert_eq!(ConstF::from_uniform_bytes(&bytes), expected);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serde_round_trips_and_matches_canonical_bytes() {
        let element = ConstF::from(12345_u64);
        let json = serde_json::to_string(&element).expect("serializes");
        let back: ConstF = serde_json::from_str(&json).expect("deserializes");
        assert_eq!(back, element);

        // The `serde` payload is the canonical encoding, not the Montgomery one.
        let payload: vec::Vec<u8> = serde_json::from_str(&json).expect("byte sequence");
        assert_eq!(payload, element.to_canonical_bytes());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serde_rejects_a_non_canonical_payload() {
        let bytes = [0xff_u8; WIDTH];
        let json = serde_json::to_string(&bytes.to_vec()).expect("serializes");
        assert!(serde_json::from_str::<ConstF>(&json).is_err());
    }

    #[test]
    fn uniform_sampling_matches_between_paths() {
        let bytes: vec::Vec<u8> = (0..48_u8).collect();
        let m = monty();
        assert_eq!(m.uniform_byte_len(), WIDTH + UNIFORM_BYTES_MARGIN);
        let from_config = m.from_uniform_bytes(&bytes);
        let from_element = ConstF::from_uniform_bytes(&bytes);
        assert_eq!(m.lift(&from_config), from_element.lift());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primitive_round_trip() {
        for value in [0_u64, 1, 255, 256, u64::MAX] {
            let bytes = value.to_canonical_bytes();
            assert_eq!(bytes.len(), 8);
            assert_eq!(u64::from_canonical_bytes(&bytes), Ok(value));
        }
    }

    #[test]
    fn primitive_rejects_wrong_length() {
        assert_eq!(
            u64::from_canonical_bytes(&[0; 7]),
            Err(CanonicalBytesError::InvalidLength {
                expected: 8,
                actual: 7
            })
        );
    }

    #[test]
    fn width_follows_the_modulus() {
        // 251 needs 8 bits, so one byte, whatever holds it.
        assert_eq!(canonical_width(&251_u64), 1);
        assert_eq!(canonical_width(&251_u8), 1);
        // 2 needs 2 bits, so still one byte.
        assert_eq!(canonical_width(&2_u8), 1);
    }

    #[test]
    fn write_le_pads_and_rejects_overflow() {
        let mut out = Vec::new();
        assert!(3_u64.write_le(1, &mut out).is_ok());
        assert_eq!(out, [3]);

        out.clear();
        assert!(3_u64.write_le(4, &mut out).is_ok());
        assert_eq!(out, [3, 0, 0, 0]);

        out.clear();
        assert_eq!(
            256_u64.write_le(1, &mut out),
            Err(CanonicalBytesError::Overflow { width: 1 })
        );
    }
}
