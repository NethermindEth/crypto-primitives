//! Canonical byte serialization intended for Fiat-Shamir transcripts.
//!
//!
//! - [`CanonicalBytes`] serializes self-sufficient elements.
//! - [`CanonicalBytesWithConfig`] serializes elements that need a config such as a field with a
//!   runtime modulus.
//! - [`FromUniformBytes`] and [`FromUniformBytesWithConfig`] map squeezed transcript output to an
//!   element. They are a different map, and they are not injective. Never use them to read a prover
//!   message.
//!
//! # Security
//!
//! Encoding and decoding are not constant-time. Use them only on public data,
//! such as transcript messages. Never use them on secrets.

use crate::{BaseFieldConfig, ProjectElementWithConfig, SetConfig};
use alloc::vec::Vec;
use thiserror::Error;

/// Bytes drawn beyond the modulus width when sampling an element from
/// transcript output. Bounds the sampling bias by `2^-128`.
pub(crate) const UNIFORM_BYTES_MARGIN: usize = 16;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum CanonicalBytesError {
    #[error("expected {expected} bytes, got {actual}")]
    InvalidLength { expected: usize, actual: usize },
    #[error("value is not reduced modulo the modulus")]
    NonCanonical,
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
    ///
    /// Not constant-time. See the
    /// [module docs](self#security).
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

    /// Maps uniform bytes to a near-uniform element, non-injective.
    ///
    /// # Panics
    ///
    /// If `bytes.len()` is not [`Self::uniform_byte_len`]. A fixed length keeps
    /// the challenge map the same for the prover and the verifier.
    fn from_uniform_bytes(bytes: &[u8]) -> Self;
}

//
// Config-side traits
//

/// Canonical byte encoding for elements that need a config.
///
/// The config pins the modulus so the width is constant for one config
/// instance.
#[allow(
    clippy::wrong_self_convention,
    reason = "kept symmetric with the element-side trait, as `lift` already is"
)]
pub trait CanonicalBytesWithConfig: SetConfig {
    /// Number of bytes [`Self::write_canonical`] appends for any element.
    fn canonical_byte_len(&self) -> usize;

    fn write_canonical(&self, value: &Self::Element, out: &mut Vec<u8>);

    /// Not constant-time. Use only on public data, see the
    /// [module docs](self#security).
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

    /// # Panics
    ///
    /// If `bytes.len()` is not [`Self::uniform_byte_len`].
    fn from_uniform_bytes(&self, bytes: &[u8]) -> Self::Element;
}

//
// Integer layer
//

/// Little-endian byte access for the integer types that back our sets.
pub(crate) trait CanonicalIntBytes: Sized {
    /// Number of bits needed to hold this value. Zero for zero.
    fn bit_len(&self) -> u32;

    /// Append exactly `width` little-endian bytes.
    ///
    /// # Panics
    ///
    /// If the value needs more than `width` bytes.
    fn write_le(&self, width: usize, out: &mut Vec<u8>);

    /// Read a little-endian value from `bytes`.
    ///
    /// # Panics
    ///
    /// If `bytes` is wider than the integer type.
    fn read_le(bytes: &[u8]) -> Self;
}

/// Bytes a base field element occupies: `ceil(modulus_bits / 8)`.
///
/// The width follows the modulus, never the limb count or a runtime precision.
#[inline]
pub(crate) fn canonical_width<I: CanonicalIntBytes>(modulus: &I) -> usize {
    usize::try_from(modulus.bit_len().div_ceil(8)).expect("u32 fits in usize on supported targets")
}

//
// Shared base field logic
//

pub(crate) fn write_base_field<C>(cfg: &C, value: &C::Element, out: &mut Vec<u8>)
where
    C: BaseFieldConfig,
    C::Integer: CanonicalIntBytes,
{
    let width = canonical_width(&cfg.modulus());
    cfg.lift(value).write_le(width, out);
}

pub(crate) fn read_base_field<C>(cfg: &C, bytes: &[u8]) -> Result<C::Element, CanonicalBytesError>
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
    let value = C::Integer::read_le(bytes);
    if value >= modulus {
        return Err(CanonicalBytesError::NonCanonical);
    }
    Ok(cfg.project(&value))
}

//
//  uniform sampling logic
//

/// Reduces `bytes` into the field with base 2^8 Horner's rule.
///
/// # Panics
///
/// Panics if `bytes.len()` is not `uniform_width(&cfg.modulus())`.
pub(crate) fn base_field_from_uniform_bytes<C>(cfg: &C, bytes: &[u8]) -> C::Element
where
    C: BaseFieldConfig + ProjectElementWithConfig<u64>,
    C::Integer: CanonicalIntBytes,
{
    assert!(
        bytes.len() == uniform_width(&cfg.modulus()),
        "uniform sampling needs exactly the uniform byte length"
    );
    let radix = cfg.project(&(256_u64));
    let mut acc = cfg.project(&0_u64);

    for &byte in bytes.iter().rev() {
        acc = cfg.add(&cfg.mul(&acc, &radix), &cfg.project(&u64::from(byte)));
    }

    acc
}

/// Bytes needed to sample a base field element with bias at most `2^-128`.
#[inline]
pub(crate) fn uniform_width<I: CanonicalIntBytes>(modulus: &I) -> usize {
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

impl<F> FromUniformBytesWithConfig for crate::FixedConfig<F>
where
    F: crate::SetElement + FromUniformBytes,
{
    fn uniform_byte_len(&self) -> usize {
        F::uniform_byte_len()
    }

    fn from_uniform_bytes(&self, bytes: &[u8]) -> Self::Element {
        F::from_uniform_bytes(bytes)
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

                fn write_le(&self, width: usize, out: &mut Vec<u8>) {
                    assert!(
                        canonical_width(self) <= width,
                        "value does not fit into {width} bytes"
                    );
                    let bytes = self.to_le_bytes();
                    let taken = bytes.len().min(width);
                    out.extend_from_slice(&bytes[..taken]);
                    out.resize(out.len().saturating_add(width.saturating_sub(taken)), 0);
                }

                fn read_le(bytes: &[u8]) -> Self {
                    assert!(
                        bytes.len() <= size_of::<$t>(),
                        "input is wider than the integer type"
                    );
                    let mut buf = [0_u8; size_of::<$t>()];
                    buf[..bytes.len()].copy_from_slice(bytes);
                    <$t>::from_le_bytes(buf)
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
                    Ok(Self::read_le(bytes))
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
        BaseFieldConfig, LiftElement, LiftElementWithConfig,
        crypto_bigint_boxed_monty::BoxedMontyField, crypto_bigint_boxed_uint::BoxedUint,
        crypto_bigint_const_monty::ConstMontyField, crypto_bigint_int::Int,
        crypto_bigint_monty::MontyField, crypto_bigint_uint::Uint,
    };
    use alloc::vec;
    use crypto_bigint::{Resize, U256, const_monty_params};
    use proptest::prelude::*;

    /// secp256k1 field prime, 2^256 - 2^32 - 977. 256 bits, so 32 bytes.
    const MODULUS_HEX: &str = "fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f";
    const WIDTH: usize = 32;
    /// `2^256 - p`, the count of 32-byte values at or above the modulus.
    const ABOVE_MODULUS: u64 = 4_294_968_273;

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

    /// A decode followed by an encode of the decoded value.
    type Reencoded = Result<vec::Vec<u8>, CanonicalBytesError>;

    /// Decodes `bytes` as `T` and encodes the result again.
    fn reencode<T: CanonicalBytes>(bytes: &[u8]) -> Reencoded {
        T::from_canonical_bytes(bytes).map(|value| value.to_canonical_bytes())
    }

    /// The bytes of `value` under each backend that can hold this modulus.
    fn all_encodings(value: i128) -> vec::Vec<vec::Vec<u8>> {
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

    /// [`reencode`] under each backend that can hold this modulus.
    fn all_decodings(bytes: &[u8]) -> vec::Vec<Reencoded> {
        let m = monty();
        let b256 = boxed(256);
        let b320 = boxed(320);
        let mut all = vec![
            reencode::<ConstF>(bytes),
            m.from_canonical_bytes(bytes)
                .map(|value| m.to_canonical_bytes(&value)),
            b256.from_canonical_bytes(bytes)
                .map(|value| b256.to_canonical_bytes(&value)),
            b320.from_canonical_bytes(bytes)
                .map(|value| b320.to_canonical_bytes(&value)),
        ];
        all.extend(ark_decodings(bytes));
        all
    }

    /// `p + offset` as `WIDTH` little-endian bytes.
    fn modulus_plus(offset: u64) -> vec::Vec<u8> {
        let value = U256::from_be_hex(MODULUS_HEX).wrapping_add(&U256::from(offset));
        crypto_bigint::Encoding::to_le_bytes(&value)
            .as_ref()
            .to_vec()
    }

    #[cfg(not(feature = "ark_ff"))]
    fn ark_encodings(_value: i128) -> vec::Vec<vec::Vec<u8>> {
        vec::Vec::new()
    }

    #[cfg(not(feature = "ark_ff"))]
    fn ark_decodings(_bytes: &[u8]) -> vec::Vec<Reencoded> {
        vec::Vec::new()
    }

    /// The same modulus through the two arkworks backends.
    #[cfg(feature = "ark_ff")]
    mod ark {
        use ark_ff::{Fp256, MontBackend, MontConfig};

        #[derive(MontConfig)]
        #[modulus = "115792089237316195423570985008687907853269984665640564039457584007908834671663"]
        #[generator = "3"]
        pub struct TestConfig;

        pub type ArkF = crate::ark_ff_field::ArkField<Fp256<MontBackend<TestConfig, 4>>>;
        pub type FpF = crate::ark_ff_fp::Fp<MontBackend<TestConfig, 4>, 4>;
    }

    #[cfg(feature = "ark_ff")]
    fn ark_encodings(value: i128) -> vec::Vec<vec::Vec<u8>> {
        vec![
            ark::ArkF::from(value).to_canonical_bytes(),
            ark::FpF::from(value).to_canonical_bytes(),
        ]
    }

    #[cfg(feature = "ark_ff")]
    fn ark_decodings(bytes: &[u8]) -> vec::Vec<Reencoded> {
        vec![reencode::<ark::ArkF>(bytes), reencode::<ark::FpF>(bytes)]
    }

    #[test]
    fn boxed_uint_rejects_a_zero_width_payload() {
        // Regression: `[0, 0, 0, 0]` used to decode to zero, giving zero a
        // second encoding alongside its canonical one.
        assert_eq!(
            BoxedUint::from_canonical_bytes(&[0, 0, 0, 0]),
            Err(CanonicalBytesError::NonCanonical)
        );
    }

    #[test]
    fn boxed_uint_zero_has_exactly_one_encoding() {
        let zero = BoxedUint::new(crypto_bigint::BoxedUint::zero());
        let bytes = zero.to_canonical_bytes();
        assert_eq!(bytes, vec![1, 0, 0, 0, 0]);
        assert_eq!(BoxedUint::from_canonical_bytes(&bytes), Ok(zero));
    }

    #[test]
    fn boxed_uint_encoding_ignores_precision() {
        let narrow = BoxedUint::new(crypto_bigint::BoxedUint::from(5_u64));
        let wide = BoxedUint::new(crypto_bigint::BoxedUint::from(5_u64).resize_unchecked(512));
        let bytes = wide.to_canonical_bytes();
        assert_eq!(bytes, narrow.to_canonical_bytes());

        // The encoding carries no precision, so decoding gives the minimal one.
        let back = BoxedUint::from_canonical_bytes(&bytes).expect("decodes");
        assert_eq!(back, wide);
        assert!(back.0.bits_precision() < wide.0.bits_precision());
    }

    #[test]
    #[should_panic(expected = "uniform sampling needs exactly the uniform byte length")]
    fn uniform_sampling_rejects_short_input() {
        let _ = ConstF::from_uniform_bytes(&[]);
    }

    #[test]
    #[should_panic(expected = "uniform sampling needs exactly the uniform byte length")]
    fn uniform_sampling_rejects_long_input() {
        let _ = ConstF::from_uniform_bytes(&[0; WIDTH + UNIFORM_BYTES_MARGIN + 1]);
    }

    #[test]
    fn backends_agree_byte_for_byte() {
        // Guard against the comparison going vacuous if a backend drops out.
        let expected_backends = if cfg!(feature = "ark_ff") { 6 } else { 4 };
        assert_eq!(all_encodings(1).len(), expected_backends);

        for value in [0, 1, 2, 3, 255, 256, i128::from(u64::MAX)] {
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

        // p - 1 = 2^256 - 2^32 - 978 sets every limb, so it pins the limb
        // order of every backend.
        let mut minus_one = [0xff_u8; WIDTH];
        minus_one[..5].copy_from_slice(&[0x2e, 0xfc, 0xff, 0xff, 0xfe]);
        for bytes in all_encodings(-1) {
            assert_eq!(bytes, minus_one);
        }

        // The two edges of the range decode under every backend.
        for edge in [[0_u8; WIDTH], minus_one] {
            for decoded in all_decodings(&edge) {
                assert_eq!(decoded, Ok(edge.to_vec()));
            }
        }
    }

    #[test]
    fn fixed_width_ints_are_little_endian() {
        let low = [1_u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let mut expected = [0_u8; WIDTH];
        expected[..16].copy_from_slice(&low);
        assert_eq!(
            Uint::<{ U256::LIMBS }>::from(u128::from_le_bytes(low)).to_canonical_bytes(),
            expected
        );
        // Two's complement sign extension.
        assert_eq!(
            Int::<{ U256::LIMBS }>::from(-1_i64).to_canonical_bytes(),
            [0xff_u8; WIDTH]
        );
        #[cfg(feature = "ark_ff")]
        {
            let mut expected = [0_u8; WIDTH];
            expected[..8].copy_from_slice(&low[..8]);
            let value = u64::from_le_bytes([1, 2, 3, 4, 5, 6, 7, 8]);
            assert_eq!(
                crate::ark_ff_bigint::BigInt::<4>::from(value).to_canonical_bytes(),
                expected
            );
        }
    }

    proptest! {
        #[test]
        fn backends_round_trip_any_element(
            seed in proptest::collection::vec(any::<u8>(), WIDTH + UNIFORM_BYTES_MARGIN)
        ) {
            let bytes = ConstF::from_uniform_bytes(&seed).to_canonical_bytes();
            for decoded in all_decodings(&bytes) {
                prop_assert_eq!(decoded, Ok(bytes.clone()));
            }
        }

        #[test]
        fn backends_reject_every_value_from_the_modulus_up(offset in 0..ABOVE_MODULUS) {
            let bytes = modulus_plus(offset);
            for decoded in all_decodings(&bytes) {
                prop_assert_eq!(decoded, Err(CanonicalBytesError::NonCanonical));
            }
        }

        #[test]
        fn fixed_width_ints_accept_every_bit_pattern(bytes in any::<[u8; WIDTH]>()) {
            #[allow(unused_mut, reason = "only the ark_ff build pushes")]
            let mut decoders: vec::Vec<fn(&[u8]) -> Reencoded> =
                vec![reencode::<Uint<{ U256::LIMBS }>>, reencode::<Int<{ U256::LIMBS }>>];
            #[cfg(feature = "ark_ff")]
            decoders.push(reencode::<crate::ark_ff_bigint::BigInt<4>>);
            for decode in decoders {
                prop_assert_eq!(decode(&bytes), Ok(bytes.to_vec()));
            }
        }
    }

    #[test]
    fn backends_reject_both_ends_of_the_non_canonical_range() {
        // p itself, and 2^256 - 1 (all bytes 0xff).
        for bytes in [modulus_plus(0), modulus_plus(ABOVE_MODULUS - 1)] {
            for decoded in all_decodings(&bytes) {
                assert_eq!(decoded, Err(CanonicalBytesError::NonCanonical));
            }
        }
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
    fn uniform_sampling_reduces_the_input() {
        let full = ConstF::uniform_byte_len();

        // A small value inside a full-width buffer must map to itself.
        for value in [0_u8, 1, 5, 200] {
            let mut bytes = vec::Vec::from_iter(core::iter::repeat_n(0_u8, full));
            bytes[0] = value;
            assert_eq!(
                ConstF::from_uniform_bytes(&bytes),
                ConstF::from(u64::from(value))
            );
        }

        // 2^64 little-endian, above a u64 but far below the modulus, so exact.
        let mut bytes = vec::Vec::from_iter(core::iter::repeat_n(0_u8, full));
        bytes[8] = 1;
        let expected = ConstF::from(1_u64 << 32) * ConstF::from(1_u64 << 32);
        assert_eq!(ConstF::from_uniform_bytes(&bytes), expected);
    }

    /// `MontyFieldElement` and `BoxedMontyFieldElement` keep a `serde` impl for
    /// storage, but it writes the Montgomery residue. Pin that difference so it
    /// cannot be mistaken for the canonical encoding.
    #[test]
    fn element_serde_payload_is_not_the_canonical_encoding() {
        use crate::crypto_bigint_monty::MontyFieldElement;

        let m = monty();
        let element: MontyFieldElement<{ U256::LIMBS }> = m.project(&3_u64);

        // `element.0` is what `serde` writes: the raw Montgomery residue.
        let stored = element.0.to_canonical_bytes();
        let canonical = m.to_canonical_bytes(&element);

        assert_eq!(stored.len(), canonical.len());
        assert_ne!(
            stored, canonical,
            "serde must not be mistaken for canonical"
        );

        let mut expected = [0_u8; WIDTH];
        expected[0] = 3;
        assert_eq!(canonical.as_slice(), &expected);
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

    /// Bytes 0 and 1 round-trip. Every other byte is rejected.
    fn assert_only_zero_and_one<T: CanonicalBytes>() {
        for byte in 0..=u8::MAX {
            match T::from_canonical_bytes(&[byte]) {
                Ok(value) => {
                    assert!(byte <= 1, "accepted {byte}");
                    assert_eq!(value.to_canonical_bytes(), [byte]);
                }
                Err(err) => {
                    assert!(byte > 1, "rejected {byte}");
                    assert_eq!(err, CanonicalBytesError::NonCanonical);
                }
            }
        }
    }

    #[test]
    fn single_byte_types_accept_only_zero_and_one() {
        assert_only_zero_and_one::<crate::f2::F2>();
        assert_only_zero_and_one::<crate::boolean::Boolean>();
    }

    #[test]
    fn boolean_rejects_wrong_length() {
        for bytes in [&[][..], &[0, 0]] {
            assert_eq!(
                crate::boolean::Boolean::from_canonical_bytes(bytes),
                Err(CanonicalBytesError::InvalidLength {
                    expected: 1,
                    actual: bytes.len()
                })
            );
        }
    }

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
    fn write_le_pads_to_width() {
        let mut out = Vec::new();
        3_u64.write_le(1, &mut out);
        assert_eq!(out, [3]);

        out.clear();
        3_u64.write_le(4, &mut out);
        assert_eq!(out, [3, 0, 0, 0]);
    }

    #[test]
    #[should_panic(expected = "value does not fit into 1 bytes")]
    fn write_le_rejects_overflow() {
        256_u64.write_le(1, &mut Vec::new());
    }
}
