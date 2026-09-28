//! Contains helper utility functions and macros not visible to the outside
//! world.

#[cfg(feature = "crypto_bigint")]
pub(crate) mod crypto_bigint;

/// Define an empty trait with the given supertraits, and make a blanket
/// implementation for it.
macro_rules! define_blanket_trait {
    ($(#[$attr:meta])* $vis:vis trait $trait_name:ident: $($bound:tt)+) => {
        $(#[$attr])*
        $vis trait $trait_name: $($bound)+ {}

        impl<T> $trait_name for T where T: $($bound)* {}
    };
}
pub(crate) use define_blanket_trait;

/// Implement exponentiation using repeated squaring
#[cfg(any(feature = "ark_ff", feature = "crypto_bigint"))]
macro_rules! pow_via_repeated_squaring {
    ($self:expr, $rhs:expr, $one:expr) => {{
        if $rhs == 0 {
            return $one;
        }

        let mut base = $self;
        let mut result = $one;
        let mut exp = $rhs;

        while exp > 0 {
            if exp & 1 == 1 {
                result = result
                    .checked_mul(&base)
                    .expect("overflow in exponentiation");
            }
            exp >>= 1;
            if exp > 0 {
                base = base.checked_mul(&base).expect("overflow in exponentiation");
            }
        }

        result
    }};
}
#[cfg(any(feature = "ark_ff", feature = "crypto_bigint"))]
pub(crate) use pow_via_repeated_squaring;

#[cfg(any(feature = "ark_ff", feature = "crypto_bigint"))]
pub(crate) trait ToFloatHelper {
    /// Returns `(m, e)` with `self ~= m * 2^e`, where `m` is the top 64 bits of
    /// `self` rounded to odd: its LSB is set if any truncated bit is set, so that rounding $m$
    /// to a float with round-to-nearest-even is correct. Mirrors `num_bigint`'s
    /// `high_bits_to_u64`.
    fn float_mantissa_and_exponent(&self) -> (u64, u32);
}

#[cfg(any(feature = "ark_ff", feature = "crypto_bigint"))]
macro_rules! impl_to_float {
    () => {
        #[allow(clippy::cast_precision_loss)] // Rounding is intended
        #[inline]
        fn to_f32(&self) -> Option<f32> {
            let (mantissa, exponent) = self.float_mantissa_and_exponent();
            match i32::try_from(exponent) {
                Ok(exponent) if exponent <= f32::MAX_EXP => {
                    Some(mantissa as f32 * FloatCore::powi(2.0_f32, exponent))
                }
                _ => Some(f32::INFINITY),
            }
        }

        #[allow(clippy::cast_precision_loss)] // Rounding is intended
        #[inline]
        fn to_f64(&self) -> Option<f64> {
            let (mantissa, exponent) = self.float_mantissa_and_exponent();
            match i32::try_from(exponent) {
                Ok(exponent) if exponent <= f64::MAX_EXP => {
                    Some(mantissa as f64 * FloatCore::powi(2.0_f64, exponent))
                }
                _ => Some(f64::INFINITY),
            }
        }
    };
}
#[cfg(any(feature = "ark_ff", feature = "crypto_bigint"))]
pub(crate) use impl_to_float;

/// Decomposes `n`, truncated toward zero, as `m * 2^e` with `m < 2^53`. Returns `None` if `n` is
/// not finite or truncates to a negative integer.
#[cfg(any(feature = "ark_ff", feature = "crypto_bigint"))]
pub(crate) fn decompose_truncated_f64(n: f64) -> Option<(u64, u32)> {
    use num_traits::{Zero, float::FloatCore};

    if !n.is_finite() {
        return None;
    }
    let n = FloatCore::trunc(n);
    if n.is_zero() {
        return Some((0, 0));
    }
    let (mantissa, exponent, sign) = FloatCore::integer_decode(n);
    if sign < 0 {
        return None;
    }
    match u32::try_from(exponent) {
        Ok(exponent) => Some((mantissa, exponent)),
        // `n` is an integer, so the shifted-out bits are zero
        Err(_) => Some((mantissa >> exponent.unsigned_abs(), 0)),
    }
}

/// Will fail compilation if trait is not implemented for the type.
#[cfg(test)]
#[macro_export]
macro_rules! ensure_type_implements_trait {
    ($type_name:ty, $trait_name:path) => {{
        fn _assert_impl<T: $trait_name>() {}
        _assert_impl::<$type_name>();
    }};
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
pub(crate) use delegate_to_ref_binary;
