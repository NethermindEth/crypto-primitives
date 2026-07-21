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
pub(crate) use pow_via_repeated_squaring;

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
