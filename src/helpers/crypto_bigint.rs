#![allow(clippy::arithmetic_side_effects, clippy::needless_range_loop)]
use crypto_bigint::Limb;

/// How many words/limbs there are per 64 bits.
#[cfg(target_pointer_width = "64")]
pub const WORD_FACTOR: usize = 1;
/// How many words/limbs there are per 64 bits.
#[cfg(target_pointer_width = "32")]
pub const WORD_FACTOR: usize = 2;

/// Copy-paste from `crypto-bigint`
#[inline(always)]
pub const fn monty_retrieve_inner(
    x: &[Limb],
    out: &mut [Limb],
    modulus: &[Limb],
    mod_neg_inv: Limb,
) {
    let nlimbs = modulus.len();
    assert!(nlimbs == x.len() && nlimbs == out.len());

    let mut i = 0;
    while i < nlimbs {
        let xi = x[i];
        let u = out[0].wrapping_add(xi).wrapping_mul(mod_neg_inv);

        out[0] = u.carrying_mul_add(modulus[0], xi, out[0]).1;

        let mut j = 1;
        while j < nlimbs {
            (out[j - 1], out[j]) = u.carrying_mul_add(modulus[j], out[j], out[j - 1]);
            j += 1;
        }

        i += 1;
    }
}

/// Optimized Montgomery multiplication.
///
/// Uses CIOS (Coarsely Integrated Operand Scanning) method which is more
/// cache-friendly than the FIOS method used in crypto-bigint.
/// Based on the implementation described in the Section 5 of the paper
/// "Analyzing and comparing Montgomery multiplication algorithms" with some
/// slight tweaks: <https://www.microsoft.com/en-us/research/wp-content/uploads/1996/01/j37acmon.pdf>
pub mod mul {
    use crypto_bigint::{Limb, Uint, WideWord, Word};
    use num_traits::ConstZero;

    const LOG2_WORD_BITS: u32 = Word::BITS.trailing_zeros();

    /// Compute modulus^-1 mod 2^word_bits (the negative of it, for Montgomery
    /// reduction). Uses Newton's method: x_{n+1} = x_n * (2 - m * x_n)
    #[inline(always)]
    #[allow(clippy::arithmetic_side_effects)]
    const fn compute_mod_neg_inv(m0: Word) -> Word {
        const TWO: Word = 2;

        let mut inv: Word = 1;
        let mut i = 0;
        // Newton iterations - converges in log2(word_bits) = 6 iterations for 64-bit
        while i < LOG2_WORD_BITS {
            inv = inv.wrapping_mul(TWO.wrapping_sub(m0.wrapping_mul(inv)));
            i += 1;
        }
        inv.wrapping_neg()
    }

    /// Montgomery multiplication: compute `a * b * R^-1 mod m` more efficiently
    /// than crypto-bigint does it.
    ///
    /// Uses CIOS (Coarsely Integrated Operand Scanning) method.
    #[inline(always)]
    pub fn monty_mul<const LIMBS: usize>(
        a: &Uint<LIMBS>,
        b: &Uint<LIMBS>,
        modulus: &Uint<LIMBS>,
    ) -> Uint<LIMBS> {
        let a_words = a.as_words();
        let b_words = b.as_words();
        let mod_words = modulus.as_words();
        let mod_neg_inv = compute_mod_neg_inv(mod_words[0]);

        let mut result = [0; LIMBS];
        let carry = monty_mul_cios::<LIMBS>(a_words, b_words, mod_words, mod_neg_inv, &mut result);

        // Conditional subtraction: subtract modulus if carry != 0 OR result >= modulus
        // First compute result - modulus
        let mut diff: [Word; _] = [0; LIMBS];
        let mut borrow: Word = 0;
        for i in 0..LIMBS {
            let (d, b1) = result[i].overflowing_sub(mod_words[i]);
            let (d, b2) = d.overflowing_sub(borrow);
            diff[i] = d;
            borrow = Word::from(b1) | Word::from(b2);
        }

        // Use diff if: carry != 0 (overflow) OR borrow == 0 (result >= modulus)
        // i.e., use result only if: carry == 0 AND borrow != 0
        let use_diff = (carry != 0) | (borrow == 0);
        let mask = Word::ZERO.wrapping_sub(Word::from(use_diff));
        for i in 0..LIMBS {
            result[i] = (diff[i] & mask) | (result[i] & !mask);
        }

        Uint::from_words(result)
    }

    /// CIOS (Coarsely Integrated Operand Scanning) Montgomery multiplication.
    ///
    /// For each limb of a, multiply by all of b, add to accumulator,
    /// then do Montgomery reduction step. Returns (result, carry) where
    /// carry indicates if an additional modulus subtraction is needed.
    ///
    /// Expects `out` to be initialized to zero.
    #[inline(always)]
    #[allow(clippy::arithmetic_side_effects)]
    pub fn monty_mul_cios<const LIMBS: usize>(
        a: &[Word; LIMBS],
        b: &[Word; LIMBS],
        modulus: &[Word; LIMBS],
        mod_neg_inv: Word,
        out: &mut [Word; LIMBS],
    ) -> Word {
        let mut acc_hi: Word = 0;

        for &a_i in a {
            // Step 1: acc += a_i * b
            let mut carry = 0;
            for j in 0..LIMBS {
                let (lo, hi) = mul_add_carry(a_i, b[j], out[j], carry);
                out[j] = lo;
                carry = hi;
            }
            let (new_acc_hi, meta_carry) = acc_hi.overflowing_add(carry);
            acc_hi = new_acc_hi;

            // Step 2: Montgomery reduction
            // u = acc[0] * mod_neg_inv mod 2^word_bits
            let u = out[0].wrapping_mul(mod_neg_inv);

            // acc += u * modulus, then shift right by one limb
            let (_, hi) = mul_add_carry(u, modulus[0], out[0], 0);
            carry = hi;

            for j in 1..LIMBS {
                let (lo, hi) = mul_add_carry(u, modulus[j], out[j], carry);
                out[j - 1] = lo;
                carry = hi;
            }

            let (sum, c) = acc_hi.overflowing_add(carry);
            out[LIMBS - 1] = sum;
            acc_hi = Word::from(meta_carry) + Word::from(c);
        }

        // Return carry - if non-zero, result >= 2^(word_bits*LIMBS) and needs reduction
        acc_hi
    }

    /// Compute a * b + c + d, returning (lo, hi) of WideWord result.
    #[inline(always)]
    #[allow(clippy::cast_possible_truncation, clippy::arithmetic_side_effects)]
    fn mul_add_carry(a: Word, b: Word, c: Word, d: Word) -> (Word, Word) {
        let wide = WideWord::from(a) * WideWord::from(b) + WideWord::from(c) + WideWord::from(d);
        (wide as Word, (wide >> Word::BITS) as Word)
    }

    /// Copy-paste from `crypto-bigint`
    #[inline(always)]
    #[allow(clippy::cast_possible_truncation)]
    pub const fn monty_mul_limbs(
        x: &[Limb],
        y: &[Limb],
        out: &mut [Limb],
        modulus: &[Limb],
        mod_neg_inv: Limb,
    ) -> Limb {
        let nlimbs = modulus.len();
        assert!(nlimbs == x.len() && nlimbs == y.len() && nlimbs == out.len());

        let mut meta_carry = 0;

        let mut i = 0;
        while i < nlimbs {
            let xi = x[i];
            // A[0] + x[i]y[0] <= (2^64 - 1)^2 + (2^64 - 1) = 2^128 - 2^64
            let axy = (xi.0 as WideWord) * (y[0].0 as WideWord) + out[0].0 as WideWord;
            let u = (axy as Word).wrapping_mul(mod_neg_inv.0);

            let mut carry;
            // A[0] + x[i]y[0] + um[0] <= (2^128 - 1) + (2^128 - 2^64) = 2^129 - 2^64 - 1
            let (a, c) = ((u as WideWord) * (modulus[0].0 as WideWord)).overflowing_add(axy);
            // carry <= (2^129 - 2^64 - 1) / 2^64 <= 2^65 - 2
            carry = ((c as WideWord) << Word::BITS) | (a >> Word::BITS);

            let mut j = 1;
            while j < nlimbs {
                // A[j] + x[i]y[j] <= (2^64 - 1)^2 + (2^64 - 1) = 2^128 - 2^64
                let axy = (xi.0 as WideWord) * (y[j].0 as WideWord) + out[j].0 as WideWord;
                // um[j] + carry <= (2^64 - 1)^2 + (2^65 - 2) = 2^128 - 1
                let umc = (u as WideWord) * (modulus[j].0 as WideWord) + carry;
                let (ab, c) = axy.overflowing_add(umc);
                out[j - 1] = Limb(ab as Word);
                // carry <= (2^129 - 2^64 - 1) / 2^64 <= 2^65 - 2
                carry = ((c as WideWord) << Word::BITS) | (ab >> Word::BITS);
                j += 1;
            }

            carry += meta_carry;
            (out[nlimbs - 1], meta_carry) = (Limb(carry as Word), carry >> Word::BITS);

            i += 1;
        }

        Limb(meta_carry as Word)
    }
}

pub mod pow {
    use crate::IntSemiring;
    use core::{array, mem};
    use crypto_bigint::{Limb, UintRef, Word};

    const WINDOW: u32 = 4;
    const WINDOW_COUNT: usize = 1 << WINDOW;
    const WINDOW_MASK: Word = (1 << WINDOW) - 1;
    const BITS_PER_WINDOW: u32 = Limb::BITS / WINDOW;

    /// Performs modular exponentiation using "Almost Montgomery
    /// Multiplication".
    ///
    /// Returns a result which has been fully reduced by the modulus specified
    /// in `params`.
    ///
    /// `exponent_bits` represents the length of the exponent in bits.
    ///
    /// NOTE: `exponent_bits` is leaked in the time pattern.
    pub fn pow_montgomery_form_amm<U>(
        x: &U,
        exponent: &[Limb],
        exponent_bits: u32,
        one: U,
        modulus: &U,
        mut mul_amm_assign: impl FnMut(&mut U, &U),
        mut square_amm_assign: impl FnMut(&mut U),
    ) -> U
    where
        U: AsRef<UintRef> + AsMut<UintRef> + Clone,
    {
        if exponent_bits == 0 {
            return one; // 1 in Montgomery form
        }

        let mut power = x.clone();

        // powers[i] contains x^i
        let powers: [U; WINDOW_COUNT] = array::from_fn(|n| {
            if n == 0 {
                one.clone()
            } else if n == WINDOW_COUNT - 1 {
                power.clone()
            } else {
                let mut new_power = power.clone();
                mul_amm_assign(&mut new_power, x);

                mem::swap(&mut power, &mut new_power);
                new_power
            }
        });

        let (starting_limb, starting_window, starting_window_mask) = pow_init(exponent_bits);

        let mut z = one; // 1 in Montgomery form
        let mut power = powers[0].clone();

        for limb_num in (0..=starting_limb).rev() {
            let w = exponent[limb_num].0;

            let mut window_num = if limb_num == starting_limb {
                starting_window + 1
            } else {
                BITS_PER_WINDOW
            };

            while window_num > 0 {
                window_num -= 1;

                let mut idx = (w >> (window_num * WINDOW)) & WINDOW_MASK;

                if limb_num == starting_limb && window_num == starting_window {
                    idx &= starting_window_mask;
                } else {
                    for _ in 1..=WINDOW {
                        square_amm_assign(&mut z);
                    }
                }

                power
                    .as_mut()
                    .as_mut_limbs()
                    .copy_from_slice(powers[0].as_ref().as_limbs());
                for i in 1..WINDOW_COUNT {
                    if (i as Word) == idx {
                        power = powers[i].clone();
                    }
                }

                mul_amm_assign(&mut z, &power);
            }
        }

        almost_montgomery_reduce(z.as_mut(), modulus.as_ref());
        z
    }

    /// Raise `x` (in Montgomery form) to a `u32` exponent, returning a fully
    /// reduced result in Montgomery form.
    pub fn pow_u32<U>(x: &U, exponent: u32, one: U, modulus: &U, mod_neg_inv: Limb) -> U
    where
        U: AsRef<UintRef> + AsMut<UintRef> + Clone,
    {
        let exponent = [Limb(Word::from(exponent)); 1];
        pow_bounded_exp(x, &exponent, u32::BITS, one, modulus, mod_neg_inv)
    }

    /// Raise `x` (in Montgomery form) to the `exponent_bits` least significant
    /// bits of `exponent`, returning a fully reduced result in Montgomery
    /// form.
    ///
    /// Wraps [`pow_montgomery_form_amm`] with the AMM closures and the scratch
    /// buffers they need (mirroring `BoxedMontyMultiplier`'s reusable product
    /// buffer; one per closure so the mutable borrows stay disjoint).
    ///
    /// NOTE: `exponent_bits` may be leaked in the time pattern.
    pub fn pow_bounded_exp<U>(
        x: &U,
        exponent: &[Limb],
        exponent_bits: u32,
        one: U,
        modulus: &U,
        mod_neg_inv: Limb,
    ) -> U
    where
        U: AsRef<UintRef> + AsMut<UintRef> + Clone,
    {
        // Only the size matters: `almost_montgomery_mul` zeroes its output
        // buffer on every use.
        let mut mul_product = modulus.clone();
        let mut square_product = modulus.clone();

        pow_montgomery_form_amm(
            x,
            exponent,
            exponent_bits,
            one,
            modulus,
            |a, b| mul_amm_assign(a, b, &mut mul_product, modulus, mod_neg_inv),
            |a| square_amm_assign(a, &mut square_product, modulus, mod_neg_inv),
        )
    }

    /// Mirrors `BoxedMontyMultiplier::mul_amm_assign` from `crypto-bigint`:
    /// perform an "Almost Montgomery Multiplication", assigning the product to
    /// `a`.
    ///
    /// `product` is a reusable scratch buffer of the same precision as
    /// `modulus`.
    ///
    /// NOTE: the result is reduced to the *bit length* of the modulus, but may
    /// still exceed the modulus; a final [`almost_montgomery_reduce`] is
    /// required for a fully reduced result.
    fn mul_amm_assign<U: AsRef<UintRef> + AsMut<UintRef>>(
        a: &mut U,
        b: &U,
        product: &mut U,
        modulus: &U,
        mod_neg_inv: Limb,
    ) {
        almost_montgomery_mul(
            a.as_ref().as_limbs(),
            b.as_ref().as_limbs(),
            product.as_mut(),
            modulus.as_ref(),
            mod_neg_inv,
        );
        a.as_mut()
            .as_mut_limbs()
            .copy_from_slice(product.as_ref().as_limbs());
    }

    /// Mirrors `BoxedMontyMultiplier::square_amm_assign` from `crypto-bigint`;
    /// see [`mul_amm_assign`].
    fn square_amm_assign<U: AsRef<UintRef> + AsMut<UintRef>>(
        a: &mut U,
        product: &mut U,
        modulus: &U,
        mod_neg_inv: Limb,
    ) {
        let a_limbs = a.as_ref().as_limbs();
        almost_montgomery_mul(
            a_limbs,
            a_limbs,
            product.as_mut(),
            modulus.as_ref(),
            mod_neg_inv,
        );
        a.as_mut()
            .as_mut_limbs()
            .copy_from_slice(product.as_ref().as_limbs());
    }

    /// Mirrors `almost_montgomery_mul` from `crypto-bigint`, delegating the
    /// inner multiplication to [`super::mul::monty_mul_limbs`].
    fn almost_montgomery_mul(
        x: &[Limb],
        y: &[Limb],
        out: &mut UintRef,
        modulus: &UintRef,
        mod_neg_inv: Limb,
    ) {
        out.as_mut_limbs().fill(Limb::ZERO);
        let overflow =
            super::mul::monty_mul_limbs(x, y, out.as_mut_limbs(), modulus.as_limbs(), mod_neg_inv);
        conditional_borrowing_sub_assign(out, modulus, overflow.0.is_odd());
    }

    fn pow_init(exponent_bits: u32) -> (usize, u32, Word) {
        let starting_limb = ((exponent_bits - 1) / Limb::BITS) as usize;
        let starting_bit_in_limb = (exponent_bits - 1) % Limb::BITS;
        let starting_window = starting_bit_in_limb / WINDOW;
        let starting_window_mask = (1 << (starting_bit_in_limb % WINDOW + 1)) - 1;
        (starting_limb, starting_window, starting_window_mask)
    }

    fn almost_montgomery_reduce(z: &mut UintRef, modulus: &UintRef) {
        let choice = UintRef::cmp_vartime(z, modulus).is_ge();
        conditional_borrowing_sub_assign(z, modulus, choice);
        conditional_borrowing_sub_assign(z, modulus, choice);
    }

    fn conditional_borrowing_sub_assign(lhs: &mut UintRef, rhs: &UintRef, choice: bool) -> bool {
        debug_assert!(lhs.bits_precision() <= rhs.bits_precision());
        let mask = if choice { Limb::MAX } else { Limb::ZERO };
        let mut borrow = Limb::ZERO;

        for i in 0..lhs.nlimbs() {
            let masked_rhs = *rhs.as_limbs().get(i).unwrap_or(&Limb::ZERO) & mask;
            let (limb, b) = lhs.as_limbs()[i].borrowing_sub(masked_rhs, borrow);
            lhs.as_mut_limbs()[i] = limb;
            borrow = b;
        }

        borrow.0.is_odd()
    }
}
