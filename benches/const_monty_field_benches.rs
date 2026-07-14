mod field_ops_bench_common;

use criterion::{Criterion, criterion_group, criterion_main};
use crypto_bigint::{U256, const_monty_params};
use crypto_primitives::{FixedFieldConfig, crypto_bigint_const_monty::ConstMontyField};

use crate::field_ops_bench_common::field_benchmarks;

const_monty_params!(
    Params,
    U256,
    "0000000000000000000000000000000000860995AE68FC80E1B1BD1E39D54B33"
);

type F = ConstMontyField<Params, { U256::LIMBS }>;

pub fn const_monty_field_benches(c: &mut Criterion) {
    field_benchmarks(
        c,
        "Const Monty Field Arithmetic",
        &FixedFieldConfig::<F>::default(),
        |_, num| F::from(num),
    );
}

criterion_group!(benches, const_monty_field_benches);
criterion_main!(benches);
