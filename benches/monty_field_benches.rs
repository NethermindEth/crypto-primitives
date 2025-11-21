#![allow(non_local_definitions)]
#![allow(clippy::eq_op)]

use std::{
    hint::black_box,
    iter::{Product, Sum},
};

use criterion::{
    AxisScale, BatchSize, BenchmarkId, Criterion, PlotConfiguration, criterion_group,
    criterion_main,
};
use crypto_bigint::{Odd, modular::MontyParams};
use crypto_primitives::{FromWithConfig, crypto_bigint_monty::F256};

const LIMBS: usize = 4;

type F = F256;

fn bench_config() -> MontyParams<LIMBS> {
    let modulus = crypto_bigint::Uint::<LIMBS>::from_be_hex(
        "0000000000000000000000000000000000860995AE68FC80E1B1BD1E39D54B33",
    );
    let modulus = Odd::new(modulus).expect("modulus should be odd");
    MontyParams::new(modulus)
}

#[allow(clippy::arithmetic_side_effects)]
fn bench_random_field(
    group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
    num: u64,
) {
    let field_elem = F::from_with_cfg(num, &bench_config());
    let param = format!("Param = {}", num);

    group.bench_with_input(
        BenchmarkId::new("Mul owned by owned", &param),
        &field_elem,
        |b, unop_elem| {
            b.iter_batched(
                || {
                    (
                        vec![unop_elem.clone(); 10000],
                        vec![unop_elem.clone(); 10000],
                    )
                },
                |(lhs, rhs)| {
                    for (lhs, rhs) in lhs.into_iter().zip(rhs.into_iter()) {
                        let _ = black_box(lhs * rhs);
                    }
                },
                BatchSize::SmallInput,
            );
        },
    );

    group.bench_with_input(
        BenchmarkId::new("Mul owned by ref", &param),
        &field_elem,
        |b, unop_elem| {
            b.iter_batched(
                || {
                    (
                        vec![unop_elem.clone(); 10000],
                        vec![unop_elem.clone(); 10000],
                    )
                },
                |(lhs, rhs)| {
                    for (lhs, rhs) in lhs.into_iter().zip(rhs.into_iter()) {
                        let _ = black_box(lhs * &rhs);
                    }
                },
                BatchSize::SmallInput,
            );
        },
    );

    group.bench_with_input(
        BenchmarkId::new("Mul ref by ref", &param),
        &field_elem,
        |b, unop_elem| {
            b.iter(|| {
                for _ in 0..10000 {
                    let _ = black_box(unop_elem * unop_elem);
                }
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("Add owned to owned", &param),
        &field_elem,
        |b, unop_elem| {
            b.iter_batched(
                || {
                    (
                        vec![unop_elem.clone(); 10000],
                        vec![unop_elem.clone(); 10000],
                    )
                },
                |(lhs, rhs)| {
                    for (lhs, rhs) in lhs.into_iter().zip(rhs.into_iter()) {
                        let _ = black_box(lhs + rhs);
                    }
                },
                BatchSize::SmallInput,
            );
        },
    );

    group.bench_with_input(
        BenchmarkId::new("Add owned to ref", &param),
        &field_elem,
        |b, unop_elem| {
            b.iter_batched(
                || {
                    (
                        vec![unop_elem.clone(); 10000],
                        vec![unop_elem.clone(); 10000],
                    )
                },
                |(lhs, rhs)| {
                    for (lhs, rhs) in lhs.into_iter().zip(rhs.into_iter()) {
                        let _ = black_box(lhs + &rhs);
                    }
                },
                BatchSize::SmallInput,
            );
        },
    );

    group.bench_with_input(
        BenchmarkId::new("Add ref to ref", &param),
        &field_elem,
        |b, unop_elem| {
            b.iter(|| {
                for _ in 0..10000 {
                    let _ = black_box(unop_elem + unop_elem);
                }
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("Div owned by owned", &param),
        &field_elem,
        |b, unop_elem| {
            b.iter_batched(
                || {
                    (
                        vec![unop_elem.clone(); 10000],
                        vec![unop_elem.clone(); 10000],
                    )
                },
                |(lhs, rhs)| {
                    for (lhs, rhs) in lhs.into_iter().zip(rhs.into_iter()) {
                        let _ = black_box(lhs / rhs);
                    }
                },
                BatchSize::SmallInput,
            );
        },
    );

    group.bench_with_input(
        BenchmarkId::new("Div owned by ref", &param),
        &field_elem,
        |b, unop_elem| {
            b.iter_batched(
                || {
                    (
                        vec![unop_elem.clone(); 10000],
                        vec![unop_elem.clone(); 10000],
                    )
                },
                |(lhs, rhs)| {
                    for (lhs, rhs) in lhs.into_iter().zip(rhs.into_iter()) {
                        let _ = black_box(lhs / &rhs);
                    }
                },
                BatchSize::SmallInput,
            );
        },
    );

    group.bench_with_input(
        BenchmarkId::new("Div ref by ref", &param),
        &field_elem,
        |b, unop_elem| {
            b.iter(|| {
                for _ in 0..10000 {
                    let _ = black_box(unop_elem / unop_elem);
                }
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("Negation", &param),
        &field_elem,
        |b, unop_elem| {
            b.iter_batched(
                || vec![unop_elem.clone(); 10000],
                |unop_elem| {
                    for x in unop_elem.into_iter() {
                        let _ = black_box(-x);
                    }
                },
                BatchSize::SmallInput,
            );
        },
    );

    let v = vec![field_elem; 10];

    group.bench_with_input(BenchmarkId::new("Sum", &param), &v, |b, v| {
        b.iter(|| {
            for _ in 0..10000 {
                let _ = black_box(F::sum(v.iter()));
            }
        });
    });

    group.bench_with_input(BenchmarkId::new("Product", &param), &v, |b, v| {
        b.iter(|| {
            for _ in 0..10000 {
                let _ = black_box(F::product(v.iter()));
            }
        });
    });
}

pub fn field_benchmarks(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);

    let mut group = c.benchmark_group("Monty Field Arithmetic");
    group.plot_config(plot_config);

    bench_random_field(&mut group, 695962179703_u64);
    bench_random_field(&mut group, 2345695962179703_u64);
    bench_random_field(&mut group, 111111111111111111_u64);
    bench_random_field(&mut group, 12345678124578658568_u64);
    group.finish();
}

criterion_group!(benches, field_benchmarks);
criterion_main!(benches);
