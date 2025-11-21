#![allow(non_local_definitions)]
#![allow(clippy::eq_op)]

use std::{
    hint::black_box,
    iter::{Product, Sum},
};

use criterion::{
    AxisScale, BenchmarkId, Criterion, PlotConfiguration, criterion_group, criterion_main,
};
use crypto_bigint::{U256, const_monty_params};
use crypto_primitives::crypto_bigint_const_monty::F256;

const_monty_params!(
    Params,
    U256,
    "0000000000000000000000000000000000860995AE68FC80E1B1BD1E39D54B33"
);

type F = F256<Params>;

#[allow(clippy::arithmetic_side_effects)]
fn bench_random_field(
    group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
    num: u64,
) {
    let field_elem = F::from(num);
    let param = format!("Param = {}", num);

    group.bench_with_input(
        BenchmarkId::new("Multiply owned by owned", &param),
        &field_elem,
        |b, unop_elem| {
            b.iter(|| {
                for _ in 0..10000 {
                    let _ = black_box(*unop_elem * *unop_elem);
                }
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("Multiply owned by ref", &param),
        &field_elem,
        |b, unop_elem| {
            b.iter(|| {
                for _ in 0..10000 {
                    let _ = black_box(*unop_elem * unop_elem);
                }
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("Multiply ref by ref", &param),
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
            b.iter(|| {
                for _ in 0..10000 {
                    let _ = black_box(*unop_elem + *unop_elem);
                }
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("Add owned to ref", &param),
        &field_elem,
        |b, unop_elem| {
            b.iter(|| {
                for _ in 0..10000 {
                    let _ = black_box(*unop_elem + unop_elem);
                }
            });
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
            b.iter(|| {
                for _ in 0..10000 {
                    let _ = black_box(*unop_elem / *unop_elem);
                }
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("Divide owned by ref", &param),
        &field_elem,
        |b, unop_elem| {
            b.iter(|| {
                for _ in 0..10000 {
                    let _ = black_box(*unop_elem / unop_elem);
                }
            });
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
            b.iter(|| {
                for _ in 0..10000 {
                    let _ = black_box(-*unop_elem);
                }
            });
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

    let mut group = c.benchmark_group("Const Monty Field Arithmetic");
    group.plot_config(plot_config);

    bench_random_field(&mut group, 695962179703_u64);
    bench_random_field(&mut group, 2345695962179703_u64);
    bench_random_field(&mut group, 111111111111111111_u64);
    bench_random_field(&mut group, 12345678124578658568_u64);
    group.finish();
}

criterion_group!(benches, field_benchmarks);
criterion_main!(benches);
