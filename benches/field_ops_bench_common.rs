use std::hint::black_box;

use criterion::{AxisScale, BenchmarkId, Criterion, PlotConfiguration};
use crypto_primitives::BaseFieldConfig;

fn bench_random_field<C: BaseFieldConfig>(
    group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
    num: u64,
    config: &C,
    project: &impl Fn(&C, u64) -> C::Element,
) {
    let field_elem = project(config, num);
    let param = format!("Param = {num}");

    group.bench_with_input(
        BenchmarkId::new("Mul ref by ref", &param),
        &field_elem,
        |b, elem| {
            b.iter(|| {
                for _ in 0..10000 {
                    let _ = black_box(config.mul(elem, elem));
                }
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("Mul assign", &param),
        &field_elem,
        |b, elem| {
            b.iter(|| {
                let mut acc = elem.clone();
                for _ in 0..10000 {
                    config.mul_assign(&mut acc, elem);
                }
                black_box(acc)
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("Add ref to ref", &param),
        &field_elem,
        |b, elem| {
            b.iter(|| {
                for _ in 0..10000 {
                    let _ = black_box(config.add(elem, elem));
                }
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("Add assign", &param),
        &field_elem,
        |b, elem| {
            b.iter(|| {
                let mut acc = elem.clone();
                for _ in 0..10000 {
                    config.add_assign(&mut acc, elem);
                }
                black_box(acc)
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("Div ref by ref", &param),
        &field_elem,
        |b, elem| {
            b.iter(|| {
                for _ in 0..10000 {
                    let _ = black_box(config.div(elem, elem));
                }
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("Negation", &param),
        &field_elem,
        |b, elem| {
            b.iter(|| {
                for _ in 0..10000 {
                    let _ = black_box(config.neg(elem));
                }
            });
        },
    );

    let v = vec![field_elem; 10];

    group.bench_with_input(BenchmarkId::new("Sum", &param), &v, |b, v| {
        b.iter(|| {
            for _ in 0..10000 {
                let _ = black_box(config.sum_refs(v.iter()));
            }
        });
    });

    group.bench_with_input(BenchmarkId::new("Product", &param), &v, |b, v| {
        b.iter(|| {
            for _ in 0..10000 {
                let _ = black_box(config.product_refs(v.iter()));
            }
        });
    });
}

pub fn field_benchmarks<C: BaseFieldConfig>(
    c: &mut Criterion,
    name: &str,
    config: &C,
    project: impl Fn(&C, u64) -> C::Element,
) {
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);

    let mut group = c.benchmark_group(name);
    group.plot_config(plot_config);

    bench_random_field(&mut group, 695962179703_u64, config, &project);
    bench_random_field(&mut group, 2345695962179703_u64, config, &project);
    bench_random_field(&mut group, 111111111111111111_u64, config, &project);
    bench_random_field(&mut group, 12345678124578658568_u64, config, &project);
    group.finish();
}
