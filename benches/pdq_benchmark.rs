use criterion::{criterion_group, criterion_main, Criterion};
use pdqhash;

fn criterion_benchmark(c: &mut Criterion) {
    let bytes = include_bytes!("../src/test_data/bridge-1-original.jpg");
    let image = image::load_from_memory(bytes).unwrap();

    c.bench_function("load_bridge", |b| {
        b.iter(|| pdqhash::generate_pdq_full_size(&image))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
