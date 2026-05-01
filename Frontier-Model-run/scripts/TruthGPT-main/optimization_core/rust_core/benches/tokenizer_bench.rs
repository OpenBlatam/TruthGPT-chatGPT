//! Tokenizer Benchmarks
//!
//! Benchmarks for tokenization operations.
//!
//! Note: These benchmarks require a tokenizer.json file to run.
//! You can generate one using:
//! ```bash
//! python -c "from transformers import AutoTokenizer; t = AutoTokenizer.from_pretrained('gpt2'); t.save_pretrained('.')"
//! ```

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};

fn generate_test_texts(count: usize, avg_length: usize) -> Vec<String> {
    let base_text = "The quick brown fox jumps over the lazy dog. ";
    let repeat_count = avg_length / base_text.len() + 1;
    
    (0..count)
        .map(|i| {
            let text = base_text.repeat(repeat_count);
            format!("{} {}", i, &text[..avg_length.min(text.len())])
        })
        .collect()
}

fn bench_tokenization_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("tokenization_throughput");
    group.sample_size(10);
    
    let batch_sizes = [1, 10, 100];
    let text_lengths = [50, 200, 500];
    
    for &batch_size in &batch_sizes {
        for &text_len in &text_lengths {
            let texts = generate_test_texts(batch_size, text_len);
            let total_chars: usize = texts.iter().map(|t| t.len()).sum();
            
            group.throughput(Throughput::Bytes(total_chars as u64));
            
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("batch{}_len{}", batch_size, text_len)),
                &texts,
                |b, _texts| {
                    b.iter(|| {
                        // Placeholder - would use actual tokenizer here
                        // tokenizer.encode_batch(&texts, true)
                    })
                },
            );
        }
    }
    
    group.finish();
}

fn bench_string_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("string_ops");
    
    let texts = generate_test_texts(1000, 200);
    
    group.bench_function("clone_batch", |b| {
        b.iter(|| {
            let _cloned: Vec<String> = texts.iter().cloned().collect();
        })
    });
    
    group.bench_function("len_sum", |b| {
        b.iter(|| {
            let _total: usize = texts.iter().map(|t| t.len()).sum();
        })
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_tokenization_throughput,
    bench_string_operations,
);

criterion_main!(benches);












