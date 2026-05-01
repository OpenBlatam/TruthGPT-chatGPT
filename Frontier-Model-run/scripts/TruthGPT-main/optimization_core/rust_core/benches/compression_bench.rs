//! Compression Benchmarks
//!
//! Benchmarks for LZ4 and Zstd compression/decompression.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use truthgpt_rust::compression::{compress, decompress, compress_with_stats, compress_zstd_level, CompressionAlgorithm};

fn generate_test_data(size: usize, pattern: &str) -> Vec<u8> {
    match pattern {
        "random" => (0..size).map(|i| (i * 17 + 23) as u8).collect(),
        "repetitive" => (0..size).map(|i| (i % 10) as u8).collect(),
        "zeros" => vec![0u8; size],
        "text" => "Hello, World! This is a test of the compression system. ".repeat(size / 50).into_bytes(),
        _ => vec![0u8; size],
    }
}

fn bench_lz4_compress(c: &mut Criterion) {
    let mut group = c.benchmark_group("lz4_compress");
    
    for size in [1024, 16384, 65536, 262144, 1048576].iter() {
        let data = generate_test_data(*size, "repetitive");
        
        group.throughput(Throughput::Bytes(*size as u64));
        
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}KB", size / 1024)),
            &data,
            |b, data| {
                b.iter(|| {
                    black_box(compress(data, &CompressionAlgorithm::LZ4).unwrap())
                })
            },
        );
    }
    
    group.finish();
}

fn bench_lz4_decompress(c: &mut Criterion) {
    let mut group = c.benchmark_group("lz4_decompress");
    
    for size in [1024, 16384, 65536, 262144, 1048576].iter() {
        let data = generate_test_data(*size, "repetitive");
        let compressed = compress(&data, &CompressionAlgorithm::LZ4).unwrap();
        
        group.throughput(Throughput::Bytes(*size as u64));
        
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}KB", size / 1024)),
            &compressed,
            |b, compressed| {
                b.iter(|| {
                    black_box(decompress(compressed, &CompressionAlgorithm::LZ4).unwrap())
                })
            },
        );
    }
    
    group.finish();
}

fn bench_zstd_compress(c: &mut Criterion) {
    let mut group = c.benchmark_group("zstd_compress");
    
    for size in [16384, 65536, 262144, 1048576].iter() {
        let data = generate_test_data(*size, "repetitive");
        
        group.throughput(Throughput::Bytes(*size as u64));
        
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}KB", size / 1024)),
            &data,
            |b, data| {
                b.iter(|| {
                    black_box(compress(data, &CompressionAlgorithm::Zstd).unwrap())
                })
            },
        );
    }
    
    group.finish();
}

fn bench_zstd_decompress(c: &mut Criterion) {
    let mut group = c.benchmark_group("zstd_decompress");
    
    for size in [16384, 65536, 262144, 1048576].iter() {
        let data = generate_test_data(*size, "repetitive");
        let compressed = compress(&data, &CompressionAlgorithm::Zstd).unwrap();
        
        group.throughput(Throughput::Bytes(*size as u64));
        
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}KB", size / 1024)),
            &compressed,
            |b, compressed| {
                b.iter(|| {
                    black_box(decompress(compressed, &CompressionAlgorithm::Zstd).unwrap())
                })
            },
        );
    }
    
    group.finish();
}

fn bench_zstd_levels(c: &mut Criterion) {
    let mut group = c.benchmark_group("zstd_levels");
    
    let data = generate_test_data(65536, "repetitive");
    
    for level in [1, 3, 5, 9, 15, 19].iter() {
        group.throughput(Throughput::Bytes(data.len() as u64));
        
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("level_{}", level)),
            level,
            |b, &level| {
                b.iter(|| {
                    black_box(compress_zstd_level(&data, level).unwrap())
                })
            },
        );
    }
    
    group.finish();
}

fn bench_compression_ratio(c: &mut Criterion) {
    let mut group = c.benchmark_group("compression_ratio");
    
    let patterns = ["random", "repetitive", "zeros", "text"];
    let size = 65536;
    
    for pattern in patterns.iter() {
        let data = generate_test_data(size, pattern);
        
        group.bench_function(format!("lz4_{}", pattern), |b| {
            b.iter(|| {
                let (compressed, stats) = compress_with_stats(&data, &CompressionAlgorithm::LZ4).unwrap();
                black_box((compressed.len(), stats.compression_ratio()))
            })
        });
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_lz4_compress,
    bench_lz4_decompress,
    bench_zstd_compress,
    bench_zstd_decompress,
    bench_zstd_levels,
    bench_compression_ratio,
);

criterion_main!(benches);












