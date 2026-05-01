//! Quantization Benchmarks
//!
//! Benchmarks for quantization/dequantization operations.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use truthgpt_rust::quantization::{
    quantize_int8, dequantize_int8, quantize_int4, dequantize_int4,
    quantize_fp16, dequantize_fp16, quantize_bf16, dequantize_bf16,
    quantize_grouped_int8, dequantize_grouped_int8,
    QuantizationParams, QuantizationType,
};

fn generate_test_data(size: usize) -> Vec<f32> {
    (0..size).map(|i| (i as f32 * 0.1) - (size as f32 * 0.05)).collect()
}

fn bench_int8_quantize(c: &mut Criterion) {
    let mut group = c.benchmark_group("int8_quantize");
    
    for size in [1024, 16384, 65536, 262144, 1048576].iter() {
        let data = generate_test_data(*size);
        let params = QuantizationParams::from_tensor(&data, QuantizationType::INT8);
        
        group.throughput(Throughput::Bytes((*size * 4) as u64));
        
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}K", size / 1024)),
            &data,
            |b, data| {
                b.iter(|| {
                    black_box(quantize_int8(data, &params))
                })
            },
        );
    }
    
    group.finish();
}

fn bench_int8_dequantize(c: &mut Criterion) {
    let mut group = c.benchmark_group("int8_dequantize");
    
    for size in [1024, 16384, 65536, 262144, 1048576].iter() {
        let data = generate_test_data(*size);
        let params = QuantizationParams::from_tensor(&data, QuantizationType::INT8);
        let quantized = quantize_int8(&data, &params);
        
        group.throughput(Throughput::Bytes(*size as u64));
        
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}K", size / 1024)),
            &quantized,
            |b, quantized| {
                b.iter(|| {
                    black_box(dequantize_int8(quantized, &params))
                })
            },
        );
    }
    
    group.finish();
}

fn bench_int4_quantize(c: &mut Criterion) {
    let mut group = c.benchmark_group("int4_quantize");
    
    for size in [1024, 16384, 65536, 262144].iter() {
        let data = generate_test_data(*size);
        let params = QuantizationParams::from_tensor(&data, QuantizationType::INT4);
        
        group.throughput(Throughput::Bytes((*size * 4) as u64));
        
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}K", size / 1024)),
            &data,
            |b, data| {
                b.iter(|| {
                    black_box(quantize_int4(data, &params))
                })
            },
        );
    }
    
    group.finish();
}

fn bench_fp16_quantize(c: &mut Criterion) {
    let mut group = c.benchmark_group("fp16_quantize");
    
    for size in [1024, 16384, 65536, 262144, 1048576].iter() {
        let data = generate_test_data(*size);
        
        group.throughput(Throughput::Bytes((*size * 4) as u64));
        
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}K", size / 1024)),
            &data,
            |b, data| {
                b.iter(|| {
                    black_box(quantize_fp16(data))
                })
            },
        );
    }
    
    group.finish();
}

fn bench_fp16_dequantize(c: &mut Criterion) {
    let mut group = c.benchmark_group("fp16_dequantize");
    
    for size in [1024, 16384, 65536, 262144, 1048576].iter() {
        let data = generate_test_data(*size);
        let quantized = quantize_fp16(&data);
        
        group.throughput(Throughput::Bytes((*size * 2) as u64));
        
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}K", size / 1024)),
            &quantized,
            |b, quantized| {
                b.iter(|| {
                    black_box(dequantize_fp16(quantized))
                })
            },
        );
    }
    
    group.finish();
}

fn bench_bf16_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("bf16_roundtrip");
    
    for size in [16384, 65536, 262144].iter() {
        let data = generate_test_data(*size);
        
        group.throughput(Throughput::Bytes((*size * 4) as u64));
        
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}K", size / 1024)),
            &data,
            |b, data| {
                b.iter(|| {
                    let quantized = quantize_bf16(data);
                    black_box(dequantize_bf16(&quantized))
                })
            },
        );
    }
    
    group.finish();
}

fn bench_grouped_quantization(c: &mut Criterion) {
    let mut group = c.benchmark_group("grouped_int8");
    
    let data = generate_test_data(65536);
    
    for group_size in [32, 64, 128, 256].iter() {
        group.throughput(Throughput::Bytes((data.len() * 4) as u64));
        
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("group_{}", group_size)),
            group_size,
            |b, &group_size| {
                b.iter(|| {
                    black_box(quantize_grouped_int8(&data, group_size))
                })
            },
        );
    }
    
    group.finish();
}

fn bench_quantization_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantization_comparison");
    
    let size = 65536;
    let data = generate_test_data(size);
    let params_int8 = QuantizationParams::from_tensor(&data, QuantizationType::INT8);
    let params_int4 = QuantizationParams::from_tensor(&data, QuantizationType::INT4);
    
    group.throughput(Throughput::Bytes((size * 4) as u64));
    
    group.bench_function("int8", |b| {
        b.iter(|| black_box(quantize_int8(&data, &params_int8)))
    });
    
    group.bench_function("int4", |b| {
        b.iter(|| black_box(quantize_int4(&data, &params_int4)))
    });
    
    group.bench_function("fp16", |b| {
        b.iter(|| black_box(quantize_fp16(&data)))
    });
    
    group.bench_function("bf16", |b| {
        b.iter(|| black_box(quantize_bf16(&data)))
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_int8_quantize,
    bench_int8_dequantize,
    bench_int4_quantize,
    bench_fp16_quantize,
    bench_fp16_dequantize,
    bench_bf16_roundtrip,
    bench_grouped_quantization,
    bench_quantization_comparison,
);

criterion_main!(benches);












