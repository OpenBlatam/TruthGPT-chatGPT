"""
Examples for data processing.

Demonstrates usage of Polars and other data processors.
"""
from pathlib import Path

# Example 1: Using Polars Processor
def example_polars_processor():
    """Example of using Polars processor."""
    from data.polars_processor import PolarsProcessor
    
    # Create processor
    processor = PolarsProcessor(lazy=True)
    
    # Read data
    df = processor.read_parquet("data.parquet")
    
    # Process data
    result = processor.process_training_data(
        input_path="input.parquet",
        output_path="output.parquet",
        min_tokens=1000
    )
    
    return result


# Example 2: Using Processor Factory
def example_processor_factory():
    """Example of using processor factory."""
    from data.processor_factory import create_data_processor, ProcessorType
    
    # Auto-select processor
    processor = create_data_processor(processor_type=ProcessorType.AUTO)
    
    # Or specify processor
    processor = create_data_processor(processor_type=ProcessorType.POLARS)
    
    df = processor.read_parquet("data.parquet")
    return df


# Example 3: Processing Large Datasets
def example_large_dataset():
    """Example of processing large datasets."""
    from data.polars_processor import PolarsProcessor
    
    processor = PolarsProcessor(lazy=True)
    
    # Lazy evaluation for large datasets
    df = processor.read_parquet("large_data.parquet")
    
    # Chain operations (not executed yet)
    result = (
        df
        .filter(pl.col("tokens") > 1000)
        .group_by("category")
        .agg([
            pl.mean("loss").alias("avg_loss"),
            pl.count().alias("count")
        ])
        .sort("avg_loss")
    )
    
    # Execute when needed
    result = result.collect()
    return result













