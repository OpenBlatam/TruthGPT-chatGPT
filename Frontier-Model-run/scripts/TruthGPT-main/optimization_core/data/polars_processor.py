"""
Polars Data Processor - 10-100x faster than pandas.

This module provides high-performance DataFrame operations using Polars,
which has a Rust core and supports lazy evaluation with query optimization.

Features:
- 10-100x faster than pandas
- Lazy evaluation with query optimization
- Multi-threaded by default
- 2-3x less memory than pandas
- Native Parquet/CSV/JSONL support
- Streaming operations for large datasets
- Type-safe operations with schema validation
"""
import logging
from typing import List, Optional, Union, Dict, Any, Tuple, Callable
from pathlib import Path
import os
from enum import Enum

from .utils.validators import (
    validate_file_path,
    validate_dataframe_schema,
    validate_positive_number,
    validate_non_empty_string,
)
from .utils.file_utils import (
    detect_file_format,
    ensure_output_directory,
)

logger = logging.getLogger(__name__)

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    pl = None
    logger.warning(
        "Polars not available. Install with: pip install polars"
    )


# ════════════════════════════════════════════════════════════════════════════════
# ENUMS & CONSTANTS
# ════════════════════════════════════════════════════════════════════════════════

class FileFormat(Enum):
    """Supported file formats."""
    PARQUET = "parquet"
    CSV = "csv"
    JSONL = "jsonl"
    JSON = "json"
    ARROW = "arrow"
    EXCEL = "xlsx"


class JoinType(Enum):
    """Join types for DataFrame operations."""
    INNER = "inner"
    LEFT = "left"
    OUTER = "outer"
    ANTI = "anti"
    SEMI = "semi"


# ════════════════════════════════════════════════════════════════════════════════
# POLARS PROCESSOR
# ════════════════════════════════════════════════════════════════════════════════

class PolarsProcessor:
    """
    High-performance DataFrame processor using Polars.
    
    Features:
    - 10-100x faster than pandas
    - Lazy evaluation with query optimization
    - Multi-threaded by default
    - 2-3x less memory than pandas
    - Native Parquet/CSV/JSONL support
    - Streaming operations for large datasets
    """
    
    def __init__(
        self,
        lazy: bool = True,
        streaming: bool = False,
        num_threads: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize Polars processor.
        
        Args:
            lazy: Use lazy evaluation (recommended for performance)
            streaming: Enable streaming for very large datasets
            num_threads: Number of threads (None = auto)
            **kwargs: Additional Polars configuration
        """
        if not POLARS_AVAILABLE:
            raise ImportError(
                "Polars is not installed. Install with: pip install polars"
            )
        
        self.lazy = lazy
        self.streaming = streaming
        
        # Configure Polars threading
        if num_threads is not None:
            os.environ["POLARS_MAX_THREADS"] = str(num_threads)
        
        logger.info(
            f"Polars processor initialized (lazy={lazy}, streaming={streaming}, "
            f"threads={num_threads or 'auto'})"
        )
    
    # ════════════════════════════════════════════════════════════════════════════
    # FILE I/O OPERATIONS
    # ════════════════════════════════════════════════════════════════════════════
    
    def read_parquet(
        self,
        path: Union[str, List[str], Path],
        columns: Optional[List[str]] = None,
        row_index_name: Optional[str] = None,
        row_index_offset: int = 0,
        parallel: str = "auto",
        **kwargs
    ) -> Union[pl.DataFrame, pl.LazyFrame]:
        """
        Read Parquet file(s) with Polars.
        
        Args:
            path: Path to Parquet file or list of paths
            columns: Columns to read (None = all)
            row_index_name: Name for row index column
            row_index_offset: Offset for row index
            parallel: Parallel strategy (auto, columns, row_groups, none)
            **kwargs: Additional read arguments
        
        Returns:
            DataFrame or LazyFrame
        
        Raises:
            FileNotFoundError: If file(s) don't exist
            ValueError: If path is invalid
        """
        if not POLARS_AVAILABLE:
            raise ImportError("Polars is not installed")
        
        if isinstance(path, (str, Path)):
            path = [path]
        
        if not path:
            raise ValueError("path cannot be empty")
        
        # Validate all paths exist
        validated_paths = []
        for p in path:
            validated_paths.append(
                validate_file_path(p, must_exist=True, allowed_extensions=['.parquet'])
            )
        
        try:
            read_fn = pl.scan_parquet if (self.lazy or self.streaming) else pl.read_parquet
            
            if len(validated_paths) == 1:
                df = read_fn(
                    str(validated_paths[0]),
                    columns=columns,
                    row_index_name=row_index_name,
                    row_index_offset=row_index_offset,
                    parallel=parallel,
                    **kwargs
                )
            else:
                # Read multiple files
                dfs = [
                    read_fn(
                        str(p),
                        columns=columns,
                        row_index_name=row_index_name,
                        row_index_offset=row_index_offset,
                        parallel=parallel,
                        **kwargs
                    )
                    for p in validated_paths
                ]
                df = pl.concat(dfs) if not self.lazy else pl.concat(dfs)
            
            return df
        except Exception as e:
            logger.error(f"Failed to read Parquet file(s): {e}", exc_info=True)
            raise
    
    def read_csv(
        self,
        path: Union[str, List[str], Path],
        separator: str = ",",
        has_header: bool = True,
        infer_schema_length: int = 100,
        **kwargs
    ) -> Union[pl.DataFrame, pl.LazyFrame]:
        """
        Read CSV file(s) with Polars.
        
        Args:
            path: Path to CSV file or list of paths
            separator: Column separator
            has_header: Whether file has header row
            infer_schema_length: Number of rows to infer schema
            **kwargs: Additional read arguments
        
        Returns:
            DataFrame or LazyFrame
        """
        if not POLARS_AVAILABLE:
            raise ImportError("Polars is not installed")
        
        if isinstance(path, (str, Path)):
            path = [path]
        
        read_fn = pl.scan_csv if (self.lazy or self.streaming) else pl.read_csv
        
        if len(path) == 1:
            df = read_fn(
                str(path[0]),
                separator=separator,
                has_header=has_header,
                infer_schema_length=infer_schema_length,
                **kwargs
            )
        else:
            dfs = [
                read_fn(
                    str(p),
                    separator=separator,
                    has_header=has_header,
                    infer_schema_length=infer_schema_length,
                    **kwargs
                )
                for p in path
            ]
            df = pl.concat(dfs) if not self.lazy else pl.concat(dfs)
        
        return df
    
    def read_jsonl(
        self,
        path: Union[str, List[str], Path],
        **kwargs
    ) -> Union[pl.DataFrame, pl.LazyFrame]:
        """
        Read JSONL file(s) with Polars.
        
        Args:
            path: Path to JSONL file or list of paths
            **kwargs: Additional read arguments
        
        Returns:
            DataFrame or LazyFrame
        """
        if not POLARS_AVAILABLE:
            raise ImportError("Polars is not installed")
        
        if isinstance(path, (str, Path)):
            path = [path]
        
        read_fn = pl.scan_ndjson if (self.lazy or self.streaming) else pl.read_ndjson
        
        if len(path) == 1:
            df = read_fn(str(path[0]), **kwargs)
        else:
            dfs = [read_fn(str(p), **kwargs) for p in path]
            df = pl.concat(dfs) if not self.lazy else pl.concat(dfs)
        
        return df
    
    def read_json(
        self,
        path: Union[str, Path],
        **kwargs
    ) -> Union[pl.DataFrame, pl.LazyFrame]:
        """
        Read JSON file with Polars.
        
        Args:
            path: Path to JSON file
            **kwargs: Additional read arguments
        
        Returns:
            DataFrame or LazyFrame
        """
        if not POLARS_AVAILABLE:
            raise ImportError("Polars is not installed")
        
        read_fn = pl.read_json  # JSON doesn't support lazy scan yet
        
        df = read_fn(str(path), **kwargs)
        
        if self.lazy:
            return df.lazy()
        return df
    
    # ════════════════════════════════════════════════════════════════════════════
    # DATA TRANSFORMATION OPERATIONS
    # ════════════════════════════════════════════════════════════════════════════
    
    def filter_tokens(
        self,
        df: Union[pl.DataFrame, pl.LazyFrame],
        min_tokens: int = 1000,
        max_tokens: Optional[int] = None,
        token_col: str = "tokens"
    ) -> Union[pl.DataFrame, pl.LazyFrame]:
        """
        Filter rows by token count.
        
        Args:
            df: DataFrame or LazyFrame
            min_tokens: Minimum token count
            max_tokens: Maximum token count (None = no limit)
            token_col: Token column name
        
        Returns:
            Filtered DataFrame or LazyFrame
        """
        validate_positive_number(min_tokens, "min_tokens", min_value=0)
        if max_tokens is not None:
            validate_positive_number(max_tokens, "max_tokens", min_value=min_tokens)
        
        condition = pl.col(token_col) >= min_tokens
        if max_tokens is not None:
            condition = condition & (pl.col(token_col) <= max_tokens)
        
        return df.filter(condition)
    
    def group_by_category(
        self,
        df: Union[pl.DataFrame, pl.LazyFrame],
        category_col: str = "category",
        agg_cols: Optional[List[str]] = None,
        aggregations: Optional[Dict[str, List[str]]] = None
    ) -> Union[pl.DataFrame, pl.LazyFrame]:
        """
        Group by category with aggregations.
        
        Args:
            df: DataFrame or LazyFrame
            category_col: Category column name
            agg_cols: Columns to aggregate (default: all numeric)
            aggregations: Custom aggregations {col: [agg1, agg2, ...]}
        
        Returns:
            Grouped DataFrame or LazyFrame
        """
        validate_non_empty_string(category_col, "category_col")
        
        if aggregations is None:
            # Auto-detect numeric columns
            if isinstance(df, pl.LazyFrame):
                schema = df.schema
            else:
                schema = df.schema
            
            agg_cols = agg_cols or [
                col for col, dtype in schema.items()
                if col != category_col and dtype in (pl.Int64, pl.Float64, pl.Float32, pl.Int32, pl.UInt64)
            ]
            
            aggregations = {
                col: ["mean", "count", "min", "max", "std"]
                for col in agg_cols
            }
        
        # Build aggregation expressions
        agg_exprs = []
        for col, aggs in aggregations.items():
            for agg in aggs:
                if agg == "mean":
                    agg_exprs.append(pl.mean(col).alias(f"avg_{col}"))
                elif agg == "count":
                    agg_exprs.append(pl.count().alias(f"count_{col}"))
                elif agg == "min":
                    agg_exprs.append(pl.min(col).alias(f"min_{col}"))
                elif agg == "max":
                    agg_exprs.append(pl.max(col).alias(f"max_{col}"))
                elif agg == "std":
                    agg_exprs.append(pl.std(col).alias(f"std_{col}"))
                elif agg == "sum":
                    agg_exprs.append(pl.sum(col).alias(f"sum_{col}"))
        
        # Add overall count
        agg_exprs.append(pl.count().alias("total_count"))
        
        result = df.group_by(category_col).agg(agg_exprs)
        
        return result.sort("total_count", descending=True)
    
    def join_datasets(
        self,
        df1: Union[pl.DataFrame, pl.LazyFrame],
        df2: Union[pl.DataFrame, pl.LazyFrame],
        on: Union[str, List[str]],
        how: Union[str, JoinType] = JoinType.INNER,
        suffix: str = "_right"
    ) -> Union[pl.DataFrame, pl.LazyFrame]:
        """
        Join two DataFrames efficiently.
        
        Args:
            df1: First DataFrame
            df2: Second DataFrame
            on: Join key(s)
            how: Join type (inner, left, outer, anti, semi)
            suffix: Suffix for duplicate columns
        
        Returns:
            Joined DataFrame
        """
        if isinstance(how, str):
            how = JoinType(how.lower())
        
        how_str = how.value
        
        return df1.join(df2, on=on, how=how_str, suffix=suffix)
    
    def apply_transformation(
        self,
        df: Union[pl.DataFrame, pl.LazyFrame],
        transformation: Callable[[pl.DataFrame], pl.DataFrame],
        lazy: Optional[bool] = None
    ) -> Union[pl.DataFrame, pl.LazyFrame]:
        """
        Apply a custom transformation function.
        
        Args:
            df: DataFrame or LazyFrame
            transformation: Function that takes DataFrame and returns DataFrame
            lazy: Whether to keep lazy (None = use instance default)
        
        Returns:
            Transformed DataFrame or LazyFrame
        """
        if isinstance(df, pl.LazyFrame):
            df = df.collect()
        
        result = transformation(df)
        
        if (lazy if lazy is not None else self.lazy):
            return result.lazy()
        return result
    
    # ════════════════════════════════════════════════════════════════════════════
    # TRAINING DATA PROCESSING
    # ════════════════════════════════════════════════════════════════════════════
    
    def process_training_data(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        min_tokens: int = 1000,
        max_tokens: Optional[int] = None,
        category_col: str = "category",
        required_cols: Optional[List[str]] = None,
        transformations: Optional[List[Callable]] = None,
        **kwargs
    ) -> pl.DataFrame:
        """
        Process training data with optimized pipeline.
        
        Args:
            input_path: Input file path (Parquet/CSV/JSONL)
            output_path: Optional output path
            min_tokens: Minimum token count filter
            max_tokens: Maximum token count filter
            category_col: Category column name
            required_cols: Required columns (default: ["tokens", "loss", category_col])
            transformations: List of transformation functions to apply
            **kwargs: Additional processing arguments
        
        Returns:
            Processed DataFrame
        
        Raises:
            FileNotFoundError: If input_path doesn't exist
            ValueError: If parameters are invalid or file format unsupported
        """
        if not POLARS_AVAILABLE:
            raise ImportError("Polars is not installed")
        
        # Validate input path
        input_path_obj = validate_file_path(
            input_path,
            must_exist=True,
            allowed_extensions=['.parquet', '.csv', '.jsonl', '.json']
        )
        
        # Validate parameters
        validate_positive_number(min_tokens, "min_tokens", min_value=0)
        if max_tokens is not None:
            validate_positive_number(max_tokens, "max_tokens", min_value=min_tokens)
        validate_non_empty_string(category_col, "category_col")
        
        try:
            # Read data based on file format
            file_format = detect_file_format(input_path_obj)
            
            if file_format == 'parquet':
                df = self.read_parquet(str(input_path_obj), **kwargs)
            elif file_format == 'csv':
                df = self.read_csv(str(input_path_obj), **kwargs)
            elif file_format in ('jsonl', 'json'):
                if file_format == 'jsonl':
                    df = self.read_jsonl(str(input_path_obj), **kwargs)
                else:
                    df = self.read_json(str(input_path_obj), **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
            
            # Validate required columns
            required_cols = required_cols or ["tokens", "loss", category_col]
            if isinstance(df, pl.LazyFrame):
                schema = df.schema
            else:
                schema = df.schema
            
            validate_dataframe_schema(schema, required_cols, "Input DataFrame")
            
            # Build lazy evaluation pipeline
            if isinstance(df, pl.LazyFrame):
                pipeline = df
            else:
                pipeline = df.lazy()
            
            # Apply token filtering
            pipeline = pipeline.filter(pl.col("tokens") >= min_tokens)
            if max_tokens is not None:
                pipeline = pipeline.filter(pl.col("tokens") <= max_tokens)
            
            # Apply custom transformations
            if transformations:
                for transform in transformations:
                    pipeline = pipeline.map_batches(transform, schema=None)
            
            # Group by category with aggregations
            pipeline = (
                pipeline
                .group_by(category_col)
                .agg([
                    pl.mean("loss").alias("avg_loss"),
                    pl.std("loss").alias("std_loss"),
                    pl.min("loss").alias("min_loss"),
                    pl.max("loss").alias("max_loss"),
                    pl.count().alias("count")
                ])
                .sort("avg_loss")
            )
            
            # Collect (execute with optimization)
            result = pipeline.collect()
            
            # Save if output path provided
            if output_path:
                output_path_obj = ensure_output_directory(output_path)
                result.write_parquet(str(output_path_obj))
                logger.info(f"Processed data saved to {output_path}")
            
            logger.info(
                f"Processed {len(result)} categories from {input_path} "
                f"({len(result)} rows)"
            )
            return result
            
        except Exception as e:
            logger.error(f"Failed to process training data: {e}", exc_info=True)
            raise
    
    # ════════════════════════════════════════════════════════════════════════════
    # STREAMING OPERATIONS
    # ════════════════════════════════════════════════════════════════════════════
    
    def stream_parquet(
        self,
        path: Union[str, Path],
        batch_size: int = 10000,
        **kwargs
    ):
        """
        Stream Parquet file in batches (for very large files).
        
        Args:
            path: Path to Parquet file
            batch_size: Batch size for streaming
            **kwargs: Additional read arguments
        
        Yields:
            DataFrame batches
        """
        if not POLARS_AVAILABLE:
            raise ImportError("Polars is not installed")
        
        validate_file_path(path, must_exist=True, allowed_extensions=['.parquet'])
        validate_positive_number(batch_size, "batch_size", min_value=1)
        
        lazy_df = pl.scan_parquet(str(path), **kwargs)
        
        # Stream in batches
        for batch in lazy_df.collect(streaming=True).iter_slices(batch_size):
            yield batch
    
    # ════════════════════════════════════════════════════════════════════════════
    # WRITE OPERATIONS
    # ════════════════════════════════════════════════════════════════════════════
    
    def write_parquet(
        self,
        df: Union[pl.DataFrame, pl.LazyFrame],
        path: Union[str, Path],
        compression: str = "zstd",
        compression_level: int = 3,
        row_group_size: int = 100000,
        **kwargs
    ):
        """
        Write DataFrame to Parquet with optimization.
        
        Args:
            df: DataFrame or LazyFrame
            path: Output path
            compression: Compression algorithm (zstd, snappy, lz4, gzip)
            compression_level: Compression level (1-22 for zstd)
            row_group_size: Row group size
            **kwargs: Additional write arguments
        """
        if isinstance(df, pl.LazyFrame):
            df = df.collect()
        
        df.write_parquet(
            str(path),
            compression=compression,
            compression_level=compression_level,
            row_group_size=row_group_size,
            **kwargs
        )
        logger.info(f"DataFrame written to {path} ({len(df)} rows)")
    
    def write_csv(
        self,
        df: Union[pl.DataFrame, pl.LazyFrame],
        path: Union[str, Path],
        separator: str = ",",
        **kwargs
    ):
        """
        Write DataFrame to CSV.
        
        Args:
            df: DataFrame or LazyFrame
            path: Output path
            separator: Column separator
            **kwargs: Additional write arguments
        """
        if isinstance(df, pl.LazyFrame):
            df = df.collect()
        
        df.write_csv(str(path), separator=separator, **kwargs)
        logger.info(f"DataFrame written to {path}")
    
    def write_jsonl(
        self,
        df: Union[pl.DataFrame, pl.LazyFrame],
        path: Union[str, Path],
        **kwargs
    ):
        """
        Write DataFrame to JSONL.
        
        Args:
            df: DataFrame or LazyFrame
            path: Output path
            **kwargs: Additional write arguments
        """
        if isinstance(df, pl.LazyFrame):
            df = df.collect()
        
        df.write_ndjson(str(path), **kwargs)
        logger.info(f"DataFrame written to {path}")
    
    # ════════════════════════════════════════════════════════════════════════════
    # UTILITY METHODS
    # ════════════════════════════════════════════════════════════════════════════
    
    def get_schema(
        self,
        df: Union[pl.DataFrame, pl.LazyFrame]
    ) -> Dict[str, Any]:
        """
        Get DataFrame schema.
        
        Args:
            df: DataFrame or LazyFrame
        
        Returns:
            Schema dictionary
        """
        if isinstance(df, pl.LazyFrame):
            schema = df.schema
        else:
            schema = df.schema
        
        return {name: str(dtype) for name, dtype in schema.items()}
    
    def get_stats(
        self,
        df: Union[pl.DataFrame, pl.LazyFrame]
    ) -> Dict[str, Any]:
        """
        Get DataFrame statistics.
        
        Args:
            df: DataFrame or LazyFrame
        
        Returns:
            Statistics dictionary
        """
        if isinstance(df, pl.LazyFrame):
            df = df.collect()
        
        return {
            "rows": len(df),
            "columns": len(df.columns),
            "memory_usage_bytes": df.estimated_size(),
            "schema": self.get_schema(df),
        }
    
    def collect(
        self,
        df: Union[pl.DataFrame, pl.LazyFrame]
    ) -> pl.DataFrame:
        """
        Collect LazyFrame to DataFrame.
        
        Args:
            df: DataFrame or LazyFrame
        
        Returns:
            DataFrame
        """
        if isinstance(df, pl.LazyFrame):
            return df.collect(streaming=self.streaming)
        return df


# ════════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════

def create_polars_processor(
    lazy: bool = True,
    streaming: bool = False,
    num_threads: Optional[int] = None,
    **kwargs
) -> PolarsProcessor:
    """
    Factory function to create Polars processor.
    
    Args:
        lazy: Use lazy evaluation
        streaming: Enable streaming for large datasets
        num_threads: Number of threads (None = auto)
        **kwargs: Additional processor arguments
    
    Returns:
        PolarsProcessor instance
    """
    return PolarsProcessor(
        lazy=lazy,
        streaming=streaming,
        num_threads=num_threads,
        **kwargs
    )

