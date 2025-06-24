#!/usr/bin/env python3
"""
Test script for enhanced PP3 Pandas features
Author: George Dorochov
Email: jordanaftermidnight@gmail.com
"""

import pandas as pd
import numpy as np
import logging
import sys
from functools import wraps
from contextlib import contextmanager
import time
import traceback
from typing import Optional, List, Union, Callable, Any

# Set up logging for testing
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Custom Exception Classes
class DataAnalysisError(Exception):
    """Base exception for data analysis operations"""
    pass

class DataValidationError(DataAnalysisError):
    """Raised when data validation fails"""
    pass

class InsufficientDataError(DataAnalysisError):
    """Raised when dataset is too small for analysis"""
    pass

class ProcessingError(DataAnalysisError):
    """Raised when data processing fails"""
    pass

# Error handling decorator
def error_handler(operation_name: str = "Operation"):
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                logger.info(f"Starting {operation_name}: {func.__name__}")
                result = func(*args, **kwargs)
                logger.info(f"Successfully completed {operation_name}: {func.__name__}")
                return result
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}")
                raise
        return wrapper
    return decorator

# Memory monitoring context manager
@contextmanager
def memory_monitor(operation_name: str = "Operation"):
    try:
        import psutil
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024**2
        logger.info(f"Starting {operation_name} - Memory: {start_memory:.2f} MB")
        yield
        end_memory = process.memory_info().rss / 1024**2
        logger.info(f"Completed {operation_name} - Memory: {end_memory:.2f} MB")
    except ImportError:
        logger.warning("psutil not available - memory monitoring disabled")
        yield

# Data validation function
@error_handler("Data Validation")
def validate_dataframe(df: pd.DataFrame, min_rows: int = 1, required_columns: Optional[List[str]] = None) -> bool:
    if df is None:
        raise DataValidationError("DataFrame is None")
    
    if df.empty:
        raise InsufficientDataError("DataFrame is empty")
    
    if len(df) < min_rows:
        raise InsufficientDataError(f"DataFrame has {len(df)} rows, minimum required: {min_rows}")
    
    if required_columns:
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise DataValidationError(f"Missing required columns: {missing_columns}")
    
    logger.info(f"DataFrame validation passed: {df.shape[0]} rows, {df.shape[1]} columns")
    return True

# Data quality check function
@error_handler("Data Quality Check")
def check_data_quality(df: pd.DataFrame) -> dict:
    validate_dataframe(df)
    
    quality_report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': len(df.select_dtypes(include=['object', 'category']).columns),
        'completeness': 1 - (df.isnull().sum().sum() / df.size)
    }
    
    logger.info(f"Data quality: {quality_report['completeness']:.2%} complete")
    return quality_report

# Memory optimization function
@error_handler("Data Type Optimization")
def optimize_datatypes(df: pd.DataFrame) -> pd.DataFrame:
    original_memory = df.memory_usage(deep=True).sum()
    df_optimized = df.copy()
    
    # Optimize integer columns
    for col in df_optimized.select_dtypes(include=['int64']).columns:
        col_min = df_optimized[col].min()
        col_max = df_optimized[col].max()
        
        if col_min >= 0:  # Unsigned integers
            if col_max < 255:
                df_optimized[col] = df_optimized[col].astype('uint8')
            elif col_max < 65535:
                df_optimized[col] = df_optimized[col].astype('uint16')
        else:  # Signed integers
            if col_min > -128 and col_max < 127:
                df_optimized[col] = df_optimized[col].astype('int8')
            elif col_min > -32768 and col_max < 32767:
                df_optimized[col] = df_optimized[col].astype('int16')
    
    # Convert object columns with low cardinality to category
    for col in df_optimized.select_dtypes(include=['object']).columns:
        if df_optimized[col].nunique() / len(df_optimized) < 0.5:
            df_optimized[col] = df_optimized[col].astype('category')
    
    optimized_memory = df_optimized.memory_usage(deep=True).sum()
    memory_reduction = (original_memory - optimized_memory) / original_memory * 100
    
    logger.info(f"Memory optimization: {memory_reduction:.1f}% reduction")
    return df_optimized

def run_tests():
    """Run comprehensive tests for enhanced features"""
    print("="*60)
    print("PP3 PANDAS - ENHANCED FEATURES TEST SUITE")
    print("="*60)
    
    # Test 1: Basic functionality
    print("\n1. Testing Basic Functionality")
    try:
        df = pd.DataFrame({
            'id': range(1000),
            'category': np.random.choice(['A', 'B', 'C'], 1000),
            'value': np.random.randn(1000)
        })
        print(f"✅ Created test DataFrame: {df.shape}")
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False
    
    # Test 2: Data validation
    print("\n2. Testing Data Validation")
    try:
        validate_dataframe(df, min_rows=100, required_columns=['id', 'category'])
        print("✅ Data validation passed")
    except Exception as e:
        print(f"❌ Data validation test failed: {e}")
        return False
    
    # Test 3: Data quality assessment
    print("\n3. Testing Data Quality Assessment")
    try:
        quality_report = check_data_quality(df)
        print(f"✅ Data quality assessment completed: {quality_report['completeness']:.2%} complete")
    except Exception as e:
        print(f"❌ Data quality test failed: {e}")
        return False
    
    # Test 4: Memory optimization
    print("\n4. Testing Memory Optimization")
    try:
        with memory_monitor("Memory Optimization Test"):
            optimized_df = optimize_datatypes(df)
        
        original_memory = df.memory_usage(deep=True).sum() / 1024**2
        optimized_memory = optimized_df.memory_usage(deep=True).sum() / 1024**2
        print(f"✅ Memory optimization: {original_memory:.2f}MB → {optimized_memory:.2f}MB")
    except Exception as e:
        print(f"❌ Memory optimization test failed: {e}")
        return False
    
    # Test 5: Error handling
    print("\n5. Testing Error Handling")
    test_cases = [
        ("Empty DataFrame", pd.DataFrame(), False),
        ("Missing columns", df.drop('category', axis=1), False),
        ("Valid DataFrame", df.head(100), True)
    ]
    
    error_tests_passed = 0
    for test_name, test_df, should_pass in test_cases:
        try:
            validate_dataframe(test_df, min_rows=10, required_columns=['category'])
            if should_pass:
                print(f"✅ {test_name}: Passed as expected")
                error_tests_passed += 1
            else:
                print(f"❌ {test_name}: Should have failed but passed")
        except DataAnalysisError:
            if not should_pass:
                print(f"✅ {test_name}: Failed as expected")
                error_tests_passed += 1
            else:
                print(f"❌ {test_name}: Should have passed but failed")
        except Exception as e:
            print(f"⚠️  {test_name}: Unexpected error - {e}")
    
    if error_tests_passed == len(test_cases):
        print("✅ Error handling tests passed")
    else:
        print(f"❌ Error handling tests: {error_tests_passed}/{len(test_cases)} passed")
        return False
    
    # Test 6: Performance with larger dataset
    print("\n6. Testing Performance with Larger Dataset")
    try:
        large_df = pd.DataFrame({
            'id': range(100000),
            'category': np.random.choice(['A', 'B', 'C', 'D', 'E'], 100000),
            'value1': np.random.randn(100000),
            'value2': np.random.randn(100000),
            'date': pd.date_range('2020-01-01', periods=100000, freq='H')
        })
        
        start_time = time.time()
        quality_report = check_data_quality(large_df)
        end_time = time.time()
        
        print(f"✅ Large dataset test: {large_df.shape[0]:,} rows processed in {end_time - start_time:.2f}s")
        print(f"   Memory usage: {quality_report['memory_usage_mb']:.2f} MB")
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False
    
    print("\n" + "="*60)
    print("🎉 ALL TESTS PASSED!")
    print("✅ Enhanced error handling working correctly")
    print("✅ Data validation and quality assessment functional")
    print("✅ Memory optimization operational")
    print("✅ Performance monitoring active")
    print("✅ Professional logging implemented")
    print("="*60)
    
    return True

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)