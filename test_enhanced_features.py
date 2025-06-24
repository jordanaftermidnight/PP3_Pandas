#!/usr/bin/env python3
"""
Simple test script for PP3 Pandas enhancements
Author: George Dorochov
Email: jordanaftermidnight@gmail.com
"""

import pandas as pd
import numpy as np
import time

# Simple validation functions
def validate_dataframe(df, min_rows=1):
    """Basic DataFrame validation"""
    if df is None or df.empty:
        raise ValueError("DataFrame is empty or None")
    
    if len(df) < min_rows:
        raise ValueError(f"DataFrame has {len(df)} rows, minimum required: {min_rows}")
    
    return True

def check_data_quality(df):
    """Simple data quality check"""
    return {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
    }

def optimize_memory(df):
    """Basic memory optimization"""
    df_optimized = df.copy()
    
    # Convert object columns to category if beneficial
    for col in df_optimized.select_dtypes(include=['object']).columns:
        if df_optimized[col].nunique() / len(df_optimized) < 0.5:
            df_optimized[col] = df_optimized[col].astype('category')
    
    return df_optimized

def run_tests():
    """Run simple tests for basic features"""
    print("="*50)
    print("PP3 PANDAS - BASIC TESTS")
    print("="*50)
    
    # Test 1: Basic functionality
    print("\n1. Testing Basic Functionality")
    try:
        df = pd.DataFrame({
            'id': range(1000),
            'category': np.random.choice(['A', 'B', 'C'], 1000),
            'value': np.random.randn(1000),
            'text': [f'item_{i}' for i in range(1000)]
        })
        print(f"✅ Created test DataFrame: {df.shape}")
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False
    
    # Test 2: Data validation
    print("\n2. Testing Data Validation")
    try:
        validate_dataframe(df, min_rows=100)
        print("✅ Data validation passed")
    except Exception as e:
        print(f"❌ Data validation test failed: {e}")
        return False
    
    # Test 3: Data quality assessment
    print("\n3. Testing Data Quality")
    try:
        quality_report = check_data_quality(df)
        print(f"✅ Data quality: {quality_report['total_rows']} rows, {quality_report['missing_values']} missing values")
    except Exception as e:
        print(f"❌ Data quality test failed: {e}")
        return False
    
    # Test 4: Memory optimization
    print("\n4. Testing Memory Optimization")
    try:
        original_memory = df.memory_usage(deep=True).sum() / 1024**2
        optimized_df = optimize_memory(df)
        optimized_memory = optimized_df.memory_usage(deep=True).sum() / 1024**2
        print(f"✅ Memory optimization: {original_memory:.2f}MB → {optimized_memory:.2f}MB")
    except Exception as e:
        print(f"❌ Memory optimization test failed: {e}")
        return False
    
    # Test 5: Error handling
    print("\n5. Testing Error Handling")
    try:
        validate_dataframe(pd.DataFrame(), min_rows=10)
        print("❌ Should have failed but passed")
        return False
    except ValueError:
        print("✅ Error handling working correctly")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    
    print("\n" + "="*50)
    print("✅ ALL TESTS PASSED!")
    print("✅ Basic enhancements working correctly")
    print("="*50)
    
    return True

if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)