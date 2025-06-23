#!/usr/bin/env python3
"""
PP3 Pandas - Notebook Validation Script
Author: George Dorochov
Email: jordanaftermidnight@gmail.com

This script validates that all required libraries are installed
and can be imported successfully.
"""

import sys
import importlib

def test_imports():
    """Test that all required packages can be imported."""
    required_packages = [
        'pandas',
        'numpy', 
        'matplotlib',
        'seaborn',
        'datetime'
    ]
    
    print("PP3 Pandas - Testing Required Packages")
    print("=" * 50)
    
    all_passed = True
    
    for package in required_packages:
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'Unknown')
            print(f"✅ {package:12} - Version: {version}")
        except ImportError as e:
            print(f"❌ {package:12} - Import failed: {e}")
            all_passed = False
    
    print("=" * 50)
    
    if all_passed:
        print("🎉 All packages imported successfully!")
        print("📓 You can now run the PP3_Pandas_Complete.ipynb notebook")
        return True
    else:
        print("⚠️  Some packages failed to import.")
        print("💡 Please install missing packages using: pip install -r requirements.txt")
        return False

def test_basic_operations():
    """Test basic pandas operations."""
    try:
        import pandas as pd
        import numpy as np
        
        print("\nTesting basic operations...")
        
        # Test DataFrame creation
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        
        # Test basic operations
        assert len(df) == 3
        assert list(df.columns) == ['A', 'B']
        assert df['A'].sum() == 6
        
        print("✅ Basic DataFrame operations working")
        
        # Test numpy operations
        arr = np.array([1, 2, 3, 4, 5])
        assert arr.mean() == 3.0
        assert arr.std() > 1.0
        
        print("✅ Basic NumPy operations working")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic operations test failed: {e}")
        return False

if __name__ == "__main__":
    print("PP3 Pandas Project - Environment Validation")
    print(f"Python version: {sys.version}")
    print()
    
    imports_ok = test_imports()
    
    if imports_ok:
        operations_ok = test_basic_operations()
        
        if operations_ok:
            print("\n🚀 Environment is ready for PP3 Pandas notebook!")
            sys.exit(0)
        else:
            print("\n⚠️  Environment setup incomplete")
            sys.exit(1)
    else:
        print("\n❌ Please install required packages first")
        sys.exit(1)