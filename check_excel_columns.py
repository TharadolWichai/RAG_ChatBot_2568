#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import pandas as pd

# Fix Windows encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Read Excel
df = pd.read_excel("eva 50 excel.xlsx")

print(f"Total rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")
print("\nColumn names:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i}. [{col}]")

print("\nFirst 3 rows preview:")
print(df.head(3).to_string())

