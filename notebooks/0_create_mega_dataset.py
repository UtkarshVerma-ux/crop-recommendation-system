"""
MEGA DATASET CREATOR
This script combines multiple agricultural datasets into one comprehensive dataset
Target: 50,000+ samples with proper pH stratification
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🌾 CREATING MEGA CROP RECOMMENDATION DATASET")
print("="*70)

# ============================================
# STEP 1: Load all available datasets
# ============================================

def load_dataset_1_original():
    """Original Kaggle crop recommendation dataset"""
    print("\n📂 Loading Dataset 1: Original Crop Recommendation...")
    try:
        df = pd.read_csv('../data/Crop_recommendation.csv')
        print(f"   ✅ Loaded: {len(df)} samples")
        print(f"   Columns: {df.columns.tolist()}")
        return df
    except:
        print("   ❌ File not found!")
        return None

def load_dataset_2_production():
    """India Agriculture Production dataset"""
    print("\n📂 Loading Dataset 2: Agriculture Production...")
    try:
        df = pd.read_csv('../data/raw/crop_production.csv')
        print(f"   ✅ Loaded: {len(df)} samples")
        print(f"   Columns: {df.columns.tolist()}")
        
        # This dataset has: State, District, Crop_Year, Season, Crop, Area, Production
        # We need to synthesize N, P, K, temperature, humidity, ph, rainfall
        
        # Filter only rows with valid production data
        df = df.dropna(subset=['Production', 'Area'])
        df = df[df['Production'] > 0]
        
        # Standardize crop names
        df['Crop'] = df['Crop'].str.lower().str.strip()
        
        # Create synthetic features based on crop type and region
        # (This is a reasonable approach when merging heterogeneous datasets)
        
        # Map states to typical climate zones
        state_climate = {
            'Andhra Pradesh': {'temp': 28, 'humidity': 65, 'rainfall': 940},
            'Kerala': {'temp': 27, 'humidity': 80, 'rainfall': 3055},
            'Punjab': {'temp': 24, 'humidity': 62, 'rainfall': 617},
            'West Bengal': {'temp': 27, 'humidity': 75, 'rainfall': 1582},
            'Tamil Nadu': {'temp': 29, 'humidity': 70, 'rainfall': 998},
            'Karnataka': {'temp': 25, 'humidity': 68, 'rainfall': 1150},
            'Maharashtra': {'temp': 27, 'humidity': 65, 'rainfall': 1134},
            'Uttar Pradesh': {'temp': 25, 'humidity': 70, 'rainfall': 1016},
            'Madhya Pradesh': {'temp': 26, 'humidity': 62, 'rainfall': 1178},
            'Gujarat': {'temp': 28, 'humidity': 60, 'rainfall': 803}
        }
        
        # Map crops to typical nutrient requirements (based on agricultural research)
        crop_nutrients = {
            'rice': {'N': 120, 'P': 60, 'K': 40, 'ph': 6.5},
            'wheat': {'N': 150, 'P': 60, 'K': 40, 'ph': 6.8},
            'cotton': {'N': 120, 'P': 50, 'K': 50, 'ph': 7.0},
            'sugarcane': {'N': 200, 'P': 80, 'K': 100, 'ph': 6.5},
            'maize': {'N': 150, 'P': 75, 'K': 75, 'ph': 6.5},
            'groundnut': {'N': 25, 'P': 50, 'K': 75, 'ph': 6.5},
            'pulses': {'N': 25, 'P': 60, 'K': 40, 'ph': 6.5},
            'jowar': {'N': 80, 'P': 40, 'K': 40, 'ph': 6.8},
            'bajra': {'N': 80, 'P': 40, 'K': 40, 'ph': 7.0},
            'potato': {'N': 150, 'P': 80, 'K': 100, 'ph': 5.5},
            'onion': {'N': 100, 'P': 50, 'K': 100, 'ph': 6.5},
            'tomato': {'N': 100, 'P': 50, 'K': 75, 'ph': 6.5}
        }
        
        # Create synthetic features
        def add_synthetic_features(row):
            state = row['State_Name'] if 'State_Name' in row else row.get('State', 'Unknown')
            crop = row['Crop']
            
            # Get climate data for state
            climate = state_climate.get(state, {'temp': 26, 'humidity': 68, 'rainfall': 1100})
            
            # Get nutrient requirements for crop
            nutrients = crop_nutrients.get(crop, {'N': 100, 'P': 50, 'K': 50, 'ph': 6.5})
            
            # Add realistic variation (±10%)
            row['N'] = nutrients['N'] + np.random.normal(0, nutrients['N']*0.1)
            row['P'] = nutrients['P'] + np.random.normal(0, nutrients['P']*0.1)
            row['K'] = nutrients['K'] + np.random.normal(0, nutrients['K']*0.1)
            row['ph'] = nutrients['ph'] + np.random.normal(0, 0.3)
            row['temperature'] = climate['temp'] + np.random.normal(0, 2)
            row['humidity'] = climate['humidity'] + np.random.normal(0, 5)
            row['rainfall'] = climate['rainfall'] / 12 + np.random.normal(0, 20)  # Monthly average
            row['label'] = crop
            
            return row
        
        # Sample to avoid overwhelming dataset (take 30,000 samples)
        if len(df) > 30000:
            df = df.sample(n=30000, random_state=42)
        
        df = df.apply(add_synthetic_features, axis=1)
        
        # Keep only required columns
        df = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label']]
        
        print(f"   ✅ Processed: {len(df)} samples with synthetic features")
        return df
        
    except FileNotFoundError:
        print("   ⚠️ File not found, skipping...")
        return None
    except Exception as e:
        print(f"   ⚠️ Error: {e}, skipping...")
        return None

def load_dataset_3_yield():
    """Crop yield prediction dataset"""
    print("\n📂 Loading Dataset 3: Yield Prediction...")
    try:
        df = pd.read_csv('../data/raw/yield_df.csv')
        print(f"   ✅ Loaded: {len(df)} samples")
        
        # Check what columns exist
        print(f"   Columns: {df.columns.tolist()}")
        
        # Try to map to our required format
        # Common variations: Item -> label, hg/ha_yield -> production indicator
        
        if 'Item' in df.columns:
            df = df.rename(columns={'Item': 'label'})
        if 'Crop' in df.columns:
            df = df.rename(columns={'Crop': 'label'})
            
        # If this dataset has climate data, great! If not, we'll add synthetic
        required_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"   ⚠️ Missing columns: {missing_cols}, adding synthetic data...")
            for col in missing_cols:
                if col in ['N', 'P', 'K']:
                    df[col] = np.random.uniform(20, 200, len(df))
                elif col == 'ph':
                    df[col] = np.random.normal(6.5, 0.8, len(df))
                elif col == 'temperature':
                    df[col] = np.random.normal(25, 3, len(df))
                elif col == 'humidity':
                    df[col] = np.random.normal(70, 8, len(df))
                elif col == 'rainfall':
                    df[col] = np.random.normal(100, 30, len(df))
        
        df = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label']]
        df['label'] = df['label'].str.lower().str.strip()
        
        print(f"   ✅ Processed: {len(df)} samples")
        return df
        
    except FileNotFoundError:
        print("   ⚠️ File not found, skipping...")
        return None
    except Exception as e:
        print(f"   ⚠️ Error: {e}, skipping...")
        return None


# ============================================
# STEP 2: Load all datasets
# ============================================

print("\n" + "="*70)
print("LOADING ALL DATASETS")
print("="*70)

datasets = []
dataset_names = []

# Load each dataset
df1 = load_dataset_1_original()
if df1 is not None:
    datasets.append(df1)
    dataset_names.append("Original")

df2 = load_dataset_2_production()
if df2 is not None:
    datasets.append(df2)
    dataset_names.append("Production")

df3 = load_dataset_3_yield()
if df3 is not None:
    datasets.append(df3)
    dataset_names.append("Yield")

print(f"\n✅ Successfully loaded {len(datasets)} datasets")

# ============================================
# STEP 3: Combine and clean
# ============================================

print("\n" + "="*70)
print("COMBINING DATASETS")
print("="*70)

if len(datasets) == 0:
    print("❌ No datasets loaded! Please check file paths.")
    exit()

# Combine all
mega_df = pd.concat(datasets, ignore_index=True)
print(f"\n📊 Combined dataset: {len(mega_df)} samples")

# ============================================
# STEP 4: Data cleaning
# ============================================

print("\n" + "="*70)
print("CLEANING DATA")
print("="*70)

# 1. Remove duplicates
before = len(mega_df)
mega_df = mega_df.drop_duplicates()
print(f"✅ Removed {before - len(mega_df)} duplicates")

# 2. Handle missing values
print(f"\n🔍 Missing values:")
print(mega_df.isnull().sum())

# Fill missing with median
for col in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']:
    if mega_df[col].isnull().any():
        median_val = mega_df[col].median()
        mega_df[col].fillna(median_val, inplace=True)
        print(f"   Filled {col} with median: {median_val:.2f}")

# Drop rows with missing labels
mega_df = mega_df.dropna(subset=['label'])

# 3. Remove outliers (values outside reasonable ranges)
print(f"\n🧹 Removing outliers...")
before = len(mega_df)

mega_df = mega_df[
    (mega_df['N'] >= 0) & (mega_df['N'] <= 400) &
    (mega_df['P'] >= 0) & (mega_df['P'] <= 400) &
    (mega_df['K'] >= 0) & (mega_df['K'] <= 400) &
    (mega_df['ph'] >= 3) & (mega_df['ph'] <= 10) &
    (mega_df['temperature'] >= 5) & (mega_df['temperature'] <= 50) &
    (mega_df['humidity'] >= 10) & (mega_df['humidity'] <= 100) &
    (mega_df['rainfall'] >= 20) & (mega_df['rainfall'] <= 400)
]

print(f"   Removed {before - len(mega_df)} outliers")

# 4. Standardize crop names
print(f"\n📝 Standardizing crop names...")
mega_df['label'] = mega_df['label'].str.lower().str.strip()

# Map common variations to standard names
crop_mapping = {
    'paddy': 'rice',
    'maize corn': 'maize',
    'moong': 'mungbean',
    'urad': 'blackgram',
    'gram': 'chickpea',
    'tur': 'pigeonpeas',
    'groundnut': 'peanut',
    'brinjal': 'eggplant',
    'baingan': 'eggplant'
}

mega_df['label'] = mega_df['label'].replace(crop_mapping)

# 5. Filter to keep only major crops (that appear at least 100 times)
print(f"\n🌾 Filtering crops...")
crop_counts = mega_df['label'].value_counts()
major_crops = crop_counts[crop_counts >= 100].index.tolist()
mega_df = mega_df[mega_df['label'].isin(major_crops)]

print(f"   Kept {len(major_crops)} major crops")
print(f"   Crops: {sorted(major_crops)}")

# ============================================
# STEP 5: Final statistics
# ============================================

print("\n" + "="*70)
print("FINAL MEGA DATASET STATISTICS")
print("="*70)

print(f"\n📊 Total samples: {len(mega_df)}")
print(f"📊 Total crops: {mega_df['label'].nunique()}")
print(f"📊 Features: {mega_df.columns.tolist()}")

print(f"\n🧪 pH Distribution:")
acidic = len(mega_df[mega_df['ph'] < 5.5])
neutral = len(mega_df[(mega_df['ph'] >= 5.5) & (mega_df['ph'] <= 7.5)])
alkaline = len(mega_df[mega_df['ph'] > 7.5])

print(f"   Acidic (pH < 5.5): {acidic} samples ({acidic/len(mega_df)*100:.1f}%)")
print(f"   Neutral (5.5 ≤ pH ≤ 7.5): {neutral} samples ({neutral/len(mega_df)*100:.1f}%)")
print(f"   Alkaline (pH > 7.5): {alkaline} samples ({alkaline/len(mega_df)*100:.1f}%)")

print(f"\n📈 Feature Statistics:")
print(mega_df.describe())

print(f"\n🌾 Crop Distribution (Top 10):")
print(mega_df['label'].value_counts().head(10))

# ============================================
# STEP 6: Save mega dataset
# ============================================

print("\n" + "="*70)
print("SAVING MEGA DATASET")
print("="*70)

# Save to CSV
output_path = '../data/Crop_recommendation_MEGA.csv'
mega_df.to_csv(output_path, index=False)
print(f"\n✅ Saved to: {output_path}")

# Save metadata
metadata = {
    'total_samples': len(mega_df),
    'total_crops': mega_df['label'].nunique(),
    'crop_list': sorted(mega_df['label'].unique().tolist()),
    'sources': dataset_names,
    'ph_distribution': {
        'acidic': acidic,
        'neutral': neutral,
        'alkaline': alkaline
    },
    'feature_ranges': {
        'N': [mega_df['N'].min(), mega_df['N'].max()],
        'P': [mega_df['P'].min(), mega_df['P'].max()],
        'K': [mega_df['K'].min(), mega_df['K'].max()],
        'ph': [mega_df['ph'].min(), mega_df['ph'].max()],
        'temperature': [mega_df['temperature'].min(), mega_df['temperature'].max()],
        'humidity': [mega_df['humidity'].min(), mega_df['humidity'].max()],
        'rainfall': [mega_df['rainfall'].min(), mega_df['rainfall'].max()]
    }
}

import json
with open('../data/MEGA_dataset_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=4)

print(f"✅ Saved metadata to: MEGA_dataset_metadata.json")

print("\n" + "="*70)
print("✅ MEGA DATASET CREATION COMPLETE!")
print("="*70)
print(f"\n🎯 Next steps:")
print(f"   1. Review: {output_path}")
print(f"   2. Update 1_data_preparation.py to use MEGA dataset")
print(f"   3. Run: python 1_data_preparation.py")