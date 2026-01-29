import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("📊 DATA PREPARATION WITH pH STRATIFICATION (USING REAL DATA)")
print("="*70)

# ============================================
# STEP 1: Load ORIGINAL dataset (REAL Kaggle data!)
# ============================================

print("\n📂 Loading ORIGINAL dataset...")
df = pd.read_csv('../data/Crop_recommendation.csv')

print(f"\n📊 Dataset shape: {df.shape}")
print(f"\n🔍 First few rows:")
print(df.head())

print(f"\n🌾 Unique crops: {df['label'].nunique()}")
print(f"Crop distribution:")
print(df['label'].value_counts())

# ============================================
# STEP 2: Basic cleaning
# ============================================

print("\n🧹 Cleaning data...")

# Lowercase labels
df['label'] = df['label'].str.lower()

# Remove any duplicates
before = len(df)
df = df.drop_duplicates()
print(f"   Removed {before - len(df)} duplicates")

# Check for missing values
if df.isnull().any().any():
    print(f"   Found missing values, removing...")
    df = df.dropna()

print(f"✅ After cleaning: {len(df)} samples, {df['label'].nunique()} crops")

# ============================================
# STEP 3: Check pH distribution
# ============================================

print("\n🧪 pH Statistics:")
print(df['ph'].describe())

# ============================================
# STEP 4: Split by pH zones
# ============================================

print("\n" + "="*70)
print("pH ZONE STRATIFICATION")
print("="*70)

acidic_df = df[df['ph'] < 5.5].copy()
neutral_df = df[(df['ph'] >= 5.5) & (df['ph'] <= 7.5)].copy()
alkaline_df = df[df['ph'] > 7.5].copy()

print(f"\n📈 Original Dataset Split:")
print(f"Acidic (pH < 5.5): {len(acidic_df)} samples ({len(acidic_df)/len(df)*100:.1f}%)")
print(f"Neutral (5.5 ≤ pH ≤ 7.5): {len(neutral_df)} samples ({len(neutral_df)/len(df)*100:.1f}%)")
print(f"Alkaline (pH > 7.5): {len(alkaline_df)} samples ({len(alkaline_df)/len(df)*100:.1f}%)")

print(f"\n🌾 Crops per zone:")
print(f"   Acidic: {acidic_df['label'].value_counts().to_dict()}")
print(f"   Neutral: {neutral_df['label'].value_counts().to_dict()}")
print(f"   Alkaline: {alkaline_df['label'].value_counts().to_dict()}")

# ============================================
# STEP 5: Smart augmentation with SMOTE
# ============================================

def balance_with_smote(df_zone, zone_name, target_size=400):
    """
    Use SMOTE to create realistic synthetic samples
    SMOTE creates samples BETWEEN existing points (better than duplication)
    """
    print(f"\n🔄 {zone_name} Zone:")
    print(f"   Current samples: {len(df_zone)}")
    
    if len(df_zone) >= target_size:
        print(f"   ✅ Sufficient samples, no augmentation needed")
        return df_zone
    
    X = df_zone[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df_zone['label']
    
    # Check minimum samples per class
    class_counts = y.value_counts()
    min_samples = class_counts.min()
    
    print(f"   Classes: {y.nunique()}")
    print(f"   Min samples per class: {min_samples}")
    
    if min_samples < 6:
        print(f"   ⚠️ Too few samples for SMOTE, using simple oversampling")
        n_needed = target_size - len(df_zone)
        df_additional = df_zone.sample(n=n_needed, replace=True, random_state=42)
        df_balanced = pd.concat([df_zone, df_additional], ignore_index=True)
    else:
        try:
            # Use SMOTE with conservative k_neighbors
            k_neighbors = min(5, min_samples - 1)
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
            
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            df_balanced = pd.DataFrame(X_resampled, columns=X.columns)
            df_balanced['label'] = y_resampled
            
            # Sample to target
            if len(df_balanced) > target_size:
                df_balanced = df_balanced.sample(n=target_size, random_state=42)
            
            print(f"   ✅ SMOTE augmentation: {len(df_zone)} → {len(df_balanced)}")
        
        except Exception as e:
            print(f"   ⚠️ SMOTE failed: {e}")
            n_needed = target_size - len(df_zone)
            df_additional = df_zone.sample(n=n_needed, replace=True, random_state=42)
            df_balanced = pd.concat([df_zone, df_additional], ignore_index=True)
            print(f"   ✅ Simple oversampling: {len(df_zone)} → {len(df_balanced)}")
    
    return df_balanced

print("\n" + "="*70)
print("BALANCING pH ZONES")
print("="*70)

acidic_df = balance_with_smote(acidic_df, "Acidic", target_size=400)
alkaline_df = balance_with_smote(alkaline_df, "Alkaline", target_size=400)

print(f"\n✅ After Balancing:")
print(f"Acidic: {len(acidic_df)} samples")
print(f"Neutral: {len(neutral_df)} samples (kept original)")
print(f"Alkaline: {len(alkaline_df)} samples")
print(f"Total: {len(acidic_df) + len(neutral_df) + len(alkaline_df)} samples")

# ============================================
# STEP 6: Prepare datasets with scaling
# ============================================

def prepare_dataset(data, name):
    """
    Split, scale, and prepare dataset for training
    """
    print(f"\n{'='*70}")
    print(f"PREPARING {name.upper()} DATASET")
    print(f"{'='*70}")
    
    X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = data['label']
    
    print(f"\n📊 Dataset info:")
    print(f"   Total samples: {len(data)}")
    print(f"   Unique crops: {y.nunique()}")
    print(f"   Crop distribution: {y.value_counts().to_dict()}")
    
    # Split into train/test
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"\n✅ Split complete (stratified):")
    except:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(f"\n✅ Split complete (random):")
    
    print(f"   Training: {len(X_train)} samples")
    print(f"   Testing: {len(X_test)} samples")
    
    # Apply scaling
    print(f"\n🔧 Applying StandardScaler...")
    scaler = StandardScaler()
    
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    print(f"   ✅ Features scaled")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

acidic_data = prepare_dataset(acidic_df, "Acidic")
neutral_data = prepare_dataset(neutral_df, "Neutral")
alkaline_data = prepare_dataset(alkaline_df, "Alkaline")

# ============================================
# STEP 7: Save everything
# ============================================

print("\n" + "="*70)
print("SAVING PROCESSED DATASETS")
print("="*70)

datasets = {
    'acidic': acidic_data[:4],
    'neutral': neutral_data[:4],
    'alkaline': alkaline_data[:4]
}

with open('../data/processed_datasets.pkl', 'wb') as f:
    pickle.dump(datasets, f)
print("✅ Saved: processed_datasets.pkl")

scalers = {
    'acidic': acidic_data[4],
    'neutral': neutral_data[4],
    'alkaline': alkaline_data[4]
}

with open('../data/feature_scalers.pkl', 'wb') as f:
    pickle.dump(scalers, f)
print("✅ Saved: feature_scalers.pkl")

balanced_dfs = {
    'acidic': acidic_df,
    'neutral': neutral_df,
    'alkaline': alkaline_df
}

with open('../data/balanced_dataframes.pkl', 'wb') as f:
    pickle.dump(balanced_dfs, f)
print("✅ Saved: balanced_dataframes.pkl")

# ============================================
# STEP 8: Final summary
# ============================================

print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)

total_samples = len(acidic_df) + len(neutral_df) + len(alkaline_df)

print(f"\n📊 Total samples: {total_samples}")
print(f"\n🧪 pH Distribution:")
print(f"   Acidic: {len(acidic_df)} ({len(acidic_df)/total_samples*100:.1f}%)")
print(f"   Neutral: {len(neutral_df)} ({len(neutral_df)/total_samples*100:.1f}%)")
print(f"   Alkaline: {len(alkaline_df)} ({len(alkaline_df)/total_samples*100:.1f}%)")

print(f"\n🌾 Crops:")
all_crops = set(acidic_df['label'].unique()) | set(neutral_df['label'].unique()) | set(alkaline_df['label'].unique())
print(f"   Total unique crops: {len(all_crops)}")
print(f"   Crops: {sorted(all_crops)}")

print("\n✅ Data preparation complete!")
print("\n💡 Using REAL Kaggle data with proper feature separation!")
print("\n🎯 Next step: Run 2_train_models.py")