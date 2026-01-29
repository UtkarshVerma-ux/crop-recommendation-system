import pandas as pd
import numpy as np

print("🧹 CLEANING MEGA DATASET")
print("="*70)

# Load
df = pd.read_csv('../data/Crop_recommendation_MEGA.csv')
print(f"\n📊 Original: {len(df)} samples, {df['label'].nunique()} crops")

# Clean crop names
crop_mapping = {
    'rice, paddy': 'rice',
    'plantains and others': 'banana',
    'potatoes': 'potato',
    'sweet potatoes': 'sweetpotato',
    'soybeans': 'soybean',
    'kidneybeans': 'kidneybean',
    'mothbeans': 'mothbean',
    'muskmelon': 'melon'
}

df['label'] = df['label'].replace(crop_mapping)

# Remove ambiguous entries
df = df[~df['label'].str.contains('others', case=False, na=False)]

# Remove outliers (1% on each tail)
for col in ['N', 'P', 'K', 'temperature', 'humidity', 'rainfall']:
    lower = df[col].quantile(0.01)
    upper = df[col].quantile(0.99)
    df = df[(df[col] >= lower) & (df[col] <= upper)]

# Ensure valid pH range
df = df[(df['ph'] >= 3.5) & (df['ph'] <= 9.5)]

# Remove crops with < 300 samples (too rare)
crop_counts = df['label'].value_counts()
valid_crops = crop_counts[crop_counts >= 300].index
df = df[df['label'].isin(valid_crops)]

print(f"\n✅ Cleaned: {len(df)} samples, {df['label'].nunique()} crops")

# pH distribution
acidic = len(df[df['ph'] < 5.5])
neutral = len(df[(df['ph'] >= 5.5) & (df['ph'] <= 7.5)])
alkaline = len(df[df['ph'] > 7.5])

print(f"\n🧪 pH Distribution:")
print(f"   Acidic: {acidic} ({acidic/len(df)*100:.1f}%)")
print(f"   Neutral: {neutral} ({neutral/len(df)*100:.1f}%)")
print(f"   Alkaline: {alkaline} ({alkaline/len(df)*100:.1f}%)")

print(f"\n🌾 Crops retained: {sorted(df['label'].unique())}")

# Save cleaned
df.to_csv('../data/Crop_recommendation_MEGA_cleaned.csv', index=False)
print(f"\n💾 Saved to: Crop_recommendation_MEGA_cleaned.csv")