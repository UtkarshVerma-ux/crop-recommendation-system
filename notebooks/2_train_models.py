import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder  # NEW: Add this
import xgboost as xgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🤖 pH-STRATIFIED MODEL TRAINING SYSTEM")
print("="*70)

# Load prepared datasets
print("\n📂 Loading prepared datasets...")
with open('../data/processed_datasets.pkl', 'rb') as f:
    datasets = pickle.load(f)

print("✅ Datasets loaded successfully!")
print(f"   Zones: {list(datasets.keys())}")

# ============================================
# MAIN INNOVATION: Train separate models for each pH zone
# ============================================

def train_ensemble_model(X_train, y_train, X_test, y_test, zone_name):
    """
    Trains 3 models and combines them (ensemble)
    Returns the best performing model
    """
    print(f"\n{'='*70}")
    print(f"🎯 TRAINING MODEL FOR {zone_name.upper()} SOIL")
    print(f"{'='*70}")
    
    print(f"\n📊 Dataset Info:")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}")
    print(f"   Features: {X_train.columns.tolist()}")
    print(f"   Unique crops: {y_train.nunique()}")
    
    # ============================================
    # NEW: Encode labels for XGBoost
    # ============================================
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    print(f"\n🔄 Label encoding:")
    print(f"   Original labels: {label_encoder.classes_[:5]}...")
    print(f"   Encoded as: 0, 1, 2, 3, 4...")
    
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200, 
            max_depth=10, 
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=150,
            max_depth=8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss'
        ),
        'CatBoost': CatBoostClassifier(
            iterations=150,
            depth=8,
            random_state=42,
            verbose=False
        )
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n🔧 Training {model_name}...")
        
        # ============================================
        # FIXED: Use encoded labels for XGBoost, original for others
        # ============================================
        if model_name == 'XGBoost':
            # XGBoost needs numeric labels
            model.fit(X_train, y_train_encoded)
            y_pred_encoded = model.predict(X_test)
            # Convert back to original labels
            y_pred = label_encoder.inverse_transform(y_pred_encoded)
        else:
            # Random Forest and CatBoost can handle string labels
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        results[model_name] = {
            'model': model,
            'accuracy': accuracy,
            'predictions': y_pred
        }
        
        print(f"   ✅ Accuracy: {accuracy*100:.2f}%")
    
    # Find best model
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    best_model = results[best_model_name]['model']
    best_accuracy = results[best_model_name]['accuracy']
    
    print(f"\n🏆 Best Model: {best_model_name} ({best_accuracy*100:.2f}%)")
    
    # Show detailed report for best model
    y_pred_best = results[best_model_name]['predictions']
    print(f"\n📊 Detailed Classification Report for {best_model_name}:")
    print(classification_report(y_test, y_pred_best, zero_division=0))
    
    # Calculate feature importance
    print(f"\n🔍 Feature Importance:")
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n   Top Features:")
        for idx, row in feature_importance.head(7).iterrows():
            print(f"      {row['feature']}: {row['importance']:.4f}")
    else:
        feature_importance = None
    
    # ============================================
    # NEW: Save label encoder with model
    # ============================================
    return best_model, best_accuracy, best_model_name, results, feature_importance, label_encoder

# ============================================
# Train models for all three pH zones
# ============================================

print("\n" + "="*70)
print("🚀 STARTING TRAINING FOR ALL pH ZONES")
print("="*70)

trained_models = {}
all_results = {}

for zone_name, (X_train, X_test, y_train, y_test) in datasets.items():
    model, accuracy, model_type, zone_results, feature_importance, label_encoder = train_ensemble_model(
        X_train, y_train, X_test, y_test, zone_name
    )
    
    trained_models[zone_name] = {
        'model': model,
        'accuracy': accuracy,
        'model_type': model_type,
        'feature_names': X_train.columns.tolist(),
        'feature_importance': feature_importance,
        'crops': sorted(y_train.unique().tolist()),
        'n_samples_train': len(X_train),
        'n_samples_test': len(X_test),
        'label_encoder': label_encoder  # NEW: Save encoder
    }
    
    all_results[zone_name] = zone_results

# ============================================
# Cross-zone comparison
# ============================================

print("\n" + "="*70)
print("📊 CROSS-ZONE MODEL COMPARISON")
print("="*70)

comparison_data = []
for zone, info in trained_models.items():
    comparison_data.append({
        'Zone': zone.capitalize(),
        'Best Model': info['model_type'],
        'Accuracy (%)': f"{info['accuracy']*100:.2f}",
        'Train Samples': info['n_samples_train'],
        'Test Samples': info['n_samples_test'],
        'Crops': len(info['crops'])
    })

comparison_df = pd.DataFrame(comparison_data)
print("\n" + comparison_df.to_string(index=False))

# Calculate average accuracy
avg_accuracy = np.mean([info['accuracy'] for info in trained_models.values()])
print(f"\n📈 Average Accuracy Across All Zones: {avg_accuracy*100:.2f}%")

# ============================================
# Save models and results
# ============================================

print("\n" + "="*70)
print("💾 SAVING MODELS AND RESULTS")
print("="*70)

# Save main models (for API and inference)
with open('../models/ph_specific_models.pkl', 'wb') as f:
    pickle.dump(trained_models, f)
print("✅ Saved: ph_specific_models.pkl")

# Save all ensemble results (for analysis)
with open('../models/all_ensemble_results.pkl', 'wb') as f:
    pickle.dump(all_results, f)
print("✅ Saved: all_ensemble_results.pkl")

# Save comparison table
comparison_df.to_csv('../models/model_comparison.csv', index=False)
print("✅ Saved: model_comparison.csv")

# ============================================
# Final summary
# ============================================

print("\n" + "="*70)
print("🎉 MODEL TRAINING COMPLETE!")
print("="*70)

print("\n📊 Summary of Results:")
print("-" * 70)
for zone, info in trained_models.items():
    print(f"{zone.capitalize()} Soil:")
    print(f"   Model: {info['model_type']}")
    print(f"   Accuracy: {info['accuracy']*100:.2f}%")
    print(f"   Crops: {len(info['crops'])} ({', '.join(info['crops'][:5])}...)")
    print()

print("📁 Files saved:")
print("   ✅ ph_specific_models.pkl (main models)")
print("   ✅ all_ensemble_results.pkl (detailed results)")
print("   ✅ model_comparison.csv (comparison table)")

print("\n🎯 Next steps:")
print("   1. Run: python 3_add_explainability.py")
print("   2. Run: python 4_soil_fertility_assessment.py (NEW!)")
print("   3. Run: python 5_dice_soil_improvement.py (NEW!)")

print("\n" + "="*70)