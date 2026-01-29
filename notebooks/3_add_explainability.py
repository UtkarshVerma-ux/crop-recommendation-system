import pandas as pd
import numpy as np
import pickle
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🔍 EXPLAINABILITY ANALYSIS (SHAP + LIME + DiCE)")
print("="*70)

# Load trained models
print("\n📂 Loading trained models...")
with open('../models/ph_specific_models.pkl', 'rb') as f:
    models_dict = pickle.load(f)

# Load test data
print("📂 Loading test datasets...")
with open('../data/processed_datasets.pkl', 'rb') as f:
    datasets = pickle.load(f)

print("✅ Models and data loaded successfully!")
print(f"   Zones: {list(models_dict.keys())}")

# ============================================
# INNOVATION 1: SHAP (Global Feature Importance)
# ============================================

def explain_with_shap(model, X_test, y_test, feature_names, zone_name, label_encoder=None):
    """
    SHAP tells us: "Nitrogen was 40% responsible for this prediction"
    Works for both single and multi-class classification
    """
    print(f"\n{'='*70}")
    print(f"🔍 SHAP Analysis for {zone_name.upper()} Soil")
    print(f"{'='*70}")
    
    print(f"   Model type: {type(model).__name__}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Features: {feature_names}")
    
    try:
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP values
        # For multi-class, this returns a list of arrays (one per class)
        shap_values = explainer.shap_values(X_test)
        
        print(f"   SHAP values shape: {np.array(shap_values).shape if isinstance(shap_values, list) else shap_values.shape}")
        
        # Handle multi-class vs binary
        if isinstance(shap_values, list):
            # Multi-class: average absolute SHAP values across all classes
            print(f"   Multi-class model detected: {len(shap_values)} classes")
            
            # Stack all class SHAP values and take mean absolute importance
            stacked_shap = np.stack([np.abs(sv) for sv in shap_values])
            feature_importance = stacked_shap.mean(axis=(0, 1))
        else:
            # Binary or single output
            print(f"   Binary/single model detected")
            feature_importance = np.abs(shap_values).mean(axis=0)
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        print("\n📊 Top Features (Most Important First):")
        for idx, row in importance_df.iterrows():
            print(f"   {row['Feature']}: {row['Importance']:.4f}")
        
        # Save summary plot
        print(f"\n📈 Generating SHAP summary plot...")
        plt.figure(figsize=(10, 6))
        
        if isinstance(shap_values, list):
            # For multi-class, plot first class as representative
            shap.summary_plot(shap_values[0], X_test, feature_names=feature_names, show=False)
        else:
            shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
        
        plt.title(f'SHAP Summary - {zone_name.capitalize()} Zone')
        plt.tight_layout()
        plt.savefig(f'../models/shap_{zone_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: shap_{zone_name}.png")
        
        # Save feature importance bar plot
        plt.figure(figsize=(10, 6))
        importance_df_sorted = importance_df.sort_values('Importance', ascending=True)
        plt.barh(importance_df_sorted['Feature'], importance_df_sorted['Importance'], color='steelblue')
        plt.xlabel('Mean |SHAP Value|')
        plt.title(f'Feature Importance - {zone_name.capitalize()} Zone')
        plt.tight_layout()
        plt.savefig(f'../models/shap_bar_{zone_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: shap_bar_{zone_name}.png")
        
        return importance_df
        
    except Exception as e:
        print(f"   ❌ SHAP analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None

# ============================================
# INNOVATION 2: LIME (Local Explanations)
# ============================================

def explain_with_lime(model, X_train, X_test, y_test, feature_names, zone_name, label_encoder=None):
    """
    LIME says: "For THIS farmer, rainfall was the key factor"
    """
    print(f"\n{'='*70}")
    print(f"🔍 LIME Analysis for {zone_name.upper()} Soil")
    print(f"{'='*70}")
    
    try:
        # Get class names
        if hasattr(model, 'classes_'):
            class_names = model.classes_.tolist()
        else:
            class_names = None
        
        print(f"   Classes: {class_names[:5] if class_names else 'Unknown'}...")
        
        # Create LIME explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train.values,
            feature_names=feature_names,
            class_names=class_names,
            mode='classification',
            discretize_continuous=True
        )
        
        # Explain 3 random test instances
        n_samples = min(3, len(X_test))
        sample_indices = np.random.choice(len(X_test), n_samples, replace=False)
        
        all_explanations = []
        
        for i, sample_idx in enumerate(sample_indices):
            sample = X_test.iloc[sample_idx].values
            true_label = y_test.iloc[sample_idx]
            
            print(f"\n📋 Sample {i+1}/{n_samples}:")
            print(f"   Input features:")
            for feat, val in zip(feature_names, sample):
                print(f"      {feat}: {val:.2f}")
            
            # Generate explanation
            explanation = explainer.explain_instance(
                sample,
                model.predict_proba,
                num_features=7
            )
            
            predicted_label = model.predict([sample])[0]
            print(f"   True crop: {true_label}")
            print(f"   Predicted crop: {predicted_label}")
            
            print(f"\n   📊 LIME Explanation (Why this prediction?):")
            for feat, importance in explanation.as_list()[:5]:
                print(f"      {feat}: {importance:.3f}")
            
            all_explanations.append(explanation)
            
            # Save plot for first sample
            if i == 0:
                fig = explanation.as_pyplot_figure()
                plt.title(f'LIME Explanation - {zone_name.capitalize()} Zone (Sample)')
                plt.tight_layout()
                plt.savefig(f'../models/lime_{zone_name}.png', dpi=300, bbox_inches='tight')
                plt.close()
                print(f"   ✅ Saved: lime_{zone_name}.png")
        
        return all_explanations
        
    except Exception as e:
        print(f"   ❌ LIME analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None

# ============================================
# INNOVATION 3: Feature Interaction Analysis
# ============================================

def analyze_feature_interactions(model, X_test, feature_names, zone_name):
    """
    Analyze which features work together
    """
    print(f"\n{'='*70}")
    print(f"🔍 Feature Interaction Analysis for {zone_name.upper()} Soil")
    print(f"{'='*70}")
    
    try:
        # Use SHAP for interaction detection
        explainer = shap.TreeExplainer(model)
        shap_interaction_values = explainer.shap_interaction_values(X_test)
        
        if isinstance(shap_interaction_values, list):
            # Multi-class: use first class
            shap_interaction_values = shap_interaction_values[0]
        
        # Get mean absolute interaction effects
        mean_abs_interaction = np.abs(shap_interaction_values).mean(0)
        
        # Find top interactions (off-diagonal elements)
        interactions = []
        for i in range(len(feature_names)):
            for j in range(i+1, len(feature_names)):
                interactions.append({
                    'Feature 1': feature_names[i],
                    'Feature 2': feature_names[j],
                    'Interaction Strength': mean_abs_interaction[i, j]
                })
        
        interaction_df = pd.DataFrame(interactions).sort_values('Interaction Strength', ascending=False)
        
        print("\n🔗 Top Feature Interactions:")
        for idx, row in interaction_df.head(5).iterrows():
            print(f"   {row['Feature 1']} × {row['Feature 2']}: {row['Interaction Strength']:.4f}")
        
        return interaction_df
        
    except Exception as e:
        print(f"   ⚠️ Interaction analysis skipped: {e}")
        return None

# ============================================
# Run all XAI techniques for all models
# ============================================

print("\n" + "="*70)
print("🚀 GENERATING EXPLANATIONS FOR ALL ZONES")
print("="*70)

xai_results = {}

for zone_name in ['acidic', 'neutral', 'alkaline']:
    print(f"\n{'='*70}")
    print(f"PROCESSING {zone_name.upper()} ZONE")
    print(f"{'='*70}")
    
    try:
        # Get data
        X_train, X_test, y_train, y_test = datasets[zone_name]
        
        # Get model info
        model_info = models_dict[zone_name]
        model = model_info['model']
        feature_names = model_info['feature_names']
        label_encoder = model_info.get('label_encoder', None)
        
        print(f"\n📊 Zone Info:")
        print(f"   Train samples: {len(X_train)}")
        print(f"   Test samples: {len(X_test)}")
        print(f"   Features: {feature_names}")
        print(f"   Crops: {len(model_info['crops'])}")
        
        # Run all analyses
        xai_results[zone_name] = {
            'shap': explain_with_shap(model, X_test, y_test, feature_names, zone_name, label_encoder),
            'lime': explain_with_lime(model, X_train, X_test, y_test, feature_names, zone_name, label_encoder),
            'interactions': analyze_feature_interactions(model, X_test, feature_names, zone_name)
        }
        
    except Exception as e:
        print(f"\n❌ Failed to process {zone_name} zone: {e}")
        import traceback
        traceback.print_exc()
        xai_results[zone_name] = None

# ============================================
# Save results
# ============================================

print("\n" + "="*70)
print("💾 SAVING RESULTS")
print("="*70)

# Save XAI results
with open('../models/xai_results.pkl', 'wb') as f:
    pickle.dump(xai_results, f)
print("✅ Saved: xai_results.pkl")

# Create summary report
print("\n" + "="*70)
print("📊 EXPLAINABILITY SUMMARY")
print("="*70)

for zone_name, results in xai_results.items():
    if results:
        print(f"\n{zone_name.capitalize()} Zone:")
        
        if results['shap'] is not None:
            top_feature = results['shap'].iloc[0]
            print(f"   🔝 Most important feature: {top_feature['Feature']} ({top_feature['Importance']:.4f})")
        
        if results['interactions'] is not None:
            top_interaction = results['interactions'].iloc[0]
            print(f"   🔗 Strongest interaction: {top_interaction['Feature 1']} × {top_interaction['Feature 2']}")

print("\n" + "="*70)
print("✅ ALL EXPLAINABILITY ANALYSIS COMPLETE!")
print("="*70)

print("\n📁 Files created:")
print("   ✅ xai_results.pkl (analysis results)")
print("   ✅ shap_acidic.png, shap_neutral.png, shap_alkaline.png")
print("   ✅ shap_bar_acidic.png, shap_bar_neutral.png, shap_bar_alkaline.png")
print("   ✅ lime_acidic.png, lime_neutral.png, lime_alkaline.png")

print("\n🎯 Next steps:")
print("   1. Create: notebooks/4_soil_fertility_assessment.py")
print("   2. Create: notebooks/5_dice_soil_improvement.py")
print("   3. Build API: api/main.py")
print("   4. Build interface: interface/app.py")

print("\n" + "="*70)