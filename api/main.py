"""
pH-Stratified Crop Recommendation API with Full XAI Support
============================================================
Includes: SHAP, DiCE, and LIME for complete explainability
"""

import warnings
warnings.filterwarnings('ignore')

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict
from datetime import datetime
import pickle
import pandas as pd
import numpy as np
import requests
import sys
import os

# XAI Libraries
import shap
import dice_ml
import lime
import lime.lime_tabular

# Weather API configuration
WEATHER_API_KEY = "15b3a9327f4039ff4e81dffc63a1f616"
WEATHER_API_URL = "http://api.openweathermap.org/data/2.5/weather"

# ============================================
# XAI EXPLAINER CLASS
# ============================================

class XAIExplainer:
    """
    Comprehensive XAI explanations using SHAP, DiCE, and LIME
    """
    
    def __init__(self, models_dict, scalers, training_data=None):
        self.models_dict = models_dict
        self.scalers = scalers
        self.training_data = training_data
        
        # Initialize SHAP explainers for each zone
        self.shap_explainers = {}
        for zone in models_dict.keys():
            model = models_dict[zone]['model']
            self.shap_explainers[zone] = shap.TreeExplainer(model)
        
        print("✅ SHAP explainers initialized for all pH zones")
        
        # Initialize DiCE (if training data available)
        self.dice_explainers = {}
        if training_data is not None:
            for zone in models_dict.keys():
                try:
                    zone_data = training_data[training_data['pH_zone'] == zone].copy()
                    if len(zone_data) > 0:
                        # Prepare DiCE data
                        features = models_dict[zone]['feature_names']
                        
                        dice_data = dice_ml.Data(
                            dataframe=zone_data[features + ['crop']],
                            continuous_features=features,
                            outcome_name='crop'
                        )
                        
                        dice_model = dice_ml.Model(
                            model=models_dict[zone]['model'],
                            backend='sklearn'
                        )
                        
                        self.dice_explainers[zone] = dice_ml.Dice(
                            dice_data,
                            dice_model,
                            method='random'
                        )
                    
                except Exception as e:
                    print(f"⚠️ DiCE initialization failed for {zone} zone: {e}")
            
            print("✅ DiCE explainers initialized")
        
        # LIME will be initialized on-demand (lighter weight)
        print("✅ XAI system ready (SHAP + DiCE + LIME)")
    
    def get_shap_explanation(self, X, zone):
        """Generate SHAP explanation for a prediction"""
        try:
            explainer = self.shap_explainers[zone]
            shap_values = explainer.shap_values(X)
            
            # Get feature importance
            if isinstance(shap_values, list):
                # Multi-class: take mean across classes
                mean_shap = np.mean([np.abs(sv) for sv in shap_values], axis=0)
                feature_importance = np.abs(mean_shap[0])
            else:
                feature_importance = np.abs(shap_values[0])
            
            # Get feature names
            feature_names = self.models_dict[zone]['feature_names']
            
            # Create importance dict
            importance_dict = dict(zip(feature_names, feature_importance))
            
            # Normalize to percentages
            total = sum(importance_dict.values())
            importance_pct = {k: (v/total)*100 for k, v in importance_dict.items()}
            
            # Sort by importance
            importance_pct = dict(sorted(
                importance_pct.items(),
                key=lambda x: x[1],
                reverse=True
            ))
            
            # Generate text explanation
            top_3 = list(importance_pct.items())[:3]
            explanation_text = "Key factors: "
            explanation_text += ", ".join([
                f"{feat} ({imp:.1f}%)" 
                for feat, imp in top_3
            ])
            
            return {
                'feature_importance': importance_pct,
                'explanation_text': explanation_text,
                'top_features': top_3,
                'method': 'SHAP'
            }
            
        except Exception as e:
            print(f"⚠️ SHAP explanation error: {e}")
            return {
                'feature_importance': {},
                'explanation_text': 'SHAP explanation unavailable',
                'method': 'SHAP',
                'error': str(e)
            }
    
    def get_dice_counterfactuals(self, X_dict, zone, current_crop, target_crop=None, num_counterfactuals=3):
        """Generate DiCE counterfactual explanations"""
        try:
            if zone not in self.dice_explainers:
                return {
                    'counterfactuals': [],
                    'message': f'DiCE not available for {zone} zone',
                    'method': 'DiCE'
                }
            
            dice_exp = self.dice_explainers[zone]
            
            # Generate counterfactuals
            query_instance = pd.DataFrame([X_dict])
            
            cf_examples = dice_exp.generate_counterfactuals(
                query_instance,
                total_CFs=num_counterfactuals,
                desired_class=target_crop if target_crop else None
            )
            
            # Extract counterfactuals
            counterfactuals = []
            
            if hasattr(cf_examples, 'cf_examples_list'):
                for cf in cf_examples.cf_examples_list:
                    if cf is not None:
                        cf_dict = cf.to_dict('records')[0]
                        
                        # Calculate changes
                        changes = {}
                        for key in X_dict.keys():
                            if abs(cf_dict.get(key, X_dict[key]) - X_dict[key]) > 0.01:
                                changes[key] = {
                                    'from': X_dict[key],
                                    'to': cf_dict.get(key, X_dict[key]),
                                    'change': cf_dict.get(key, X_dict[key]) - X_dict[key]
                                }
                        
                        counterfactuals.append({
                            'suggested_crop': cf_dict.get('crop', 'Unknown'),
                            'changes_needed': changes,
                            'feasible': len(changes) <= 3  # Feasible if few changes
                        })
            
            return {
                'counterfactuals': counterfactuals[:num_counterfactuals],
                'current_crop': current_crop,
                'method': 'DiCE',
                'message': f'Generated {len(counterfactuals)} alternative scenarios'
            }
            
        except Exception as e:
            print(f"⚠️ DiCE explanation error: {e}")
            return {
                'counterfactuals': [],
                'method': 'DiCE',
                'error': str(e)
            }
    
    def get_lime_explanation(self, X, X_dict, zone, feature_names):
        """Generate LIME explanation"""
        try:
            model = self.models_dict[zone]['model']
            
            # Create LIME explainer
            if self.training_data is not None:
                zone_data = self.training_data[
                    self.training_data['pH_zone'] == zone
                ][feature_names]
                
                lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                    zone_data.values,
                    feature_names=feature_names,
                    class_names=model.classes_ if hasattr(model, 'classes_') else None,
                    mode='classification'
                )
                
                # Generate explanation
                exp = lime_explainer.explain_instance(
                    X[0],
                    model.predict_proba,
                    num_features=len(feature_names)
                )
                
                # Extract feature importance
                lime_importance = {}
                for feat, weight in exp.as_list():
                    # Parse feature name (LIME returns strings like "feature_name <= value")
                    feat_name = feat.split()[0]
                    if feat_name in feature_names:
                        lime_importance[feat_name] = abs(weight)
                
                # Normalize to percentages
                total = sum(lime_importance.values())
                if total > 0:
                    lime_importance = {k: (v/total)*100 for k, v in lime_importance.items()}
                
                return {
                    'feature_importance': lime_importance,
                    'method': 'LIME',
                    'explanation': 'Local explanation for this specific case'
                }
            else:
                return {
                    'feature_importance': {},
                    'method': 'LIME',
                    'message': 'Training data not available for LIME'
                }
                
        except Exception as e:
            print(f"⚠️ LIME explanation error: {e}")
            return {
                'feature_importance': {},
                'method': 'LIME',
                'error': str(e)
            }

# ============================================
# SOIL FERTILITY ASSESSOR CLASS (Keep existing)
# ============================================

class SoilFertilityAssessor:
    """
    Assesses soil fertility based on NPK, pH, and environmental parameters
    Returns fertility score (0-100) and actionable recommendations
    """
    
    def __init__(self):
        # Optimal ranges for different crops (based on ICAR guidelines)
        self.optimal_ranges = {
            'rice': {'N': (80, 120), 'P': (40, 80), 'K': (30, 60), 'ph': (5.5, 7.0)},
            'wheat': {'N': (100, 150), 'P': (50, 80), 'K': (30, 60), 'ph': (6.0, 7.5)},
            'maize': {'N': (100, 150), 'P': (60, 90), 'K': (40, 80), 'ph': (5.8, 7.0)},
            # ... (keep all existing optimal ranges)
        }
        
        self.general_optimal = {
            'N': (80, 140), 'P': (40, 80), 'K': (40, 80), 'ph': (6.0, 7.0),
            'temperature': (20, 30), 'humidity': (60, 80), 'rainfall': (80, 150)
        }
    
    def calculate_parameter_score(self, value, optimal_range, param_name):
        """Calculate score (0-100) for a single parameter"""
        min_opt, max_opt = optimal_range
        
        # Perfect score if in optimal range
        if min_opt <= value <= max_opt:
            return 100.0
        
        # Below optimal range (DEFICIENCY)
        elif value < min_opt:
            if param_name == 'ph':
                penalty = (min_opt - value) * 15
                return max(0, 100 - penalty)
            else:
                deficit_ratio = (min_opt - value) / min_opt
                penalty = deficit_ratio * 100
                return max(0, 100 - penalty)
        
        # Above optimal range (EXCESS)
        else:
            if param_name == 'ph':
                penalty = (value - max_opt) * 15
                return max(0, 100 - penalty)
            else:
                excess_ratio = (value - max_opt) / max_opt
                penalty = excess_ratio * 80  # Stronger penalty
                return max(0, 100 - penalty)
    
    def assess_fertility(self, soil_data, crop=None):
        """Main fertility assessment function with STRICTER grading"""
        # Choose optimal ranges
        if crop and crop.lower() in self.optimal_ranges:
            optimal = self.optimal_ranges[crop.lower()].copy()
            optimal.update({
                'temperature': self.general_optimal['temperature'],
                'humidity': self.general_optimal['humidity'],
                'rainfall': self.general_optimal['rainfall']
            })
        else:
            optimal = self.general_optimal
        
        # Calculate scores for each parameter
        scores = {}
        for param in ['N', 'P', 'K', 'ph', 'temperature', 'humidity', 'rainfall']:
            if param in soil_data and param in optimal:
                scores[param] = self.calculate_parameter_score(
                    soil_data[param], optimal[param], param
                )
        
        # Use MINIMUM score for critical nutrients
        critical_scores = [scores.get(p, 100) for p in ['N', 'P', 'K', 'ph'] if p in scores]
        environmental_scores = [scores.get(p, 100) for p in ['temperature', 'humidity', 'rainfall'] if p in scores]
        
        if critical_scores:
            min_critical = min(critical_scores)
            avg_critical = sum(critical_scores) / len(critical_scores)
            
            # If any critical nutrient is very low, it drags down the score
            if min_critical < 50:
                critical_component = (min_critical + avg_critical) / 2
            else:
                critical_component = avg_critical
        else:
            critical_component = 100
        
        if environmental_scores:
            avg_environmental = sum(environmental_scores) / len(environmental_scores)
        else:
            avg_environmental = 100
        
        # Overall score: 80% critical nutrients, 20% environment
        overall_score = (critical_component * 0.8) + (avg_environmental * 0.2)
        
        # Stricter status thresholds
        if overall_score >= 85:
            status, emoji = "EXCELLENT", "🟢"
        elif overall_score >= 70:
            status, emoji = "GOOD", "🟡"
        elif overall_score >= 55:
            status, emoji = "MODERATE", "🟠"
        else:
            status, emoji = "POOR", "🔴"
        
        # Identify deficiencies
        deficiencies = {}
        for param, score in scores.items():
            if score < 75:
                current = soil_data[param]
                opt_range = optimal[param]
                if current < opt_range[0]:
                    deficiencies[param] = {
                        'type': 'LOW',
                        'current': current,
                        'optimal': opt_range,
                        'deficit': opt_range[0] - current,
                        'score': score
                    }
                elif current > opt_range[1]:
                    deficiencies[param] = {
                        'type': 'HIGH',
                        'current': current,
                        'optimal': opt_range,
                        'excess': current - opt_range[1],
                        'score': score
                    }
        
        # Generate recommendations
        recommendations = []
        
        if 'N' in deficiencies:
            d = deficiencies['N']
            if d['type'] == 'LOW':
                urea = (d['deficit'] / 0.46) * 1.2
                recommendations.append({
                    'issue': f'Nitrogen Deficiency (Current: {d["current"]:.0f}, Need: {d["optimal"][0]:.0f}+)',
                    'action': f"Apply {urea:.0f} kg/ha Urea",
                    'cost': f"₹{urea*6:.0f}/ha",
                    'timeline': '2-4 weeks'
                })
            else:
                recommendations.append({
                    'issue': f'Excess Nitrogen (Current: {d["current"]:.0f}, Max: {d["optimal"][1]:.0f})',
                    'action': f"Avoid nitrogen fertilizers. Plant legumes next season.",
                    'cost': '₹0/ha',
                    'timeline': '6-12 months'
                })
        
        if 'P' in deficiencies:
            d = deficiencies['P']
            if d['type'] == 'LOW':
                ssp = (d['deficit'] / 0.16) * 1.2
                recommendations.append({
                    'issue': f'Phosphorus Deficiency (Current: {d["current"]:.0f}, Need: {d["optimal"][0]:.0f}+)',
                    'action': f"Apply {ssp:.0f} kg/ha SSP",
                    'cost': f"₹{ssp*8:.0f}/ha",
                    'timeline': '4-6 weeks'
                })
            else:
                recommendations.append({
                    'issue': f'Excess Phosphorus (Current: {d["current"]:.0f}, Max: {d["optimal"][1]:.0f})',
                    'action': f"Avoid phosphorus fertilizers.",
                    'cost': '₹0/ha',
                    'timeline': '12-24 months'
                })
        
        if 'K' in deficiencies:
            d = deficiencies['K']
            if d['type'] == 'LOW':
                mop = (d['deficit'] / 0.60) * 1.2
                recommendations.append({
                    'issue': f'Potassium Deficiency (Current: {d["current"]:.0f}, Need: {d["optimal"][0]:.0f}+)',
                    'action': f"Apply {mop:.0f} kg/ha MOP",
                    'cost': f"₹{mop*15:.0f}/ha",
                    'timeline': '3-5 weeks'
                })
            else:
                recommendations.append({
                    'issue': f'Excess Potassium (Current: {d["current"]:.0f}, Max: {d["optimal"][1]:.0f})',
                    'action': f"Avoid potassium fertilizers.",
                    'cost': '₹0/ha',
                    'timeline': '6-12 months'
                })
        
        if 'ph' in deficiencies:
            d = deficiencies['ph']
            if d['type'] == 'LOW':
                lime = d['deficit'] * 2000
                recommendations.append({
                    'issue': f'Acidic Soil (pH {d["current"]:.1f}, Need: {d["optimal"][0]:.1f}+)',
                    'action': f"Apply {lime:.0f} kg/ha Agricultural Lime",
                    'cost': f"₹{lime*3:.0f}/ha",
                    'timeline': '3-6 months'
                })
            else:
                sulfur = d['excess'] * 500
                recommendations.append({
                    'issue': f'Alkaline Soil (pH {d["current"]:.1f}, Max: {d["optimal"][1]:.1f})',
                    'action': f"Apply {sulfur:.0f} kg/ha Sulfur/Gypsum",
                    'cost': f"₹{sulfur*5:.0f}/ha",
                    'timeline': '6-12 months'
                })
        
        if not recommendations:
            recommendations.append({
                'issue': 'Soil in good condition',
                'action': 'Maintain current practices',
                'cost': '₹0/ha',
                'timeline': 'Test every 6 months'
            })
        
        return {
            'fertility_score': round(overall_score, 1),
            'status': status,
            'status_emoji': emoji,
            'parameter_scores': {k: round(v, 1) for k, v in scores.items()},
            'deficiencies': deficiencies,
            'recommendations': recommendations,
            'optimal_ranges': optimal
        }

# ============================================
# FASTAPI APP INITIALIZATION
# ============================================

app = FastAPI(
    title="Smart Crop Recommendation API with Full XAI",
    description="""
    🌾 pH-Stratified Crop Recommendation with Complete Explainability
    
    **XAI Features:**
    - 🔍 SHAP: Feature importance analysis
    - 🎯 DiCE: Counterfactual scenarios
    - 💡 LIME: Local interpretable explanations
    
    **Key Features:**
    - 99.91% prediction accuracy
    - Automatic weather integration
    - Real-time soil fertility assessment
    - Cost-effective recommendations
    - Full explainability (SHAP + DiCE + LIME)
    
    **Research:** MTech Thesis 2026
    """,
    version="3.0.0",
    contact={
        "name": "Research Team",
        "email": "contact@example.com"
    }
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# LOAD ML MODELS & INITIALIZE XAI
# ============================================

print("="*70)
print("🔄 Loading AI models and XAI systems...")
print("="*70)

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Load models
    models_locations = [
        os.path.join(BASE_DIR, 'models', 'ph_specific_models.pkl'),
        os.path.join(BASE_DIR, '..', 'models', 'ph_specific_models.pkl')
    ]
    
    scalers_locations = [
        os.path.join(BASE_DIR, 'data', 'feature_scalers.pkl'),
        os.path.join(BASE_DIR, '..', 'data', 'feature_scalers.pkl')
    ]
    
    # Try loading training data for DiCE
    training_data_locations = [
        os.path.join(BASE_DIR, 'data', 'training_data.pkl'),
        os.path.join(BASE_DIR, '..', 'data', 'training_data.pkl')
    ]
    
    MODELS_PATH = None
    for path in models_locations:
        if os.path.exists(path):
            MODELS_PATH = path
            break
    
    SCALERS_PATH = None
    for path in scalers_locations:
        if os.path.exists(path):
            SCALERS_PATH = path
            break
    
    TRAINING_DATA_PATH = None
    for path in training_data_locations:
        if os.path.exists(path):
            TRAINING_DATA_PATH = path
            break
    
    if not MODELS_PATH or not SCALERS_PATH:
        raise FileNotFoundError("Models or scalers not found!")
    
    # Load files
    with open(MODELS_PATH, 'rb') as f:
        models_dict = pickle.load(f)
    
    with open(SCALERS_PATH, 'rb') as f:
        scalers = pickle.load(f)
    
    # Load training data if available
    training_data = None
    if TRAINING_DATA_PATH:
        try:
            with open(TRAINING_DATA_PATH, 'rb') as f:
                training_data = pickle.load(f)
            print("✅ Training data loaded for DiCE")
        except:
            print("⚠️ Training data not available, DiCE will be limited")
    
    # Initialize systems
    fertility_assessor = SoilFertilityAssessor()
    xai_explainer = XAIExplainer(models_dict, scalers, training_data)
    
    print("✅ All systems loaded successfully!")
    print(f"   📊 Zones: {list(models_dict.keys())}")
    print(f"   🧠 XAI: SHAP + DiCE + LIME enabled")
    print("="*70)
    
except Exception as e:
    print(f"❌ Error: {e}")
    models_dict = None
    scalers = None
    fertility_assessor = None
    xai_explainer = None

# ============================================
# WEATHER FETCHING (Keep existing)
# ============================================

def fetch_weather_data(location: str, lat: float = None, lon: float = None):
    """Fetch weather data from OpenWeatherMap API"""
    try:
        if lat and lon:
            params = {'lat': lat, 'lon': lon, 'appid': WEATHER_API_KEY, 'units': 'metric'}
        else:
            params = {'q': location, 'appid': WEATHER_API_KEY, 'units': 'metric'}
        
        response = requests.get(WEATHER_API_URL, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        return {
            'temperature': data['main']['temp'],
            'humidity': data['main']['humidity'],
            'rainfall': data.get('rain', {}).get('1h', 0) * 24 * 30,
            'location': data['name'],
            'source': 'OpenWeatherMap API'
        }
    except:
        return {
            'temperature': 25.0,
            'humidity': 75.0,
            'rainfall': 120.0,
            'location': location if location else 'Unknown',
            'source': 'default'
        }

# ============================================
# PYDANTIC MODELS
# ============================================

class SimpleSoilInput(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "N": 90,
                "P": 42,
                "K": 43,
                "ph": 6.5,
                "location": "Lucknow, IN",
                "enable_xai": True
            }
        }
    )
    
    N: float = Field(..., ge=0, le=200)
    P: float = Field(..., ge=0, le=200)
    K: float = Field(..., ge=0, le=200)
    ph: float = Field(..., ge=3, le=10)
    
    location: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    rainfall: Optional[float] = None
    
    # XAI options
    enable_xai: bool = Field(True, description="Enable SHAP/DiCE/LIME explanations")
    xai_methods: Optional[List[str]] = Field(['shap'], description="XAI methods to use: ['shap', 'dice', 'lime']")

# ============================================
# HELPER FUNCTIONS
# ============================================

def get_ph_zone(ph_value: float) -> str:
    if ph_value < 5.5:
        return 'acidic'
    elif ph_value <= 7.5:
        return 'neutral'
    else:
        return 'alkaline'

def predict_crop(soil_data: dict) -> tuple:
    zone = get_ph_zone(soil_data['ph'])
    model_info = models_dict[zone]
    model = model_info['model']
    scaler = scalers[zone]
    
    X = pd.DataFrame([soil_data])
    X = X[model_info['feature_names']]
    X_scaled = scaler.transform(X)
    
    prediction = model.predict(X_scaled)[0]
    
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_scaled)[0]
        confidence = proba.max()
        
        if hasattr(model, 'classes_'):
            crops = model.classes_
            crop_probas = dict(zip(crops, proba))
            crop_probas = dict(sorted(crop_probas.items(), key=lambda x: x[1], reverse=True))
        else:
            crop_probas = {}
    else:
        confidence = 1.0
        crop_probas = {}
    
    return prediction, confidence, crop_probas, zone, X_scaled, X

# ============================================
# API ENDPOINTS
# ============================================

@app.get("/")
async def root():
    return {
        "name": "Smart Crop Recommendation API with Full XAI",
        "version": "3.0.0",
        "status": "online",
        "xai_features": ["SHAP", "DiCE", "LIME"],
        "models_loaded": models_dict is not None,
        "xai_enabled": xai_explainer is not None,
        "endpoints": {
            "prediction": "/predict_smart",
            "weather": "/weather/{location}",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.post("/predict_smart")
async def predict_smart(soil: SimpleSoilInput):
    """
    **Smart Prediction with Full XAI Support**
    
    Includes SHAP, DiCE, and LIME explanations
    """
    if models_dict is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Get weather data
        if soil.temperature is None or soil.humidity is None or soil.rainfall is None:
            if soil.location:
                weather = fetch_weather_data(soil.location)
            elif soil.latitude and soil.longitude:
                weather = fetch_weather_data(None, soil.latitude, soil.longitude)
            else:
                raise HTTPException(status_code=400, detail="Provide location or coordinates")
            
            temperature = weather['temperature']
            humidity = weather['humidity']
            rainfall = weather['rainfall']
            weather_source = weather['source']
        else:
            temperature = soil.temperature
            humidity = soil.humidity
            rainfall = soil.rainfall
            weather_source = "user-provided"
        
        # Build complete soil data
        complete_soil_data = {
            'N': soil.N,
            'P': soil.P,
            'K': soil.K,
            'ph': soil.ph,
            'temperature': temperature,
            'humidity': humidity,
            'rainfall': rainfall
        }
        
        # Predict crop
        crop, confidence, all_probas, zone, X_scaled, X_df = predict_crop(complete_soil_data)
        
        # Assess fertility
        fertility = fertility_assessor.assess_fertility(complete_soil_data, crop=crop)
        
        # XAI Explanations
        xai_results = {}
        
        if soil.enable_xai and xai_explainer:
            xai_methods = soil.xai_methods or ['shap']
            
            # SHAP
            if 'shap' in xai_methods:
                shap_exp = xai_explainer.get_shap_explanation(X_scaled, zone)
                xai_results['shap'] = shap_exp
            
            # DiCE
            if 'dice' in xai_methods:
                dice_exp = xai_explainer.get_dice_counterfactuals(
                    complete_soil_data,
                    zone,
                    crop,
                    num_counterfactuals=3
                )
                xai_results['dice'] = dice_exp
            
            # LIME
            if 'lime' in xai_methods:
                lime_exp = xai_explainer.get_lime_explanation(
                    X_scaled,
                    complete_soil_data,
                    zone,
                    models_dict[zone]['feature_names']
                )
                xai_results['lime'] = lime_exp
        
        # Calculate costs
        total_cost = sum([
            float(rec['cost'].replace('₹', '').replace('/ha', '').replace(',', '')) 
            for rec in fertility['recommendations'] if 'cost' in rec
        ])
        
        # Return comprehensive result
        return {
            "prediction": {
                "recommended_crop": crop,
                "confidence": float(confidence),
                "ph_zone": zone,
                "alternatives": [
                    {"crop": c, "confidence": float(conf)} 
                    for c, conf in list(all_probas.items())[:3]
                ]
            },
            "soil_analysis": {
                "provided_by_user": {
                    "N": soil.N,
                    "P": soil.P,
                    "K": soil.K,
                    "ph": soil.ph
                },
                "weather_data": {
                    "temperature": temperature,
                    "humidity": humidity,
                    "rainfall": rainfall,
                    "source": weather_source
                }
            },
            "fertility": {
                "score": fertility['fertility_score'],
                "status": fertility['status'],
                "deficiencies": len(fertility['deficiencies']),
                "needs_improvement": fertility['fertility_score'] < 65,
                "parameter_scores": fertility['parameter_scores']
            },
            "recommendations": fertility['recommendations'],
            "cost_estimate": {
                "total": f"₹{total_cost:.0f}/ha" if total_cost > 0 else "₹0/ha",
                "timeline": "2-6 weeks for NPK, 3-6 months for pH"
            },
            "xai_explanations": xai_results if soil.enable_xai else {
                "message": "XAI disabled. Set enable_xai=true to get explanations"
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/weather/{location}")
async def get_weather(location: str):
    """Test weather API"""
    return fetch_weather_data(location)

@app.get("/health")
async def health():
    """System health check"""
    return {
        "status": "healthy" if models_dict else "degraded",
        "models_loaded": models_dict is not None,
        "xai_enabled": xai_explainer is not None,
        "xai_features": {
            "shap": True if xai_explainer else False,
            "dice": True if (xai_explainer and xai_explainer.dice_explainers) else False,
            "lime": True if xai_explainer else False
        },
        "timestamp": datetime.now().isoformat()
    }

# ============================================
# RUN SERVER
# ============================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*70)
    print("🚀 SMART CROP RECOMMENDATION API with FULL XAI")
    print("="*70)
    print("\n💡 Features:")
    print("   🔍 SHAP: Feature importance analysis")
    print("   🎯 DiCE: Counterfactual scenarios")
    print("   💡 LIME: Local interpretable explanations")
    print("   🌤️  Auto weather fetching")
    print("   📊 Soil fertility assessment")
    print("\n📖 Documentation: http://localhost:8000/docs")
    print("="*70 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)