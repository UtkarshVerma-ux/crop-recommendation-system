"""
pH-Stratified Crop Recommendation API with Auto Weather Fetching
=================================================================
Farmers only need to provide: N, P, K, pH, Location
API automatically fetches: Temperature, Humidity, Rainfall

Author: [Your Name]
Version: 2.0.0
Date: January 2026
"""

import warnings
warnings.filterwarnings('ignore')

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict
from datetime import datetime, timedelta
import pickle
import pandas as pd
import numpy as np
import requests
import sys
import os

# Weather API configuration
WEATHER_API_KEY = "15b3a9327f4039ff4e81dffc63a1f616"  # Your OpenWeatherMap API key
WEATHER_API_URL = "http://api.openweathermap.org/data/2.5/weather"

# ============================================
# SOIL FERTILITY ASSESSOR CLASS
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
            'cotton': {'N': (100, 150), 'P': (40, 70), 'K': (40, 70), 'ph': (6.5, 8.0)},
            'chickpea': {'N': (15, 30), 'P': (40, 70), 'K': (30, 50), 'ph': (6.0, 7.5)},
            'lentil': {'N': (15, 25), 'P': (40, 60), 'K': (20, 40), 'ph': (6.0, 7.0)},
            'mango': {'N': (100, 180), 'P': (50, 90), 'K': (80, 150), 'ph': (5.5, 7.5)},
            'banana': {'N': (150, 250), 'P': (60, 100), 'K': (200, 300), 'ph': (6.0, 7.5)},
            'apple': {'N': (100, 180), 'P': (50, 90), 'K': (80, 150), 'ph': (5.5, 7.0)},
            'grapes': {'N': (80, 150), 'P': (50, 90), 'K': (80, 150), 'ph': (6.0, 7.0)},
            'coffee': {'N': (100, 150), 'P': (60, 100), 'K': (80, 120), 'ph': (6.0, 7.0)},
            'jute': {'N': (60, 100), 'P': (30, 50), 'K': (30, 50), 'ph': (6.0, 7.0)},
            'coconut': {'N': (100, 200), 'P': (50, 100), 'K': (150, 250), 'ph': (5.5, 7.0)},
            'papaya': {'N': (80, 150), 'P': (60, 100), 'K': (80, 150), 'ph': (6.0, 7.0)},
            'orange': {'N': (100, 180), 'P': (50, 90), 'K': (80, 150), 'ph': (6.0, 7.5)},
            'watermelon': {'N': (80, 150), 'P': (50, 90), 'K': (80, 150), 'ph': (6.0, 7.0)},
            'muskmelon': {'N': (80, 120), 'P': (50, 80), 'K': (80, 120), 'ph': (6.0, 7.0)},
            'pomegranate': {'N': (80, 150), 'P': (50, 90), 'K': (80, 150), 'ph': (6.5, 8.0)},
            'blackgram': {'N': (15, 25), 'P': (30, 50), 'K': (20, 40), 'ph': (6.0, 7.0)},
            'mungbean': {'N': (15, 25), 'P': (30, 50), 'K': (20, 40), 'ph': (6.0, 7.0)},
            'mothbeans': {'N': (15, 25), 'P': (25, 45), 'K': (20, 40), 'ph': (6.0, 7.5)},
            'pigeonpeas': {'N': (15, 30), 'P': (40, 60), 'K': (30, 50), 'ph': (6.0, 7.5)},
            'kidneybeans': {'N': (20, 40), 'P': (40, 70), 'K': (40, 70), 'ph': (6.0, 7.0)}
        }
        
        # General optimal ranges (when crop not specified)
        self.general_optimal = {
            'N': (80, 140), 'P': (40, 80), 'K': (40, 80), 'ph': (6.0, 7.0),
            'temperature': (20, 30), 'humidity': (60, 80), 'rainfall': (80, 150)
        }
    
    def calculate_parameter_score(self, value, optimal_range, param_name):
        """Calculate score (0-100) for a single parameter"""
        min_opt, max_opt = optimal_range
        
        if min_opt <= value <= max_opt:
            return 100.0
        elif value < min_opt:
            if param_name == 'ph':
                return max(0, 100 - (min_opt - value) * 15)
            else:
                return max(0, 100 - (min_opt - value) / min_opt * 100)
        else:
            if param_name == 'ph':
                return max(0, 100 - (value - max_opt) * 15)
            else:
                return max(0, 100 - (value - max_opt) / max_opt * 50)
    
    def assess_fertility(self, soil_data, crop=None):
        """Main fertility assessment function"""
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
        
        # Calculate scores
        scores = {}
        for param in ['N', 'P', 'K', 'ph', 'temperature', 'humidity', 'rainfall']:
            if param in soil_data and param in optimal:
                scores[param] = self.calculate_parameter_score(
                    soil_data[param], optimal[param], param
                )
        
        # Weighted average
        weights = {
            'N': 0.20, 'P': 0.20, 'K': 0.20, 'ph': 0.20,
            'temperature': 0.07, 'humidity': 0.07, 'rainfall': 0.06
        }
        overall_score = sum(scores[p] * weights[p] for p in scores)
        
        # Determine status
        if overall_score >= 80:
            status, emoji = "EXCELLENT", "🟢"
        elif overall_score >= 65:
            status, emoji = "GOOD", "🟡"
        elif overall_score >= 50:
            status, emoji = "MODERATE", "🟠"
        else:
            status, emoji = "POOR", "🔴"
        
        # Identify deficiencies
        deficiencies = {}
        for param, score in scores.items():
            if score < 70:
                current = soil_data[param]
                opt_range = optimal[param]
                if current < opt_range[0]:
                    deficiencies[param] = {
                        'type': 'LOW', 'current': current,
                        'optimal': opt_range, 'deficit': opt_range[0] - current,
                        'score': score
                    }
        
        # Generate recommendations
        recommendations = []
        
        if 'N' in deficiencies and deficiencies['N']['type'] == 'LOW':
            d = deficiencies['N']
            urea = (d['deficit'] / 0.46) * 1.2
            recommendations.append({
                'issue': 'Nitrogen Deficiency',
                'action': f"Apply {urea:.0f} kg/ha Urea",
                'cost': f"₹{urea*6:.0f}/ha",
                'timeline': '2-4 weeks'
            })
        
        if 'P' in deficiencies and deficiencies['P']['type'] == 'LOW':
            d = deficiencies['P']
            ssp = (d['deficit'] / 0.16) * 1.2
            recommendations.append({
                'issue': 'Phosphorus Deficiency',
                'action': f"Apply {ssp:.0f} kg/ha SSP",
                'cost': f"₹{ssp*8:.0f}/ha",
                'timeline': '4-6 weeks'
            })
        
        if 'K' in deficiencies and deficiencies['K']['type'] == 'LOW':
            d = deficiencies['K']
            mop = (d['deficit'] / 0.60) * 1.2
            recommendations.append({
                'issue': 'Potassium Deficiency',
                'action': f"Apply {mop:.0f} kg/ha MOP",
                'cost': f"₹{mop*15:.0f}/ha",
                'timeline': '3-5 weeks'
            })
        
        return {
            'fertility_score': round(overall_score, 1),
            'status': status,
            'status_emoji': emoji,
            'parameter_scores': scores,
            'deficiencies': deficiencies,
            'recommendations': recommendations,
            'optimal_ranges': optimal
        }

# ============================================
# FASTAPI APP INITIALIZATION
# ============================================

app = FastAPI(
    title="Smart Crop Recommendation API",
    description="""
    🌾 pH-Stratified Crop Recommendation System with AI-powered Analysis
    
    **Key Features:**
    - 99.91% prediction accuracy across 33 crops
    - Automatic weather data integration (OpenWeatherMap)
    - pH-stratified domain-specific models
    - Real-time soil fertility assessment (0-100 scoring)
    - Cost-effective improvement recommendations
    - IoT device integration ready
    
    **Innovation:**
    Farmers only need to provide N, P, K, pH, and Location.
    Weather data (temperature, humidity, rainfall) is automatically fetched!
    
    **Research:** MTech Thesis 2026
    **Accuracy:** 99.91% (better than state-of-art 98.27%)
    """,
    version="2.0.0",
    contact={
        "name": "Research Team",
        "email": "contact@example.com"
    }
)

# CORS middleware (allows frontend to connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# LOAD ML MODELS ON STARTUP
# ============================================

print("="*70)
print("🔄 Loading AI models...")
print("="*70)

try:
    # Get base directory
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Try multiple locations for models
    models_locations = [
        os.path.join(BASE_DIR, 'models', 'ph_specific_models.pkl'),      # api/models/
        os.path.join(BASE_DIR, '..', 'models', 'ph_specific_models.pkl') # ../models/
    ]
    
    scalers_locations = [
        os.path.join(BASE_DIR, 'data', 'feature_scalers.pkl'),           # api/data/
        os.path.join(BASE_DIR, '..', 'data', 'feature_scalers.pkl')      # ../data/
    ]
    
    # Find models
    MODELS_PATH = None
    for path in models_locations:
        if os.path.exists(path):
            MODELS_PATH = path
            break
    
    if not MODELS_PATH:
        raise FileNotFoundError(
            f"Models not found! Searched:\n" + "\n".join(models_locations)
        )
    
    # Find scalers
    SCALERS_PATH = None
    for path in scalers_locations:
        if os.path.exists(path):
            SCALERS_PATH = path
            break
    
    if not SCALERS_PATH:
        raise FileNotFoundError(
            f"Scalers not found! Searched:\n" + "\n".join(scalers_locations)
        )
    
    # Load files
    with open(MODELS_PATH, 'rb') as f:
        models_dict = pickle.load(f)
    
    with open(SCALERS_PATH, 'rb') as f:
        scalers = pickle.load(f)
    
    # Initialize fertility assessor
    fertility_assessor = SoilFertilityAssessor()
    
    # Success message
    print("✅ Models loaded successfully!")
    print(f"   📁 Models: {os.path.dirname(MODELS_PATH)}")
    print(f"   📊 Zones: {list(models_dict.keys())}")
    print(f"   🌾 Total crops: {sum(len(models_dict[z].get('crops', [])) for z in models_dict)}")
    print("="*70)
    
except Exception as e:
    print(f"❌ Error loading models: {e}")
    print("="*70)
    models_dict = None
    scalers = None
    fertility_assessor = None

# ============================================
# WEATHER DATA FETCHING
# ============================================

def fetch_weather_data(location: str, lat: float = None, lon: float = None):
    """
    Fetch real-time weather data from OpenWeatherMap API
    
    Args:
        location: City name (e.g., "Lucknow, IN")
        lat, lon: Optional GPS coordinates
    
    Returns:
        dict with temperature, humidity, rainfall
    """
    try:
        # Build API request
        if lat and lon:
            params = {
                'lat': lat,
                'lon': lon,
                'appid': WEATHER_API_KEY,
                'units': 'metric'
            }
        else:
            params = {
                'q': location,
                'appid': WEATHER_API_KEY,
                'units': 'metric'
            }
        
        # Call OpenWeatherMap API
        response = requests.get(WEATHER_API_URL, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        # Extract weather data
        weather_data = {
            'temperature': data['main']['temp'],
            'humidity': data['main']['humidity'],
            'rainfall': data.get('rain', {}).get('1h', 0) * 24 * 30,  # Estimate monthly
            'location': data['name'],
            'country': data['sys']['country'],
            'source': 'OpenWeatherMap API'
        }
        
        print(f"✅ Weather: {weather_data['location']} - "
              f"{weather_data['temperature']:.1f}°C, "
              f"{weather_data['humidity']:.0f}% humidity")
        
        return weather_data
    
    except Exception as e:
        print(f"⚠️ Weather API error: {e}")
        # Fallback to default values
        return {
            'temperature': 25.0,
            'humidity': 75.0,
            'rainfall': 120.0,
            'location': location if location else 'Unknown',
            'source': 'default (weather API unavailable)'
        }

# ============================================
# PYDANTIC MODELS (Request/Response Validation)
# ============================================

class SimpleSoilInput(BaseModel):
    """Simplified input - only soil sensor data needed"""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "N": 90,
                "P": 42,
                "K": 43,
                "ph": 6.5,
                "location": "Lucknow, IN"
            }
        }
    )
    
    N: float = Field(..., ge=0, le=200, description="Nitrogen (kg/ha)")
    P: float = Field(..., ge=0, le=200, description="Phosphorus (kg/ha)")
    K: float = Field(..., ge=0, le=200, description="Potassium (kg/ha)")
    ph: float = Field(..., ge=3, le=10, description="Soil pH")
    
    # Location (one of these required)
    location: Optional[str] = Field(None, description="City name (e.g., 'Lucknow, IN')")
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)
    
    # Optional: override auto-fetched weather
    temperature: Optional[float] = Field(None, ge=0, le=50)
    humidity: Optional[float] = Field(None, ge=0, le=100)
    rainfall: Optional[float] = Field(None, ge=0, le=300)

# ============================================
# HELPER FUNCTIONS
# ============================================

def get_ph_zone(ph_value: float) -> str:
    """Determine pH zone for model selection"""
    if ph_value < 5.5:
        return 'acidic'
    elif ph_value <= 7.5:
        return 'neutral'
    else:
        return 'alkaline'

def predict_crop(soil_data: dict) -> tuple:
    """Predict crop from soil data"""
    zone = get_ph_zone(soil_data['ph'])
    model_info = models_dict[zone]
    model = model_info['model']
    scaler = scalers[zone]
    
    # Prepare input
    X = pd.DataFrame([soil_data])
    X = X[model_info['feature_names']]
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Predict
    prediction = model.predict(X_scaled)[0]
    
    # Get probabilities
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_scaled)[0]
        confidence = proba.max()
        
        if hasattr(model, 'classes_'):
            crops = model.classes_
            crop_probas = dict(zip(crops, proba))
            crop_probas = dict(sorted(crop_probas.items(), 
                                    key=lambda x: x[1], reverse=True))
        else:
            crop_probas = {}
    else:
        confidence = 1.0
        crop_probas = {}
    
    return prediction, confidence, crop_probas, zone

# ============================================
# API ENDPOINTS
# ============================================

@app.get("/")
async def root():
    """API information and health check"""
    return {
        "name": "Smart Crop Recommendation API",
        "version": "2.0.0",
        "status": "online",
        "models_loaded": models_dict is not None,
        "innovation": "Auto-fetches weather data - farmers only provide N, P, K, pH, location",
        "accuracy": "99.91%",
        "supported_crops": 33,
        "endpoints": {
            "prediction": "/predict_smart",
            "weather_test": "/weather/{location}",
            "health": "/health",
            "documentation": "/docs"
        }
    }

@app.post("/predict_smart")
async def predict_smart(soil: SimpleSoilInput):
    """
    **Smart Prediction with Automatic Weather Fetching**
    
    **What You Provide:**
    - N, P, K (from soil test or NPK sensor)
    - pH (from pH sensor)
    - Location (city name or GPS coordinates)
    
    **What We Auto-Fetch:**
    - Temperature (from weather API)
    - Humidity (from weather API)
    - Rainfall (from weather API)
    
    **Returns:** Complete analysis with crop recommendation, fertility score, 
    and improvement recommendations
    
    **Example Request:**
```json
    {
      "N": 90,
      "P": 42,
      "K": 43,
      "ph": 6.5,
      "location": "Lucknow, IN"
    }
```
    """
    if models_dict is None:
        raise HTTPException(
            status_code=503, 
            detail="Models not loaded. Please contact administrator."
        )
    
    try:
        # Step 1: Get weather data (auto-fetch or use provided)
        if soil.temperature is None or soil.humidity is None or soil.rainfall is None:
            # Auto-fetch weather
            if soil.location:
                weather = fetch_weather_data(soil.location)
            elif soil.latitude and soil.longitude:
                weather = fetch_weather_data(None, soil.latitude, soil.longitude)
            else:
                raise HTTPException(
                    status_code=400, 
                    detail="Provide either 'location' or 'latitude'+'longitude'"
                )
            
            # Use auto-fetched values
            temperature = weather['temperature']
            humidity = weather['humidity']
            rainfall = weather['rainfall']
            weather_source = weather['source']
        else:
            # Use provided values
            temperature = soil.temperature
            humidity = soil.humidity
            rainfall = soil.rainfall
            weather_source = "user-provided"
        
        # Step 2: Build complete soil data
        complete_soil_data = {
            'N': soil.N,
            'P': soil.P,
            'K': soil.K,
            'ph': soil.ph,
            'temperature': temperature,
            'humidity': humidity,
            'rainfall': rainfall
        }
        
        # Step 3: Predict crop
        crop, confidence, all_probas, zone = predict_crop(complete_soil_data)
        
        # Step 4: Assess fertility
        fertility = fertility_assessor.assess_fertility(complete_soil_data, crop=crop)
        
        # Step 5: Calculate costs
        total_cost = sum([
            float(rec['cost'].replace('₹', '').replace('/ha', '').replace(',', '')) 
            for rec in fertility['recommendations'] if 'cost' in rec
        ])
        
        # Step 6: Return comprehensive analysis
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
                "needs_improvement": fertility['fertility_score'] < 65
            },
            "recommendations": fertility['recommendations'],
            "cost_estimate": {
                "total": f"₹{total_cost:.0f}/ha" if total_cost > 0 else "₹0/ha",
                "timeline": "2-6 weeks for NPK, 3-6 months for pH"
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/weather/{location}")
async def get_weather(location: str):
    """
    **Test Weather API**
    
    Test the weather data fetching for a specific location.
    Useful for debugging and verification.
    """
    weather = fetch_weather_data(location)
    return weather

@app.get("/health")
async def health():
    """System health check"""
    return {
        "status": "healthy" if models_dict else "degraded",
        "models_loaded": models_dict is not None,
        "scalers_loaded": scalers is not None,
        "fertility_assessor": fertility_assessor is not None,
        "weather_api": "OpenWeatherMap",
        "timestamp": datetime.now().isoformat()
    }

# ============================================
# RUN SERVER
# ============================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*70)
    print("🚀 SMART CROP RECOMMENDATION API")
    print("="*70)
    print("\n💡 Innovation: Auto-fetches weather data!")
    print("📍 Farmers only provide: N, P, K, pH, Location")
    print("🌤️  API fetches: Temperature, Humidity, Rainfall")
    print("\n📖 API Documentation: http://localhost:8000/docs")
    print("🧪 Interactive Testing: http://localhost:8000/docs")
    print("\n" + "="*70 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)