"""
Farmer-Friendly Crop Recommendation Web Interface
==================================================
Beautiful Streamlit interface for pH-stratified crop prediction
with automatic weather data fetching
"""

import os
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json

# Page configuration
st.set_page_config(
    page_title="Smart Crop Recommendation System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(120deg, #a8e063 0%, #56ab2f 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #4CAF50;
    }
    .recommendation-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 1.2rem;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        border: none;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_URL = os.getenv("API_URL", "https://crop-recommendation-system-30tu.onrender.com")  # Change to your deployed URL

# ============================================
# HELPER FUNCTIONS
# ============================================

def get_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def predict_crop(soil_data):
    """Call API for prediction"""
    try:
        response = requests.post(
            f"{API_URL}/predict_smart",
            json=soil_data,
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None

def create_fertility_gauge(score, status):
    """Create a beautiful gauge chart for fertility score"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Soil Fertility Score", 'font': {'size': 24}},
        delta={'reference': 65, 'increasing': {'color': "green"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkgreen"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#ffcdd2'},
                {'range': [50, 65], 'color': '#fff9c4'},
                {'range': [65, 80], 'color': '#c8e6c9'},
                {'range': [80, 100], 'color': '#a5d6a7'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 65
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

def create_confidence_chart(alternatives):
    """Create bar chart for crop alternatives"""
    crops = [alt['crop'] for alt in alternatives]
    confidences = [alt['confidence'] * 100 for alt in alternatives]
    
    fig = go.Figure([go.Bar(
        x=confidences,
        y=crops,
        orientation='h',
        marker=dict(
            color=confidences,
            colorscale='Greens',
            showscale=True,
            colorbar=dict(title="Confidence %")
        ),
        text=[f"{c:.1f}%" for c in confidences],
        textposition='auto',
    )])
    
    fig.update_layout(
        title="Crop Suitability Analysis",
        xaxis_title="Confidence (%)",
        yaxis_title="Crop",
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)"
    )
    
    return fig

def create_npk_radar(N, P, K):
    """Create radar chart for NPK levels"""
    categories = ['Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)']
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=[N, P, K],
        theta=categories,
        fill='toself',
        name='Current Levels',
        line_color='#1976D2',
        fillcolor='rgba(25, 118, 210, 0.3)'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=[100, 60, 50],
        theta=categories,
        fill='toself',
        name='Optimal Range',
        line_color='#4CAF50',
        fillcolor='rgba(76, 175, 80, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 200])
        ),
        showlegend=True,
        height=400,
        title="NPK Analysis"
    )
    
    return fig

# ============================================
# MAIN APP
# ============================================

def main():
    # Header
    st.markdown(
        '<div class="main-header">🌾 Smart Crop Recommendation System 🌾</div>',
        unsafe_allow_html=True
    )
    
    # Check API status
    if not get_api_health():
        st.error("⚠️ API is not running! Please start the API server first.")
        st.code("cd api && python main.py", language="bash")
        st.stop()
    
    st.success("✅ API Connected - System Ready!")
    
    # Sidebar - System Info
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/wheat.png", width=80)
        st.title("📊 System Info")
        
        st.markdown("""
        ### 🎯 About
        **pH-Stratified ML System**
        - Accuracy: 99.91%
        - Crops Supported: 33
        - pH Zones: 3 (Acidic/Neutral/Alkaline)
        
        ### 🌟 Features
        - ✅ Auto weather fetching
        - ✅ Soil fertility analysis
        - ✅ Cost estimates
        - ✅ Improvement recommendations
        - ✅ IoT device ready
        
        ### 📖 How to Use
        1. Enter soil test results (N, P, K, pH)
        2. Select your location
        3. Click "Get Recommendation"
        4. View results & recommendations
        
        ### 👨‍🌾 For Farmers
        Only need soil test for N, P, K, pH.
        Weather data fetched automatically!
        """)
        
        st.markdown("---")
        st.caption("Developed for MTech Research | 2026")
    
    # Main Content
    tab1, tab2, tab3 = st.tabs(["🌾 Get Recommendation", "📚 About System", "🔧 API Integration"])
    
    # ============================================
    # TAB 1: MAIN PREDICTION INTERFACE
    # ============================================
    with tab1:
        st.header("Enter Your Soil Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🧪 Soil Test Results")
            st.markdown("*Get these from soil testing lab or NPK sensor*")
            
            N = st.number_input(
                "Nitrogen (N) - kg/ha",
                min_value=0.0,
                max_value=200.0,
                value=90.0,
                step=5.0,
                help="Nitrogen content in soil (typical range: 40-150)"
            )
            
            P = st.number_input(
                "Phosphorus (P) - kg/ha",
                min_value=0.0,
                max_value=200.0,
                value=42.0,
                step=5.0,
                help="Phosphorus content in soil (typical range: 20-100)"
            )
            
            K = st.number_input(
                "Potassium (K) - kg/ha",
                min_value=0.0,
                max_value=200.0,
                value=43.0,
                step=5.0,
                help="Potassium content in soil (typical range: 20-100)"
            )
            
            ph = st.slider(
                "Soil pH",
                min_value=3.0,
                max_value=10.0,
                value=6.5,
                step=0.1,
                help="Soil acidity/alkalinity (neutral: 6-7.5)"
            )
            
            # pH indicator
            if ph < 5.5:
                st.warning(f"⚠️ Acidic Soil (pH {ph:.1f})")
            elif ph <= 7.5:
                st.success(f"✅ Neutral Soil (pH {ph:.1f})")
            else:
                st.info(f"ℹ️ Alkaline Soil (pH {ph:.1f})")
        
        with col2:
            st.subheader("📍 Location")
            st.markdown("*Weather data will be fetched automatically*")
            
            location_method = st.radio(
                "Select location method:",
                ["City Name", "GPS Coordinates"]
            )
            
            if location_method == "City Name":
                location = st.text_input(
                    "Enter your city",
                    value="Lucknow, IN",
                    help="Format: City, Country Code (e.g., Mumbai, IN)"
                )
                latitude = None
                longitude = None
            else:
                latitude = st.number_input(
                    "Latitude",
                    min_value=-90.0,
                    max_value=90.0,
                    value=26.8467,
                    step=0.0001,
                    format="%.4f"
                )
                longitude = st.number_input(
                    "Longitude",
                    min_value=-180.0,
                    max_value=180.0,
                    value=80.9462,
                    step=0.0001,
                    format="%.4f"
                )
                location = None
            
            st.markdown("---")
            
            # Optional: Manual weather override
            with st.expander("⚙️ Advanced: Override Weather Data"):
                st.caption("Leave empty to auto-fetch from weather API")
                
                manual_temp = st.number_input(
                    "Temperature (°C)",
                    min_value=0.0,
                    max_value=50.0,
                    value=None,
                    help="Leave empty for auto-fetch"
                )
                
                manual_humidity = st.number_input(
                    "Humidity (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=None,
                    help="Leave empty for auto-fetch"
                )
                
                manual_rainfall = st.number_input(
                    "Rainfall (mm/month)",
                    min_value=0.0,
                    max_value=300.0,
                    value=None,
                    help="Leave empty for auto-fetch"
                )
        
        # Predict Button
        st.markdown("---")
        
        if st.button("🚀 Get Crop Recommendation", use_container_width=True):
            
            # Prepare API request
            soil_data = {
                "N": N,
                "P": P,
                "K": K,
                "ph": ph
            }
            
            if location:
                soil_data["location"] = location
            else:
                soil_data["latitude"] = latitude
                soil_data["longitude"] = longitude
            
            if manual_temp:
                soil_data["temperature"] = manual_temp
            if manual_humidity:
                soil_data["humidity"] = manual_humidity
            if manual_rainfall:
                soil_data["rainfall"] = manual_rainfall
            
            # Show loading spinner
            with st.spinner("🔍 Analyzing soil conditions..."):
                result = predict_crop(soil_data)
            
            if result:
                # ============================================
                # DISPLAY RESULTS
                # ============================================
                
                st.success("✅ Analysis Complete!")
                
                # Main Recommendation Box
                prediction = result['prediction']
                
                st.markdown(f"""
                <div class="recommendation-box">
                    <h1 style='text-align: center; margin-bottom: 1rem;'>
                        🌾 Recommended Crop: {prediction['recommended_crop'].upper()} 🌾
                    </h1>
                    <h3 style='text-align: center;'>
                        Confidence: {prediction['confidence']*100:.1f}%
                    </h3>
                    <p style='text-align: center; font-size: 1.1rem;'>
                        pH Zone: {prediction['ph_zone'].capitalize()} Soil
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Three column layout for key metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "🌾 Primary Crop",
                        prediction['recommended_crop'].title(),
                        f"{prediction['confidence']*100:.1f}% confidence"
                    )
                
                with col2:
                    fertility_score = result['fertility']['score']
                    fertility_status = result['fertility']['status']
                    delta_color = "normal" if fertility_score >= 65 else "inverse"
                    
                    st.metric(
                        "🌱 Soil Fertility",
                        f"{fertility_score:.1f}/100",
                        fertility_status,
                        delta_color=delta_color
                    )
                
                with col3:
                    total_cost = result['cost_estimate']['total']
                    st.metric(
                        "💰 Improvement Cost",
                        total_cost if result['fertility']['needs_improvement'] else "No cost",
                        "Investment needed" if result['fertility']['needs_improvement'] else "Soil is good!"
                    )
                
                st.markdown("---")
                
                # Detailed Analysis Section
                tab_detail1, tab_detail2, tab_detail3, tab_detail4 = st.tabs([
                    "📊 Fertility Analysis",
                    "🌾 Alternative Crops",
                    "📈 NPK Visualization",
                    "💡 Recommendations"
                ])
                
                with tab_detail1:
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        # Fertility Gauge
                        gauge_fig = create_fertility_gauge(
                            result['fertility']['score'],
                            result['fertility']['status']
                        )
                        st.plotly_chart(gauge_fig, use_container_width=True)
                    
                    with col2:
                        st.subheader("Soil Analysis Summary")
                        
                        soil_analysis = result['soil_analysis']
                        
                        st.markdown("**📍 Location Data:**")
                        weather = soil_analysis['weather_data']
                        st.write(f"- Temperature: {weather['temperature']:.1f}°C")
                        st.write(f"- Humidity: {weather['humidity']:.1f}%")
                        st.write(f"- Rainfall: {weather['rainfall']:.1f} mm/month")
                        st.caption(f"*Source: {weather['source']}*")
                        
                        st.markdown("**🧪 Soil Parameters:**")
                        provided = soil_analysis['provided_by_user']
                        st.write(f"- Nitrogen (N): {provided['N']} kg/ha")
                        st.write(f"- Phosphorus (P): {provided['P']} kg/ha")
                        st.write(f"- Potassium (K): {provided['K']} kg/ha")
                        st.write(f"- pH: {provided['ph']}")
                        
                        if result['fertility']['needs_improvement']:
                            st.warning(f"⚠️ {result['fertility']['deficiencies']} deficiencies detected")
                        else:
                            st.success("✅ Soil is in excellent condition!")
                
                with tab_detail2:
                    st.subheader("Alternative Crop Options")
                    
                    alternatives = prediction['alternatives']
                    
                    # Confidence chart
                    conf_fig = create_confidence_chart(alternatives)
                    st.plotly_chart(conf_fig, use_container_width=True)
                    
                    st.markdown("**📋 Detailed Comparison:**")
                    
                    alt_df = pd.DataFrame(alternatives)
                    alt_df['confidence'] = alt_df['confidence'].apply(lambda x: f"{x*100:.2f}%")
                    alt_df['rank'] = range(1, len(alt_df) + 1)
                    alt_df = alt_df[['rank', 'crop', 'confidence']]
                    alt_df.columns = ['Rank', 'Crop', 'Suitability']
                    
                    st.dataframe(alt_df, use_container_width=True, hide_index=True)
                    
                    st.info("💡 All listed crops are viable options. Choose based on market prices and your preference!")
                
                with tab_detail3:
                    st.subheader("NPK Analysis")
                    
                    # Radar chart
                    radar_fig = create_npk_radar(N, P, K)
                    st.plotly_chart(radar_fig, use_container_width=True)
                    
                    # NPK Status
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        n_status = "✅ Good" if 80 <= N <= 140 else "⚠️ Needs attention"
                        st.metric("Nitrogen Status", n_status, f"{N} kg/ha")
                    
                    with col2:
                        p_status = "✅ Good" if 40 <= P <= 80 else "⚠️ Needs attention"
                        st.metric("Phosphorus Status", p_status, f"{P} kg/ha")
                    
                    with col3:
                        k_status = "✅ Good" if 40 <= K <= 80 else "⚠️ Needs attention"
                        st.metric("Potassium Status", k_status, f"{K} kg/ha")
                
                with tab_detail4:
                    st.subheader("💡 Improvement Recommendations")
                    
                    recommendations = result['recommendations']
                    
                    if recommendations:
                        for i, rec in enumerate(recommendations, 1):
                            with st.expander(f"#{i} - {rec['issue']}", expanded=True):
                                col1, col2 = st.columns([2, 1])
                                
                                with col1:
                                    st.markdown(f"**Action:** {rec['action']}")
                                    st.markdown(f"**Timeline:** {rec['timeline']}")
                                
                                with col2:
                                    st.markdown(f"**Cost:** {rec['cost']}")
                        
                        st.markdown("---")
                        st.info(f"**Total Estimated Cost:** {result['cost_estimate']['total']}")
                        st.caption(f"**Timeline:** {result['cost_estimate']['timeline']}")
                    else:
                        st.success("🎉 No improvements needed! Your soil is in excellent condition.")
                
                # Download Report Button
                st.markdown("---")
                
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col2:
                    report_data = {
                        "timestamp": datetime.now().isoformat(),
                        "soil_parameters": soil_data,
                        "prediction": result
                    }
                    
                    st.download_button(
                        label="📥 Download Full Report (JSON)",
                        data=json.dumps(report_data, indent=2),
                        file_name=f"crop_recommendation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
    
    # ============================================
    # TAB 2: ABOUT SYSTEM
    # ============================================
    with tab2:
        st.header("📚 About the System")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Research Innovation")
            st.markdown("""
            This system employs **pH-Stratified Machine Learning** for crop 
            recommendation, achieving **99.91% accuracy** across 33 crops.
            
            **Key Features:**
            - 🧬 pH-zone specific models (Acidic/Neutral/Alkaline)
            - 🎯 99.91% prediction accuracy
            - 🌍 Automatic weather data integration
            - 💰 Cost-effective recommendations
            - 📊 Soil fertility assessment
            - 🔄 IoT device integration ready
            
            **Supported Crops:** 33 major Indian crops including:
            Rice, Wheat, Maize, Cotton, Sugarcane, Jute, Coffee, Tea, 
            and many more fruits and vegetables.
            """)
            
            st.subheader("🏆 Performance Metrics")
            
            metrics_df = pd.DataFrame({
                'Zone': ['Acidic', 'Neutral', 'Alkaline', 'Overall'],
                'Accuracy': ['100.00%', '99.74%', '100.00%', '99.91%'],
                'Samples': [160, 1898, 315, 2373]
            })
            
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.subheader("💡 How It Works")
            
            st.markdown("""
            ### Step-by-Step Process:
            
            1. **Input Collection**
               - You provide: N, P, K, pH, Location
               - System fetches: Temperature, Humidity, Rainfall
            
            2. **pH Zone Detection**
               - Acidic: pH < 5.5
               - Neutral: pH 5.5-7.5
               - Alkaline: pH > 7.5
            
            3. **ML Prediction**
               - Zone-specific model analyzes soil
               - Predicts best crop with confidence score
               - Generates alternative options
            
            4. **Fertility Assessment**
               - Calculates 0-100 fertility score
               - Identifies deficiencies (N, P, K, pH)
               - Generates improvement recommendations
            
            5. **Cost Estimation**
               - Calculates fertilizer requirements
               - Estimates total improvement cost
               - Provides timeline for results
            """)
            
            st.subheader("📖 Research Paper")
            st.info("""
            **Title:** pH-Stratified Ensemble Learning for Intelligent 
            Crop Recommendation with Integrated Soil Fertility Assessment
            
            **Published:** 2026 (Pending)
            
            **Institution:** Gautam Buddha University, India
            
            **Degree:** MTech Thesis
            """)
    
    # ============================================
    # TAB 3: API INTEGRATION
    # ============================================
    with tab3:
        st.header("🔧 API Integration Guide")
        
        st.markdown("""
        This system provides a REST API for integration with:
        - 🌾 IoT devices and sensors
        - 📱 Mobile applications
        - 💻 Web applications
        - 🤖 Agricultural robots
        """)
        
        st.subheader("📡 API Endpoint")
        st.code(f"{API_URL}/predict_smart", language="text")
        
        st.subheader("📝 Example Request (Python)")
        st.code("""
import requests

# Prepare soil data
soil_data = {
    "N": 90,
    "P": 42,
    "K": 43,
    "ph": 6.5,
    "location": "Lucknow, IN"
}

# Call API
response = requests.post(
    "http://localhost:8000/predict_smart",
    json=soil_data
)

# Get results
result = response.json()
print(f"Recommended Crop: {result['prediction']['recommended_crop']}")
print(f"Confidence: {result['prediction']['confidence']*100:.1f}%")
print(f"Fertility Score: {result['fertility']['score']}/100")
        """, language="python")
        
        st.subheader("📝 Example Request (Arduino/ESP32)")
        st.code("""
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

void sendToAPI() {
    HTTPClient http;
    http.begin("http://your-server.com:8000/predict_smart");
    http.addHeader("Content-Type", "application/json");
    
    // Read sensors
    float N = readNPK_N();
    float P = readNPK_P();
    float K = readNPK_K();
    float ph = readPH();
    
    // Create JSON
    StaticJsonDocument<512> doc;
    doc["N"] = N;
    doc["P"] = P;
    doc["K"] = K;
    doc["ph"] = ph;
    doc["location"] = "Lucknow, IN";
    
    String json;
    serializeJson(doc, json);
    
    // Send request
    int httpCode = http.POST(json);
    
    if (httpCode == 200) {
        String response = http.getString();
        // Parse and display results
    }
    
    http.end();
}
        """, language="cpp")
        
        st.subheader("📊 API Response Format")
        st.json({
            "prediction": {
                "recommended_crop": "rice",
                "confidence": 0.985,
                "ph_zone": "neutral",
                "alternatives": [
                    {"crop": "rice", "confidence": 0.985},
                    {"crop": "wheat", "confidence": 0.012}
                ]
            },
            "fertility": {
                "score": 85.3,
                "status": "EXCELLENT",
                "deficiencies": 0,
                "needs_improvement": False
            },
            "recommendations": [],
            "cost_estimate": {
                "total": "₹0/ha",
                "timeline": "No improvements needed"
            }
        })
        
        st.markdown("---")
        
        st.info("""
        **📖 Full API Documentation:** Visit `http://localhost:8000/docs` 
        for interactive Swagger documentation with all endpoints and examples.
        """)

if __name__ == "__main__":
    main()