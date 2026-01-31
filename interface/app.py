"""
Farmer-Friendly Crop Recommendation Web Interface with Full XAI
================================================================
Includes SHAP, DiCE, and LIME visualizations
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
    page_title="Smart Crop Recommendation System with XAI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
    .xai-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
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
API_URL = os.getenv("API_URL", "http://localhost:8000")

# ============================================
# HELPER FUNCTIONS
# ============================================

def get_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200, response.json()
    except:
        return False, None

def predict_crop(soil_data):
    """Call API for prediction"""
    try:
        response = requests.post(
            f"{API_URL}/predict_smart",
            json=soil_data,
            timeout=30  # Increased timeout for XAI processing
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None

def create_fertility_gauge(score, status):
    """Create gauge chart for fertility score"""
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

def create_shap_chart(shap_data):
    """Create SHAP feature importance bar chart"""
    if not shap_data or 'feature_importance' not in shap_data:
        return None
    
    importance = shap_data['feature_importance']
    
    if not importance:
        return None
    
    features = list(importance.keys())
    values = list(importance.values())
    
    # Create color scale based on importance
    colors = ['#1f77b4' if v > 15 else '#7fcdbb' if v > 10 else '#c7e9b4' for v in values]
    
    fig = go.Figure([go.Bar(
        x=values,
        y=features,
        orientation='h',
        marker=dict(color=colors),
        text=[f"{v:.1f}%" for v in values],
        textposition='auto',
    )])
    
    fig.update_layout(
        title={
            'text': "🔍 SHAP Feature Importance Analysis",
            'font': {'size': 18, 'color': '#1f77b4'}
        },
        xaxis_title="Importance (%)",
        yaxis_title="Features",
        height=400,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(240,240,240,0.5)"
    )
    
    return fig

def create_lime_chart(lime_data):
    """Create LIME feature importance bar chart"""
    if not lime_data or 'feature_importance' not in lime_data:
        return None
    
    importance = lime_data['feature_importance']
    
    if not importance:
        return None
    
    features = list(importance.keys())
    values = list(importance.values())
    
    colors = ['#ff7f0e' if v > 15 else '#ffbb78' if v > 10 else '#fdd0a2' for v in values]
    
    fig = go.Figure([go.Bar(
        x=values,
        y=features,
        orientation='h',
        marker=dict(color=colors),
        text=[f"{v:.1f}%" for v in values],
        textposition='auto',
    )])
    
    fig.update_layout(
        title={
            'text': "💡 LIME Local Explanation",
            'font': {'size': 18, 'color': '#ff7f0e'}
        },
        xaxis_title="Local Importance (%)",
        yaxis_title="Features",
        height=400,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(240,240,240,0.5)"
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

def display_dice_counterfactuals(dice_data):
    """Display DiCE counterfactual scenarios"""
    if not dice_data or 'counterfactuals' not in dice_data:
        st.info("DiCE counterfactuals not available")
        return
    
    counterfactuals = dice_data.get('counterfactuals', [])
    
    if not counterfactuals:
        st.info("No alternative scenarios found")
        return
    
    st.markdown("### 🎯 Alternative Scenarios (DiCE Counterfactuals)")
    
    for i, cf in enumerate(counterfactuals, 1):
        with st.expander(f"Scenario #{i}: {cf.get('suggested_crop', 'Unknown')}", expanded=(i==1)):
            
            changes = cf.get('changes_needed', {})
            
            if not changes:
                st.write("No changes needed for this alternative")
                continue
            
            st.markdown("**What would need to change:**")
            
            changes_df = []
            for param, change_info in changes.items():
                changes_df.append({
                    'Parameter': param.upper(),
                    'Current': f"{change_info['from']:.1f}",
                    'Suggested': f"{change_info['to']:.1f}",
                    'Change': f"{change_info['change']:+.1f}"
                })
            
            df = pd.DataFrame(changes_df)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            feasible = cf.get('feasible', True)
            if feasible:
                st.success("✅ This scenario is feasible with moderate changes")
            else:
                st.warning("⚠️ This scenario requires significant changes")

# ============================================
# MAIN APP
# ============================================

def main():
    # Header
    st.markdown(
        '<div class="main-header">🌾 Smart Crop Recommendation with AI Explainability 🔍</div>',
        unsafe_allow_html=True
    )
    
    # Check API status
    api_status, health_data = get_api_health()
    
    if not api_status:
        st.error("⚠️ API is not running! Please start the API server first.")
        st.code("cd api && python main.py", language="bash")
        st.stop()
    
    # Display XAI status
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.success("✅ API Connected - System Ready!")
    
    with col2:
        if health_data and health_data.get('xai_enabled'):
            xai_features = health_data.get('xai_features', {})
            xai_status = []
            if xai_features.get('shap'):
                xai_status.append("SHAP ✅")
            if xai_features.get('dice'):
                xai_status.append("DiCE ✅")
            if xai_features.get('lime'):
                xai_status.append("LIME ✅")
            
            st.info(f"XAI: {', '.join(xai_status)}")
        else:
            st.warning("XAI: Not enabled")
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/wheat.png", width=80)
        st.title("📊 System Info")
        
        st.markdown("""
        ### 🎯 About
        **pH-Stratified ML with XAI**
        - Accuracy: 99.91%
        - Crops: 33
        - pH Zones: 3
        
        ### 🔍 XAI Features
        - ✅ **SHAP**: Global feature importance
        - ✅ **DiCE**: Counterfactual scenarios
        - ✅ **LIME**: Local explanations
        
        ### 🌟 System Features
        - Auto weather fetching
        - Soil fertility analysis
        - Cost estimates
        - Improvement recommendations
        - Full explainability
        
        ### 📖 How to Use
        1. Enter soil parameters (N, P, K, pH)
        2. Select location
        3. Enable XAI options
        4. Get recommendations + explanations
        """)
        
        st.markdown("---")
        st.caption("MTech Research 2026 | Full XAI Integration")
    
    # Main Content
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌾 Get Recommendation",
        "🔍 About XAI",
        "📚 About System",
        "🔧 API Integration"
    ])
    
    # ============================================
    # TAB 1: MAIN PREDICTION INTERFACE
    # ============================================
    with tab1:
        st.header("Enter Your Soil Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🧪 Soil Test Results")
            
            N = st.number_input(
                "Nitrogen (N) - kg/ha",
                min_value=0.0,
                max_value=200.0,
                value=90.0,
                step=5.0,
                help="Nitrogen content in soil"
            )
            
            P = st.number_input(
                "Phosphorus (P) - kg/ha",
                min_value=0.0,
                max_value=200.0,
                value=42.0,
                step=5.0,
                help="Phosphorus content in soil"
            )
            
            K = st.number_input(
                "Potassium (K) - kg/ha",
                min_value=0.0,
                max_value=200.0,
                value=43.0,
                step=5.0,
                help="Potassium content in soil"
            )
            
            ph = st.slider(
                "Soil pH",
                min_value=3.0,
                max_value=10.0,
                value=6.5,
                step=0.1,
                help="Soil acidity/alkalinity"
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
            
            location_method = st.radio(
                "Select location method:",
                ["City Name", "GPS Coordinates"]
            )
            
            if location_method == "City Name":
                location = st.text_input(
                    "Enter your city",
                    value="Lucknow, IN",
                    help="Format: City, Country Code"
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
            
            # XAI Options
            st.subheader("🔍 Explainability Options")
            
            enable_xai = st.checkbox(
                "Enable XAI Explanations",
                value=True,
                help="Get SHAP, DiCE, and LIME explanations"
            )
            
            if enable_xai:
                xai_methods = st.multiselect(
                    "Select XAI methods:",
                    ['shap', 'dice', 'lime'],
                    default=['shap', 'dice'],
                    help="SHAP: Feature importance, DiCE: Alternative scenarios, LIME: Local explanations"
                )
            else:
                xai_methods = []
        
        # Predict Button
        st.markdown("---")
        
        if st.button("🚀 Get Crop Recommendation with XAI", use_container_width=True):
            
            # Prepare API request
            soil_data = {
                "N": N,
                "P": P,
                "K": K,
                "ph": ph,
                "enable_xai": enable_xai,
                "xai_methods": xai_methods if enable_xai else []
            }
            
            if location:
                soil_data["location"] = location
            else:
                soil_data["latitude"] = latitude
                soil_data["longitude"] = longitude
            
            # Show loading
            with st.spinner("🔍 Analyzing soil conditions and generating explanations..."):
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
                
                # ============================================
                # XAI EXPLANATIONS SECTION
                # ============================================
                
                if enable_xai and 'xai_explanations' in result:
                    xai_exp = result['xai_explanations']
                    
                    st.markdown("""
                    <div class="xai-box">
                        <h2 style='text-align: center; margin: 0;'>
                            🔍 AI Explainability (XAI) Results
                        </h2>
                        <p style='text-align: center; margin-top: 0.5rem;'>
                            Understanding WHY the AI recommended this crop
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Create tabs for different XAI methods
                    xai_tabs = []
                    if 'shap' in xai_exp and xai_exp['shap'].get('feature_importance'):
                        xai_tabs.append("📊 SHAP Analysis")
                    if 'dice' in xai_exp and xai_exp['dice'].get('counterfactuals'):
                        xai_tabs.append("🎯 DiCE Scenarios")
                    if 'lime' in xai_exp and xai_exp['lime'].get('feature_importance'):
                        xai_tabs.append("💡 LIME Explanation")
                    
                    if xai_tabs:
                        xai_tab_objects = st.tabs(xai_tabs)
                        
                        tab_idx = 0
                        
                        # SHAP Tab
                        if 'shap' in xai_exp and xai_exp['shap'].get('feature_importance'):
                            with xai_tab_objects[tab_idx]:
                                st.markdown("### 🔍 SHAP Feature Importance Analysis")
                                
                                shap_data = xai_exp['shap']
                                
                                col1, col2 = st.columns([2, 1])
                                
                                with col1:
                                    # SHAP chart
                                    shap_fig = create_shap_chart(shap_data)
                                    if shap_fig:
                                        st.plotly_chart(shap_fig, use_container_width=True)
                                
                                with col2:
                                    st.markdown("**Interpretation:**")
                                    st.write(shap_data.get('explanation_text', 'No explanation available'))
                                    
                                    st.markdown("**Top 3 Factors:**")
                                    top_features = shap_data.get('top_features', [])
                                    for i, (feat, imp) in enumerate(top_features, 1):
                                        st.write(f"{i}. **{feat}**: {imp:.1f}%")
                                    
                                    st.info("""
                                    **SHAP shows GLOBAL importance:**
                                    Features with highest percentages
                                    had the most impact on this
                                    prediction across the model.
                                    """)
                            
                            tab_idx += 1
                        
                        # DiCE Tab
                        if 'dice' in xai_exp and xai_exp['dice'].get('counterfactuals'):
                            with xai_tab_objects[tab_idx]:
                                st.markdown("### 🎯 DiCE Counterfactual Scenarios")
                                
                                st.info("""
                                **What are counterfactuals?**
                                
                                DiCE shows "what-if" scenarios: If you changed certain
                                soil parameters, what other crops would be suitable?
                                
                                This helps you understand decision boundaries and
                                plan soil improvements for alternative crops.
                                """)
                                
                                display_dice_counterfactuals(xai_exp['dice'])
                            
                            tab_idx += 1
                        
                        # LIME Tab
                        if 'lime' in xai_exp and xai_exp['lime'].get('feature_importance'):
                            with xai_tab_objects[tab_idx]:
                                st.markdown("### 💡 LIME Local Explanation")
                                
                                lime_data = xai_exp['lime']
                                
                                col1, col2 = st.columns([2, 1])
                                
                                with col1:
                                    # LIME chart
                                    lime_fig = create_lime_chart(lime_data)
                                    if lime_fig:
                                        st.plotly_chart(lime_fig, use_container_width=True)
                                
                                with col2:
                                    st.markdown("**About LIME:**")
                                    st.write(lime_data.get('explanation', 'Local explanation for this case'))
                                    
                                    st.info("""
                                    **LIME shows LOCAL importance:**
                                    
                                    Unlike SHAP (global), LIME explains
                                    this SPECIFIC prediction for YOUR
                                    exact soil conditions.
                                    
                                    Useful for understanding edge cases
                                    and unusual predictions.
                                    """)
                    
                    st.markdown("---")
                
                # ============================================
                # KEY METRICS
                # ============================================
                
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
                
                # ============================================
                # DETAILED ANALYSIS TABS
                # ============================================
                
                tab_detail1, tab_detail2, tab_detail3, tab_detail4 = st.tabs([
                    "📊 Fertility Analysis",
                    "🌾 Alternative Crops",
                    "📈 NPK Visualization",
                    "💡 Recommendations"
                ])
                
                with tab_detail1:
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
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
                        
                        # Parameter scores
                        if 'parameter_scores' in result['fertility']:
                            st.markdown("**📈 Parameter Scores:**")
                            param_scores = result['fertility']['parameter_scores']
                            for param, score in param_scores.items():
                                color = "🟢" if score >= 80 else "🟡" if score >= 65 else "🟠" if score >= 50 else "🔴"
                                st.write(f"{color} {param.upper()}: {score:.1f}/100")
                        
                        if result['fertility']['needs_improvement']:
                            st.warning(f"⚠️ {result['fertility']['deficiencies']} deficiencies detected")
                        else:
                            st.success("✅ Soil is in excellent condition!")
                
                with tab_detail2:
                    st.subheader("Alternative Crop Options")
                    
                    alternatives = prediction['alternatives']
                    
                    conf_fig = create_confidence_chart(alternatives)
                    st.plotly_chart(conf_fig, use_container_width=True)
                    
                    st.markdown("**📋 Detailed Comparison:**")
                    
                    alt_df = pd.DataFrame(alternatives)
                    alt_df['confidence'] = alt_df['confidence'].apply(lambda x: f"{x*100:.2f}%")
                    alt_df['rank'] = range(1, len(alt_df) + 1)
                    alt_df = alt_df[['rank', 'crop', 'confidence']]
                    alt_df.columns = ['Rank', 'Crop', 'Suitability']
                    
                    st.dataframe(alt_df, use_container_width=True, hide_index=True)
                
                with tab_detail3:
                    st.subheader("NPK Analysis")
                    
                    radar_fig = create_npk_radar(N, P, K)
                    st.plotly_chart(radar_fig, use_container_width=True)
                    
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
                        st.success("🎉 No improvements needed!")
                
                # Download Report
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
    # TAB 2: ABOUT XAI
    # ============================================
    with tab2:
        st.header("🔍 Understanding AI Explainability (XAI)")
        
        st.markdown("""
        Our system uses three cutting-edge XAI techniques to make AI decisions transparent:
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 📊 SHAP
            **SHapley Additive exPlanations**
            
            **What it does:**
            - Shows GLOBAL feature importance
            - Calculates contribution of each parameter
            - Based on game theory (Shapley values)
            
            **When to use:**
            - Understanding overall model behavior
            - Identifying key decision factors
            - Validating domain knowledge
            
            **Example:**
            "Rainfall contributed 35% to this recommendation,
            pH contributed 25%, temperature 20%..."
            
            **Benefits:**
            - ✅ Mathematically rigorous
            - ✅ Consistent across predictions
            - ✅ Scientifically validated
            """)
        
        with col2:
            st.markdown("""
            ### 🎯 DiCE
            **Diverse Counterfactual Explanations**
            
            **What it does:**
            - Generates "what-if" scenarios
            - Shows alternative paths
            - Suggests actionable changes
            
            **When to use:**
            - Planning soil improvements
            - Understanding decision boundaries
            - Exploring alternative crops
            
            **Example:**
            "If you increase Nitrogen from 80 to 100,
            and reduce pH from 7.0 to 6.5,
            the system would recommend Wheat instead."
            
            **Benefits:**
            - ✅ Actionable insights
            - ✅ Multiple scenarios
            - ✅ Helps planning
            """)
        
        with col3:
            st.markdown("""
            ### 💡 LIME
            **Local Interpretable Model-agnostic Explanations**
            
            **What it does:**
            - Explains THIS specific prediction
            - Local approximation of model
            - Instance-specific importance
            
            **When to use:**
            - Understanding unusual predictions
            - Edge cases
            - Your specific situation
            
            **Example:**
            "For YOUR exact soil conditions (N=90, P=42...),
            rainfall had 38% local importance,
            which is higher than the global 35%."
            
            **Benefits:**
            - ✅ Case-specific
            - ✅ Easy to understand
            - ✅ Model-agnostic
            """)
        
        st.markdown("---")
        
        st.markdown("""
        ### 🎓 Why Multiple XAI Methods?
        
        Different XAI techniques provide complementary insights:
        
        | Aspect | SHAP | DiCE | LIME |
        |--------|------|------|------|
        | **Scope** | Global | Counterfactual | Local |
        | **Purpose** | Feature importance | Alternative scenarios | Specific case |
        | **Best for** | Understanding model | Planning changes | Edge cases |
        | **Output** | Importance % | What-if changes | Local importance |
        
        **Together, they provide complete transparency! 🔍**
        """)
        
        st.info("""
        **💡 Pro Tip:** Use all three methods together for maximum insight:
        1. **SHAP** tells you what matters globally
        2. **DiCE** shows you how to change outcomes
        3. **LIME** explains your specific situation
        """)
    
    # ============================================
    # TAB 3: ABOUT SYSTEM (Keep existing)
    # ============================================
    with tab3:
        st.header("📚 About the System")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Research Innovation")
            st.markdown("""
            This system employs **pH-Stratified Machine Learning** with
            **Full XAI Integration** for transparent crop recommendation.
            
            **Key Features:**
            - 🧬 pH-zone specific models
            - 🎯 99.91% accuracy
            - 🔍 Complete explainability (SHAP + DiCE + LIME)
            - 🌍 Automatic weather integration
            - 💰 Cost-effective recommendations
            - 📊 Soil fertility assessment
            
            **Supported Crops:** 33 major Indian crops
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
               - Zone-specific model predicts crop
               - Confidence score calculated
               - Alternative options generated
            
            4. **XAI Explanations**
               - SHAP: Feature importance analysis
               - DiCE: Counterfactual scenarios
               - LIME: Local explanations
            
            5. **Fertility Assessment**
               - 0-100 fertility score
               - Deficiency identification
               - Improvement recommendations
            
            6. **Cost Estimation**
               - Fertilizer requirements
               - Total cost calculation
               - Timeline provided
            """)
    
    # ============================================
    # TAB 4: API INTEGRATION (Keep existing)
    # ============================================
    with tab4:
        st.header("🔧 API Integration Guide")
        
        st.markdown("""
        ### 📡 API Endpoint with XAI
        """)
        
        st.code(f"{API_URL}/predict_smart", language="text")
        
        st.markdown("### 📝 Example Request with XAI")
        
        st.code("""
import requests

soil_data = {
    "N": 90,
    "P": 42,
    "K": 43,
    "ph": 6.5,
    "location": "Lucknow, IN",
    "enable_xai": True,
    "xai_methods": ["shap", "dice", "lime"]
}

response = requests.post(
    "http://localhost:8000/predict_smart",
    json=soil_data
)

result = response.json()

# Access predictions
print(f"Crop: {result['prediction']['recommended_crop']}")
print(f"Confidence: {result['prediction']['confidence']}")

# Access XAI explanations
shap_importance = result['xai_explanations']['shap']['feature_importance']
print(f"Feature Importance: {shap_importance}")

dice_scenarios = result['xai_explanations']['dice']['counterfactuals']
print(f"Alternative Scenarios: {len(dice_scenarios)}")
        """, language="python")
        
        st.markdown("### 📊 Response with XAI")
        
        st.json({
            "prediction": {
                "recommended_crop": "rice",
                "confidence": 0.985
            },
            "xai_explanations": {
                "shap": {
                    "feature_importance": {
                        "rainfall": 35.2,
                        "ph": 25.1,
                        "temperature": 20.3
                    }
                },
                "dice": {
                    "counterfactuals": [
                        {
                            "suggested_crop": "wheat",
                            "changes_needed": {
                                "rainfall": {"from": 100, "to": 70}
                            }
                        }
                    ]
                },
                "lime": {
                    "feature_importance": {
                        "rainfall": 38.5,
                        "ph": 22.1
                    }
                }
            }
        })

if __name__ == "__main__":
    main()