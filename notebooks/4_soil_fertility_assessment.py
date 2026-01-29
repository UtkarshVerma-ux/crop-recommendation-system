"""
MODULE 4: Soil Fertility Assessment System
Calculates soil fertility score (0-100) and provides improvement recommendations
"""

import pandas as pd
import numpy as np
import pickle
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to import shared modules
sys.path.insert(0, os.path.dirname(__file__))

print("="*70)
print("🌱 SOIL FERTILITY ASSESSMENT SYSTEM")
print("="*70)

# ============================================
# Define SoilFertilityAssessor class
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
            'N': (80, 140),
            'P': (40, 80),
            'K': (40, 80),
            'ph': (6.0, 7.0),
            'temperature': (20, 30),
            'humidity': (60, 80),
            'rainfall': (80, 150)
        }
    
    def calculate_parameter_score(self, value, optimal_range, param_name):
        """
        Calculate score (0-100) for a single parameter
        100 = optimal, decreases as value moves away from optimal range
        """
        min_opt, max_opt = optimal_range
        
        if min_opt <= value <= max_opt:
            # Within optimal range - perfect score
            return 100.0
        elif value < min_opt:
            # Below optimal - deficiency
            if param_name == 'ph':
                deficiency = min_opt - value
                score = max(0, 100 - (deficiency * 15))  # pH penalty
            else:
                # Nutrient deficiency - more severe
                deficiency_pct = (min_opt - value) / min_opt * 100
                score = max(0, 100 - deficiency_pct)
        else:
            # Above optimal - excess (less problematic than deficiency)
            if param_name == 'ph':
                excess = value - max_opt
                score = max(0, 100 - (excess * 15))
            else:
                # Nutrient excess - half penalty
                excess_pct = (value - max_opt) / max_opt * 100
                score = max(0, 100 - (excess_pct * 0.5))
        
        return score
    
    def assess_fertility(self, soil_data, crop=None):
        """
        Main fertility assessment function
        
        Args:
            soil_data: dict with keys: N, P, K, ph, temperature, humidity, rainfall
            crop: optional - assess for specific crop
        
        Returns:
            dict with fertility_score, status, deficiencies, recommendations
        """
        # Choose optimal ranges
        if crop and crop.lower() in self.optimal_ranges:
            optimal = self.optimal_ranges[crop.lower()]
            # Add general ranges for climate parameters
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
                    soil_data[param], 
                    optimal[param],
                    param
                )
        
        # Weighted average (NPK and pH are most important)
        weights = {
            'N': 0.20,
            'P': 0.20,
            'K': 0.20,
            'ph': 0.20,
            'temperature': 0.07,
            'humidity': 0.07,
            'rainfall': 0.06
        }
        
        overall_score = sum(scores[p] * weights[p] for p in scores)
        
        # Determine status
        if overall_score >= 80:
            status = "EXCELLENT"
            status_color = "🟢"
        elif overall_score >= 65:
            status = "GOOD"
            status_color = "🟡"
        elif overall_score >= 50:
            status = "MODERATE"
            status_color = "🟠"
        else:
            status = "POOR"
            status_color = "🔴"
        
        # Identify deficiencies (score < 70)
        deficiencies = {}
        for param, score in scores.items():
            if score < 70:
                current_value = soil_data[param]
                optimal_range = optimal[param]
                
                if current_value < optimal_range[0]:
                    deficiency_type = "LOW"
                    deficit = optimal_range[0] - current_value
                else:
                    deficiency_type = "HIGH"
                    deficit = current_value - optimal_range[1]
                
                deficiencies[param] = {
                    'type': deficiency_type,
                    'current': current_value,
                    'optimal': optimal_range,
                    'deficit': deficit,
                    'score': score
                }
        
        # Generate recommendations
        recommendations = self.generate_recommendations(deficiencies, soil_data)
        
        return {
            'fertility_score': round(overall_score, 1),
            'status': status,
            'status_emoji': status_color,
            'parameter_scores': scores,
            'deficiencies': deficiencies,
            'recommendations': recommendations,
            'optimal_ranges': optimal
        }
    
    def generate_recommendations(self, deficiencies, soil_data):
        """
        Generate specific improvement recommendations with costs
        """
        recommendations = []
        
        # NPK recommendations
        if 'N' in deficiencies:
            d = deficiencies['N']
            if d['type'] == 'LOW':
                # Calculate urea needed (urea is 46% N)
                n_needed = d['deficit']
                urea_needed = (n_needed / 0.46) * 1.2  # 20% extra for efficiency
                cost = urea_needed * 6  # ₹6/kg
                recommendations.append({
                    'issue': 'Nitrogen Deficiency',
                    'severity': 'HIGH' if d['score'] < 50 else 'MEDIUM',
                    'current': f"{d['current']:.1f} kg/ha",
                    'target': f"{d['optimal'][0]:.1f} kg/ha",
                    'action': f"Apply {urea_needed:.0f} kg/ha of Urea fertilizer",
                    'benefit': "Increases leaf growth, chlorophyll content, and overall crop yield",
                    'cost_estimate': f"₹{cost:.0f}/ha",
                    'timeline': '2-4 weeks to see improvement',
                    'priority': 'HIGH'
                })
        
        if 'P' in deficiencies:
            d = deficiencies['P']
            if d['type'] == 'LOW':
                # Calculate SSP needed (Single Super Phosphate - 16% P2O5)
                p_needed = d['deficit']
                ssp_needed = (p_needed / 0.16) * 1.2
                cost = ssp_needed * 8  # ₹8/kg
                recommendations.append({
                    'issue': 'Phosphorus Deficiency',
                    'severity': 'HIGH' if d['score'] < 50 else 'MEDIUM',
                    'current': f"{d['current']:.1f} kg/ha",
                    'target': f"{d['optimal'][0]:.1f} kg/ha",
                    'action': f"Apply {ssp_needed:.0f} kg/ha of SSP (Single Super Phosphate)",
                    'benefit': "Improves root development, flowering, and fruit formation",
                    'cost_estimate': f"₹{cost:.0f}/ha",
                    'timeline': '4-6 weeks',
                    'priority': 'HIGH'
                })
        
        if 'K' in deficiencies:
            d = deficiencies['K']
            if d['type'] == 'LOW':
                # Calculate MOP needed (Muriate of Potash - 60% K2O)
                k_needed = d['deficit']
                mop_needed = (k_needed / 0.60) * 1.2
                cost = mop_needed * 15  # ₹15/kg
                recommendations.append({
                    'issue': 'Potassium Deficiency',
                    'severity': 'HIGH' if d['score'] < 50 else 'MEDIUM',
                    'current': f"{d['current']:.1f} kg/ha",
                    'target': f"{d['optimal'][0]:.1f} kg/ha",
                    'action': f"Apply {mop_needed:.0f} kg/ha of MOP (Muriate of Potash)",
                    'benefit': "Enhances disease resistance, fruit quality, and drought tolerance",
                    'cost_estimate': f"₹{cost:.0f}/ha",
                    'timeline': '3-5 weeks',
                    'priority': 'MEDIUM'
                })
        
        # pH recommendations
        if 'ph' in deficiencies:
            d = deficiencies['ph']
            if d['type'] == 'LOW':
                # Acidic soil - need lime
                ph_increase_needed = d['optimal'][0] - d['current']
                lime_needed = ph_increase_needed * 1000  # Rough estimate: 1 ton/ha per pH unit
                cost = lime_needed * 5  # ₹5/kg
                recommendations.append({
                    'issue': 'Acidic Soil (Low pH)',
                    'severity': 'HIGH',
                    'current': f"pH {d['current']:.2f}",
                    'target': f"pH {d['optimal'][0]:.2f}",
                    'action': f"Apply {lime_needed:.0f} kg/ha of Agricultural Lime (CaCO₃)",
                    'benefit': "Increases pH, improves nutrient availability (especially P and Mo)",
                    'cost_estimate': f"₹{cost:.0f}/ha",
                    'timeline': '3-6 months (gradual effect)',
                    'priority': 'HIGH'
                })
            else:
                # Alkaline soil - need gypsum/sulfur
                ph_decrease_needed = d['current'] - d['optimal'][1]
                gypsum_needed = ph_decrease_needed * 2000  # Rough estimate
                cost = gypsum_needed * 3  # ₹3/kg
                recommendations.append({
                    'issue': 'Alkaline Soil (High pH)',
                    'severity': 'HIGH',
                    'current': f"pH {d['current']:.2f}",
                    'target': f"pH {d['optimal'][1]:.2f}",
                    'action': f"Apply {gypsum_needed:.0f} kg/ha of Gypsum (CaSO₄) or Elemental Sulfur",
                    'benefit': "Decreases pH, improves soil structure, supplies Ca and S",
                    'cost_estimate': f"₹{cost:.0f}/ha",
                    'timeline': '3-6 months (gradual effect)',
                    'priority': 'HIGH'
                })
        
        # Climate recommendations
        if 'rainfall' in deficiencies:
            d = deficiencies['rainfall']
            if d['type'] == 'LOW':
                recommendations.append({
                    'issue': 'Insufficient Rainfall',
                    'severity': 'MEDIUM',
                    'current': f"{d['current']:.0f} mm",
                    'target': f"{d['optimal'][0]:.0f} mm",
                    'action': "Install drip irrigation system or schedule supplemental irrigation",
                    'benefit': "Ensures consistent water supply, reduces water stress",
                    'cost_estimate': "₹15,000-30,000/ha (one-time for drip system)",
                    'timeline': 'Immediate upon installation',
                    'priority': 'MEDIUM'
                })
        
        # Sort by priority
        priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        recommendations.sort(key=lambda x: priority_order[x['priority']])
        
        return recommendations

# ============================================
# TEST THE SYSTEM
# ============================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 TESTING SOIL FERTILITY ASSESSMENT")
    print("="*70)
    
    assessor = SoilFertilityAssessor()
    
    # Test Case 1: Poor soil
    print("\n📊 TEST CASE 1: Poor Acidic Soil")
    print("-"*70)
    
    poor_soil = {
        'N': 35,      # Very low
        'P': 20,      # Low
        'K': 15,      # Very low
        'ph': 4.2,    # Too acidic
        'temperature': 25,
        'humidity': 70,
        'rainfall': 90
    }
    
    result = assessor.assess_fertility(poor_soil, crop='rice')
    
    print(f"\n{result['status_emoji']} Fertility Status: {result['status']}")
    print(f"📈 Overall Score: {result['fertility_score']}/100")
    
    print(f"\n🔍 Parameter Scores:")
    for param, score in result['parameter_scores'].items():
        emoji = "✅" if score >= 80 else "⚠️" if score >= 60 else "❌"
        print(f"   {emoji} {param}: {score:.1f}/100")
    
    print(f"\n⚠️ Issues Identified: {len(result['deficiencies'])}")
    for param, info in result['deficiencies'].items():
        print(f"   - {param.upper()}: {info['type']} (score: {info['score']:.1f}/100)")
    
    print(f"\n💡 Recommendations: {len(result['recommendations'])}")
    for i, rec in enumerate(result['recommendations'], 1):
        print(f"\n   {i}. {rec['issue']} [{rec['severity']}]")
        print(f"      Current: {rec['current']}")
        print(f"      Action: {rec['action']}")
        print(f"      Cost: {rec['cost_estimate']}")
        print(f"      Timeline: {rec['timeline']}")
    
    # Test Case 2: Good soil
    print("\n\n📊 TEST CASE 2: Good Neutral Soil")
    print("-"*70)
    
    good_soil = {
        'N': 110,
        'P': 60,
        'K': 50,
        'ph': 6.5,
        'temperature': 25,
        'humidity': 75,
        'rainfall': 120
    }
    
    result = assessor.assess_fertility(good_soil, crop='wheat')
    
    print(f"\n{result['status_emoji']} Fertility Status: {result['status']}")
    print(f"📈 Overall Score: {result['fertility_score']}/100")
    
    if not result['deficiencies']:
        print(f"\n✅ Excellent soil! No deficiencies detected.")
        print(f"   All parameters are within optimal range for wheat cultivation.")
    else:
        print(f"\n⚠️ Minor Issues: {len(result['deficiencies'])}")
    
    # Save the assessor WITH the class definition
    print("\n" + "="*70)
    print("💾 SAVING FERTILITY ASSESSOR")
    print("="*70)
    
    # Save both the instance and class definition info
    save_data = {
        'assessor': assessor,
        'class_module': __name__,
        'optimal_ranges': assessor.optimal_ranges,
        'general_optimal': assessor.general_optimal
    }
    
    with open('../models/fertility_assessor.pkl', 'wb') as f:
        pickle.dump(save_data, f)
    
    print("✅ Saved: fertility_assessor.pkl")
    
    print("\n" + "="*70)
    print("✅ SOIL FERTILITY ASSESSMENT SYSTEM READY!")
    print("="*70)
    
    print("\n📋 System Capabilities:")
    print("   ✅ Calculate fertility score (0-100)")
    print("   ✅ Identify nutrient deficiencies")
    print("   ✅ Generate specific fertilizer recommendations")
    print("   ✅ Estimate costs (₹/ha)")
    print("   ✅ Provide timelines for improvement")
    print("   ✅ Support 23 different crops")
    
    print("\n🎯 Next step: Run 5_dice_soil_improvement.py")