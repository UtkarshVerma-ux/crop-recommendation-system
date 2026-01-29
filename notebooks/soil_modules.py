"""
Shared modules for soil analysis system
Contains: SoilFertilityAssessor class
"""

import pandas as pd
import numpy as np

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
        
        # General optimal ranges
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
        """Calculate score (0-100) for a single parameter"""
        min_opt, max_opt = optimal_range
        
        if min_opt <= value <= max_opt:
            return 100.0
        elif value < min_opt:
            if param_name == 'ph':
                deficiency = min_opt - value
                score = max(0, 100 - (deficiency * 15))
            else:
                deficiency_pct = (min_opt - value) / min_opt * 100
                score = max(0, 100 - deficiency_pct)
        else:
            if param_name == 'ph':
                excess = value - max_opt
                score = max(0, 100 - (excess * 15))
            else:
                excess_pct = (value - max_opt) / max_opt * 100
                score = max(0, 100 - (excess_pct * 0.5))
        
        return score
    
    def assess_fertility(self, soil_data, crop=None):
        """Main fertility assessment function"""
        # Choose optimal ranges
        if crop and crop.lower() in self.optimal_ranges:
            optimal = self.optimal_ranges[crop.lower()]
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
                    soil_data[param], 
                    optimal[param],
                    param
                )
        
        # Weighted average
        weights = {
            'N': 0.20, 'P': 0.20, 'K': 0.20, 'ph': 0.20,
            'temperature': 0.07, 'humidity': 0.07, 'rainfall': 0.06
        }
        
        overall_score = sum(scores[p] * weights[p] for p in scores)
        
        # Determine status
        if overall_score >= 80:
            status, status_color = "EXCELLENT", "🟢"
        elif overall_score >= 65:
            status, status_color = "GOOD", "🟡"
        elif overall_score >= 50:
            status, status_color = "MODERATE", "🟠"
        else:
            status, status_color = "POOR", "🔴"
        
        # Identify deficiencies
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
        """Generate specific improvement recommendations"""
        recommendations = []
        
        if 'N' in deficiencies:
            d = deficiencies['N']
            if d['type'] == 'LOW':
                n_needed = d['deficit']
                urea_needed = (n_needed / 0.46) * 1.2
                cost = urea_needed * 6
                recommendations.append({
                    'issue': 'Nitrogen Deficiency',
                    'severity': 'HIGH' if d['score'] < 50 else 'MEDIUM',
                    'current': f"{d['current']:.1f} kg/ha",
                    'target': f"{d['optimal'][0]:.1f} kg/ha",
                    'action': f"Apply {urea_needed:.0f} kg/ha of Urea fertilizer",
                    'benefit': "Increases leaf growth, chlorophyll content, and crop yield",
                    'cost_estimate': f"₹{cost:.0f}/ha",
                    'timeline': '2-4 weeks',
                    'priority': 'HIGH'
                })
        
        if 'P' in deficiencies:
            d = deficiencies['P']
            if d['type'] == 'LOW':
                p_needed = d['deficit']
                ssp_needed = (p_needed / 0.16) * 1.2
                cost = ssp_needed * 8
                recommendations.append({
                    'issue': 'Phosphorus Deficiency',
                    'severity': 'HIGH' if d['score'] < 50 else 'MEDIUM',
                    'current': f"{d['current']:.1f} kg/ha",
                    'target': f"{d['optimal'][0]:.1f} kg/ha",
                    'action': f"Apply {ssp_needed:.0f} kg/ha of SSP",
                    'benefit': "Improves root development and flowering",
                    'cost_estimate': f"₹{cost:.0f}/ha",
                    'timeline': '4-6 weeks',
                    'priority': 'HIGH'
                })
        
        if 'K' in deficiencies:
            d = deficiencies['K']
            if d['type'] == 'LOW':
                k_needed = d['deficit']
                mop_needed = (k_needed / 0.60) * 1.2
                cost = mop_needed * 15
                recommendations.append({
                    'issue': 'Potassium Deficiency',
                    'severity': 'HIGH' if d['score'] < 50 else 'MEDIUM',
                    'current': f"{d['current']:.1f} kg/ha",
                    'target': f"{d['optimal'][0]:.1f} kg/ha",
                    'action': f"Apply {mop_needed:.0f} kg/ha of MOP",
                    'benefit': "Enhances disease resistance and fruit quality",
                    'cost_estimate': f"₹{cost:.0f}/ha",
                    'timeline': '3-5 weeks',
                    'priority': 'MEDIUM'
                })
        
        if 'ph' in deficiencies:
            d = deficiencies['ph']
            if d['type'] == 'LOW':
                ph_increase = d['optimal'][0] - d['current']
                lime_needed = ph_increase * 1000
                cost = lime_needed * 5
                recommendations.append({
                    'issue': 'Acidic Soil',
                    'severity': 'HIGH',
                    'current': f"pH {d['current']:.2f}",
                    'target': f"pH {d['optimal'][0]:.2f}",
                    'action': f"Apply {lime_needed:.0f} kg/ha of Agricultural Lime",
                    'benefit': "Increases pH and nutrient availability",
                    'cost_estimate': f"₹{cost:.0f}/ha",
                    'timeline': '3-6 months',
                    'priority': 'HIGH'
                })
            else:
                ph_decrease = d['current'] - d['optimal'][1]
                gypsum_needed = ph_decrease * 2000
                cost = gypsum_needed * 3
                recommendations.append({
                    'issue': 'Alkaline Soil',
                    'severity': 'HIGH',
                    'current': f"pH {d['current']:.2f}",
                    'target': f"pH {d['optimal'][1]:.2f}",
                    'action': f"Apply {gypsum_needed:.0f} kg/ha of Gypsum",
                    'benefit': "Decreases pH and improves soil structure",
                    'cost_estimate': f"₹{cost:.0f}/ha",
                    'timeline': '3-6 months',
                    'priority': 'HIGH'
                })
        
        # Sort by priority
        priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        recommendations.sort(key=lambda x: priority_order[x['priority']])
        
        return recommendations