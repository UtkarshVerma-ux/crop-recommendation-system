"""
MODULE 5: DiCE-Based Soil Improvement Advisor
Generates "what-if" counterfactual scenarios
"""

import pandas as pd
import numpy as np
import pickle
import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))

from importlib import import_module

print("="*70)
print("🔄 DiCE COUNTERFACTUAL SOIL IMPROVEMENT ADVISOR")
print("="*70)

# Import SoilFertilityAssessor
try:
    spec = import_module('4_soil_fertility_assessment')
    SoilFertilityAssessor = spec.SoilFertilityAssessor
    print("✅ Imported SoilFertilityAssessor from module 4")
except:
    print("⚠️ Defining SoilFertilityAssessor inline...")
    
    class SoilFertilityAssessor:
        def __init__(self):
            self.optimal_ranges = {
                'rice': {'N': (80, 120), 'P': (40, 80), 'K': (30, 60), 'ph': (5.5, 7.0)},
                'wheat': {'N': (100, 150), 'P': (50, 80), 'K': (30, 60), 'ph': (6.0, 7.5)},
                'maize': {'N': (100, 150), 'P': (60, 90), 'K': (40, 80), 'ph': (5.8, 7.0)},
                'mango': {'N': (100, 180), 'P': (50, 90), 'K': (80, 150), 'ph': (5.5, 7.5)},
                'mothbeans': {'N': (15, 25), 'P': (25, 45), 'K': (20, 40), 'ph': (6.0, 7.5)},
                'pigeonpeas': {'N': (15, 30), 'P': (40, 60), 'K': (30, 50), 'ph': (6.0, 7.5)},
            }
            self.general_optimal = {
                'N': (80, 140), 'P': (40, 80), 'K': (40, 80), 'ph': (6.0, 7.0),
                'temperature': (20, 30), 'humidity': (60, 80), 'rainfall': (80, 150)
            }
        
        def calculate_parameter_score(self, value, optimal_range, param_name):
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
            if crop and crop.lower() in self.optimal_ranges:
                optimal = self.optimal_ranges[crop.lower()].copy()
                optimal.update({
                    'temperature': self.general_optimal['temperature'],
                    'humidity': self.general_optimal['humidity'],
                    'rainfall': self.general_optimal['rainfall']
                })
            else:
                optimal = self.general_optimal
            
            scores = {}
            for param in ['N', 'P', 'K', 'ph', 'temperature', 'humidity', 'rainfall']:
                if param in soil_data and param in optimal:
                    scores[param] = self.calculate_parameter_score(
                        soil_data[param], optimal[param], param
                    )
            
            weights = {'N': 0.20, 'P': 0.20, 'K': 0.20, 'ph': 0.20,
                      'temperature': 0.07, 'humidity': 0.07, 'rainfall': 0.06}
            overall_score = sum(scores[p] * weights[p] for p in scores)
            
            if overall_score >= 80:
                status, emoji = "EXCELLENT", "🟢"
            elif overall_score >= 65:
                status, emoji = "GOOD", "🟡"
            elif overall_score >= 50:
                status, emoji = "MODERATE", "🟠"
            else:
                status, emoji = "POOR", "🔴"
            
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
                    else:
                        deficiencies[param] = {
                            'type': 'HIGH', 'current': current,
                            'optimal': opt_range, 'deficit': current - opt_range[1],
                            'score': score
                        }
            
            return {
                'fertility_score': round(overall_score, 1),
                'status': status, 'status_emoji': emoji,
                'parameter_scores': scores, 'deficiencies': deficiencies,
                'optimal_ranges': optimal
            }

# Load models
print("\n📂 Loading models and scalers...")
with open('../models/ph_specific_models.pkl', 'rb') as f:
    models_dict = pickle.load(f)

with open('../data/feature_scalers.pkl', 'rb') as f:
    scalers = pickle.load(f)

fertility_assessor = SoilFertilityAssessor()
print("✅ Loaded successfully!")

class SoilImprovementAdvisor:
    def __init__(self, models_dict, scalers, fertility_assessor):
        self.models = models_dict
        self.scalers = scalers
        self.assessor = fertility_assessor
        self.optimal_ranges = fertility_assessor.optimal_ranges
    
    def get_model_for_ph(self, ph_value):
        if ph_value < 5.5:
            return 'acidic'
        elif ph_value <= 7.5:
            return 'neutral'
        else:
            return 'alkaline'
    
    def predict_crop(self, soil_data, zone=None):
        """FIXED: Predict crop handling label encoder properly"""
        if zone is None:
            zone = self.get_model_for_ph(soil_data['ph'])
        
        model_info = self.models[zone]
        model = model_info['model']
        scaler = self.scalers[zone]
        
        # Prepare input
        X = pd.DataFrame([soil_data])
        X = X[model_info['feature_names']]
        X_scaled = scaler.transform(X)
        
        # Simple prediction - model classes are already in correct format
        prediction = model.predict(X_scaled)[0]
        
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
        
        return prediction, confidence, crop_probas
    
    def generate_improvement_scenarios(self, current_soil, desired_crop=None, n_scenarios=3):
        print(f"\n{'='*70}")
        print("🔄 GENERATING IMPROVEMENT SCENARIOS")
        print(f"{'='*70}")
        
        current_zone = self.get_model_for_ph(current_soil['ph'])
        current_crop, current_conf, current_probas = self.predict_crop(current_soil)
        current_fertility = self.assessor.assess_fertility(current_soil, crop=current_crop)
        
        print(f"\n📊 Current Soil Analysis:")
        print(f"   pH Zone: {current_zone.capitalize()}")
        print(f"   Fertility: {current_fertility['fertility_score']:.1f}/100 ({current_fertility['status']})")
        print(f"   Predicted Crop: {current_crop} ({current_conf*100:.1f}% confidence)")
        print(f"   Issues: {len(current_fertility['deficiencies'])} deficiencies" if current_fertility['deficiencies'] else "   ✅ No deficiencies!")
        
        scenarios = []
        
        # Scenario 1: NPK
        print(f"\n🔧 Scenario 1: NPK Optimization...")
        s1 = self._gen_npk(current_soil, current_fertility, current_crop)
        if s1: scenarios.append(s1)
        
        # Scenario 2: pH
        if 'ph' in current_fertility['deficiencies']:
            print(f"🔧 Scenario 2: pH Correction...")
            s2 = self._gen_ph(current_soil, current_fertility, current_crop)
            if s2: scenarios.append(s2)
        
        # Scenario 3: Comprehensive
        print(f"🔧 Scenario 3: Comprehensive...")
        s3 = self._gen_comprehensive(current_soil, current_fertility, current_crop)
        if s3: scenarios.append(s3)
        
        for s in scenarios:
            s['improvement_score'] = (
                (s['new_fertility']['fertility_score'] - current_fertility['fertility_score']) * 0.6 +
                (s['new_confidence'] - current_conf) * 100 * 0.4
            )
        
        scenarios.sort(key=lambda x: x['improvement_score'], reverse=True)
        
        return {
            'current_soil': current_soil,
            'current_crop': current_crop,
            'current_confidence': current_conf,
            'current_fertility': current_fertility,
            'current_zone': current_zone,
            'scenarios': scenarios[:n_scenarios]
        }
    
    def _gen_npk(self, soil, fert, crop):
        mod = soil.copy()
        changes = {}
        cost = 0
        
        opt = self.optimal_ranges.get(crop.lower(), self.assessor.general_optimal)
        
        for n in ['N', 'P', 'K']:
            if n in fert['deficiencies']:
                target = (opt[n][0] + opt[n][1]) / 2
                changes[n] = {'from': soil[n], 'to': target, 'change': target - soil[n]}
                mod[n] = target
                
                if n == 'N': cost += abs(target - soil[n]) / 0.46 * 1.2 * 6
                elif n == 'P': cost += abs(target - soil[n]) / 0.16 * 1.2 * 8
                elif n == 'K': cost += abs(target - soil[n]) / 0.60 * 1.2 * 15
        
        if not changes: return None
        
        new_crop, new_conf, _ = self.predict_crop(mod)
        new_fert = self.assessor.assess_fertility(mod, crop=new_crop)
        
        actions = []
        if 'N' in changes: actions.append(f"1. Apply {abs(changes['N']['change']) / 0.46 * 1.2:.0f} kg/ha Urea")
        if 'P' in changes: actions.append(f"2. Apply {abs(changes['P']['change']) / 0.16 * 1.2:.0f} kg/ha SSP")
        if 'K' in changes: actions.append(f"3. Apply {abs(changes['K']['change']) / 0.60 * 1.2:.0f} kg/ha MOP")
        actions.extend([f"{len(actions)+1}. Water immediately", f"{len(actions)+2}. Retest after 4-6 weeks"])
        
        return {
            'name': 'NPK Optimization',
            'description': 'Adjust N, P, K to optimal levels',
            'changes': changes, 'modified_soil': mod,
            'new_crop': new_crop, 'new_confidence': new_conf, 'new_fertility': new_fert,
            'cost_estimate': f"₹{cost:.0f}/ha", 'timeline': '2-6 weeks',
            'feasibility': 'HIGH', 'actions': actions
        }
    
    def _gen_ph(self, soil, fert, crop):
        if 'ph' not in fert['deficiencies']: return None
        
        mod = soil.copy()
        opt = self.optimal_ranges.get(crop.lower(), self.assessor.general_optimal)
        target = (opt['ph'][0] + opt['ph'][1]) / 2
        
        changes = {'ph': {'from': soil['ph'], 'to': target, 'change': target - soil['ph']}}
        mod['ph'] = target
        
        if target > soil['ph']:
            amt = abs(target - soil['ph']) * 1000
            cost = amt * 5
            amend = "Lime"
        else:
            amt = abs(target - soil['ph']) * 2000
            cost = amt * 3
            amend = "Gypsum"
        
        new_crop, new_conf, _ = self.predict_crop(mod)
        new_fert = self.assessor.assess_fertility(mod, crop=new_crop)
        
        return {
            'name': 'pH Correction',
            'description': f'Adjust pH {soil["ph"]:.2f} → {target:.2f} with {amend}',
            'changes': changes, 'modified_soil': mod,
            'new_crop': new_crop, 'new_confidence': new_conf, 'new_fertility': new_fert,
            'cost_estimate': f"₹{cost:.0f}/ha", 'timeline': '3-6 months',
            'feasibility': 'MEDIUM',
            'actions': [f"1. Apply {amt:.0f} kg/ha {amend}", "2. Water", "3. Retest after 3 months"]
        }
    
    def _gen_comprehensive(self, soil, fert, crop):
        mod = soil.copy()
        changes = {}
        cost = 0
        
        opt = self.optimal_ranges.get(crop.lower(), self.assessor.general_optimal)
        
        for p in ['N', 'P', 'K', 'ph']:
            if p in fert['deficiencies']:
                target = (opt[p][0] + opt[p][1]) / 2
                changes[p] = {'from': soil[p], 'to': target, 'change': target - soil[p]}
                mod[p] = target
                
                if p == 'N': cost += abs(target - soil[p]) / 0.46 * 1.2 * 6
                elif p == 'P': cost += abs(target - soil[p]) / 0.16 * 1.2 * 8
                elif p == 'K': cost += abs(target - soil[p]) / 0.60 * 1.2 * 15
                elif p == 'ph':
                    if target > soil[p]: cost += abs(target - soil[p]) * 1000 * 5
                    else: cost += abs(target - soil[p]) * 2000 * 3
        
        if not changes: return None
        
        new_crop, new_conf, _ = self.predict_crop(mod)
        new_fert = self.assessor.assess_fertility(mod, crop=new_crop)
        
        return {
            'name': 'Comprehensive Improvement',
            'description': 'Fix all deficiencies',
            'changes': changes, 'modified_soil': mod,
            'new_crop': new_crop, 'new_confidence': new_conf, 'new_fertility': new_fert,
            'cost_estimate': f"₹{cost:.0f}/ha", 'timeline': '3-6 months',
            'feasibility': 'MEDIUM',
            'actions': ["1. Apply all amendments", "2. Water", "3. Retest after 4-6 weeks"]
        }

# TEST
print("\n" + "="*70)
print("🧪 TESTING DiCE ADVISOR")
print("="*70)

advisor = SoilImprovementAdvisor(models_dict, scalers, fertility_assessor)

test_soil = {
    'N': 40, 'P': 25, 'K': 20, 'ph': 5.2,
    'temperature': 25, 'humidity': 70, 'rainfall': 100
}

print(f"\n🌱 Input: N={test_soil['N']}, P={test_soil['P']}, K={test_soil['K']}, pH={test_soil['ph']}")

results = advisor.generate_improvement_scenarios(test_soil, n_scenarios=3)

print(f"\n{'='*70}")
print("📊 SCENARIOS GENERATED")
print(f"{'='*70}")

for i, s in enumerate(results['scenarios'], 1):
    print(f"\n{'='*70}")
    print(f"SCENARIO {i}: {s['name']}")
    print(f"{'='*70}")
    print(f"📝 {s['description']}")
    print(f"💰 Cost: {s['cost_estimate']}")
    print(f"⏱️  Timeline: {s['timeline']}")
    print(f"\n🔄 Changes:")
    for p, c in s['changes'].items():
        print(f"   {p}: {c['from']:.1f} → {c['to']:.1f} (Δ{c['change']:+.1f})")
    print(f"\n📈 Results:")
    print(f"   Crop: {results['current_crop']} → {s['new_crop']}")
    print(f"   Confidence: {results['current_confidence']*100:.1f}% → {s['new_confidence']*100:.1f}%")
    print(f"   Fertility: {results['current_fertility']['fertility_score']:.1f} → {s['new_fertility']['fertility_score']:.1f}")
    print(f"\n📋 Actions:")
    for a in s['actions']:
        print(f"   {a}")

with open('../models/improvement_advisor.pkl', 'wb') as f:
    pickle.dump(advisor, f)

print("\n✅ DiCE ADVISOR READY!")
print("\n🎯 Next: Build API or write paper!")