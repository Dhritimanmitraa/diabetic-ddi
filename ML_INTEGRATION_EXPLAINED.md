# 🤖 ML INTEGRATION & SUGGESTIONS EXPLAINED

## 📋 TABLE OF CONTENTS
1. [How ML is Integrated](#how-ml-is-integrated)
2. [Where ML is Used](#where-ml-is-used)
3. [How Suggestions Work](#how-suggestions-work)
4. [ML vs Rules Comparison](#ml-vs-rules-comparison)
5. [Complete Flow Diagrams](#complete-flow-diagrams)

---

## 🔧 HOW ML IS INTEGRATED

### **1. ML Models Location**
- **Trained Models**: `backend/models/`
  - `random_forest_model.pkl` - Random Forest model
  - `xgboost_model.pkl` - XGBoost model
  - `lightgbm_model.pkl` - LightGBM model
  - `optimal_threshold.json` - Optimal classification threshold (0.2345)

### **2. ML Predictor Service**
**File**: `backend/app/ml/predictor.py`

**What it does**:
- Loads 3 trained models (Random Forest, XGBoost, LightGBM)
- Creates **ensemble predictions** (averages probabilities from all 3 models)
- Uses **optimal threshold** (0.2345) instead of default 0.5
- Feature engineering: Converts drug names/classes to numerical vectors

**Key Function**:
```python
def predict(drug1: Dict, drug2: Dict) -> PredictionResult:
    # 1. Extract features (hash encoding)
    features = extract_features_simple(drug1, drug2)
    
    # 2. Get predictions from each model
    rf_proba = random_forest.predict_proba(features)[0, 1]
    xgb_proba = xgboost.predict_proba(features)[0, 1]
    lgb_proba = lightgbm.predict_proba(features)[0, 1]
    
    # 3. Ensemble (average)
    ensemble_proba = (rf_proba + xgb_proba + lgb_proba) / 3
    
    # 4. Binary prediction using optimal threshold
    predicted_interaction = ensemble_proba >= 0.2345
    
    # 5. Predict severity based on probability
    if ensemble_proba >= 0.8: severity = "contraindicated"
    elif ensemble_proba >= 0.6: severity = "major"
    elif ensemble_proba >= 0.4: severity = "moderate"
    elif ensemble_proba >= 0.2: severity = "minor"
    else: severity = "none"
    
    return PredictionResult(...)
```

### **3. Training Data Source**
- **TWOSIDES Database**: 42M+ drug-drug interactions
- **Training Set**: Created by `build_training_set.py`
  - Positive samples: Known interactions
  - Negative samples: No interaction pairs
  - Split: 80% train, 10% val, 10% test

---

## 📍 WHERE ML IS USED

### **1. General Drug Interaction Check** (`/interactions/check`)

**File**: `backend/app/main.py` (lines 290-370)

**Flow**:
```
1. User checks interaction: Drug A + Drug B
2. Rules-based check (database lookup)
3. ML prediction (if models loaded)
4. Hybrid decision:
   - If rules say CONTRAINDICATED/MAJOR → Rules win (safety first!)
   - Otherwise → Use ML prediction
```

**Code**:
```python
# Rules-based result (database lookup)
result = await service.check_interaction(drug1_name, drug2_name)

# ML prediction
ml_result = predictor.predict(drug1_dict, drug2_dict)
ml_probability = ml_result.interaction_probability  # e.g., 0.15
ml_predicted = ml_result.predicted_interaction      # True/False

# Hybrid decision
if result.interaction.severity in ["contraindicated", "major"]:
    # Rules override ML (safety first!)
    final_decision = result.interaction
    decision_source = "rule_override"
else:
    # Use ML prediction
    final_decision = ml_predicted
    decision_source = "ml_primary"
```

**Example**:
- **Drugs**: Metformin + Lisinopril
- **Rules**: No interaction found in database
- **ML**: Probability = 0.12 → "Safe" (0.12 < 0.2345)
- **Final**: Safe (ML decision)

---

### **2. Diabetic Patient Drug Risk Check** (`/diabetic/risk-check`)

**File**: `backend/app/diabetic/service.py` (lines 232-304)

**Flow**:
```
1. User checks drug for diabetic patient
2. Rules-based assessment (PRIMARY) - eGFR, complications, etc.
3. ML prediction (SUPPLEMENTARY only)
4. Rules ALWAYS win - ML never overrides
```

**Code**:
```python
# RULES ARE PRIMARY - clinically validated logic
assessment = self.rules.assess_drug_risk(drug_name, patient_context, current_meds)

# ML is SUPPLEMENTARY only - for additional insights
ml_result = self.ml_predictor.predict(drug_name, patient_context)

# ML is supplementary - never overrides rules
# Only add ML info for transparency
response.ml_risk_level = ml_result.risk_level
response.ml_probability = ml_result.probability

# Log if ML disagrees with rules (for monitoring/retraining)
if ml_result.risk_level != assessment.risk_level:
    logger.warning("ML disagrees with rules!")
```

**Example**:
- **Patient**: DEMO003 (eGFR = 28, severe CKD)
- **Drug**: Verapamil
- **Rules**: CONTRAINDICATED (eGFR < 30)
- **ML**: Probability = 0.15 → "Safe"
- **Final**: CONTRAINDICATED (Rules win, ML ignored)

---

## 💡 HOW SUGGESTIONS WORK

### **⚠️ IMPORTANT: Suggestions are NOT from ML Models!**

**Suggestions use rule-based similarity matching, NOT ML predictions.**

### **1. General Alternative Suggestions** (`/alternatives`)

**File**: `backend/app/services/interaction_service.py` (lines 272-494)

**How it works**:
```
1. Find drugs in the SAME CLASS as the original drug
2. Check if those drugs interact with the other drug (database lookup)
3. Calculate similarity score based on:
   - Same drug class (40% weight)
   - Similar indication (30% weight)
   - Similar mechanism (20% weight)
   - Name similarity (10% weight)
4. Return drugs with NO interaction or MINOR interaction only
```

**Code**:
```python
async def find_alternatives(drug1_name, drug2_name):
    # Get drugs in same class as drug1
    similar_drugs = await self._get_similar_drugs(drug1, limit=20)
    
    for similar_drug in similar_drugs:
        # Check if similar_drug interacts with drug2 (database lookup)
        interaction = await self._get_interaction(similar_drug.id, drug2.id)
        
        # Only suggest if NO interaction or MINOR interaction
        if not interaction or interaction.severity == "minor":
            alternatives.append(similar_drug)
    
    # Sort by similarity score
    alternatives.sort(key=lambda x: x.similarity_score, reverse=True)
    return alternatives[:5]
```

**Example**:
- **Original**: Metformin + Verapamil (interaction found)
- **Find alternatives for Metformin**:
  1. Get drugs in same class: Glipizide, Glyburide, Glimepiride
  2. Check each against Verapamil:
     - Glipizide + Verapamil → No interaction ✅
     - Glyburide + Verapamil → Minor interaction ✅
     - Glimepiride + Verapamil → Major interaction ❌
  3. Return: Glipizide, Glyburide (sorted by similarity)

---

### **2. Diabetic Patient Safe Alternatives** (`/diabetic/alternatives`)

**File**: `backend/app/diabetic/rules.py` (lines 612-700)

**How it works**:
```
1. Hard-coded alternative mappings based on drug classes
2. Clinical knowledge-based (not ML)
3. Specific to diabetic patients
```

**Code**:
```python
def _get_alternatives(self, drug: str, patient: Dict) -> List[str]:
    """Get safer alternatives for a drug class."""
    alternatives_map = {
        "metformin": ["GLP-1 agonists", "SGLT2 inhibitors"],
        "glyburide": ["Glipizide", "Glimepiride"],
        "verapamil": ["Cardioselective beta-blockers"],
        "furosemide": ["Thiazide diuretics"],
        # ... more mappings
    }
    return alternatives_map.get(drug, [])
```

**Example**:
- **Drug**: Metformin (contraindicated in eGFR < 30)
- **Alternatives**: ["GLP-1 agonists", "SGLT2 inhibitors"]
- **Reason**: These don't require kidney function adjustment

---

## 🔄 ML VS RULES COMPARISON

### **General Interaction Check** (`/interactions/check`)

| Aspect | Rules-Based | ML-Based |
|--------|------------|----------|
| **Source** | Database lookup (`drug_interactions` table) | Trained models (TWOSIDES) |
| **Coverage** | Known interactions only | Can predict unknown pairs |
| **Priority** | Overrides ML for high-risk | Used when no rules found |
| **Speed** | Fast (SQL query) | Slower (feature extraction + prediction) |
| **Accuracy** | 100% for known interactions | 85-96% accuracy |

**Decision Logic**:
```
IF rules.find_interaction() == CONTRAINDICATED or MAJOR:
    → Use rules (safety first!)
ELSE IF ml.predict() is available:
    → Use ML prediction
ELSE:
    → Use rules result
```

---

### **Diabetic Patient Risk Check** (`/diabetic/risk-check`)

| Aspect | Rules-Based | ML-Based |
|--------|------------|----------|
| **Source** | Clinical rules (eGFR thresholds, drug classes) | Trained models |
| **Priority** | **PRIMARY** (always wins) | **SUPPLEMENTARY** (info only) |
| **Use Case** | Patient-specific risk (eGFR, complications) | General interaction probability |
| **Override** | Always overrides ML | Never overrides rules |

**Decision Logic**:
```
1. Rules assess risk (PRIMARY)
   → Risk level: safe/caution/high_risk/contraindicated/fatal
   → Severity: minor/moderate/major/contraindicated
   → Explanation: Clinical reasoning

2. ML predicts (SUPPLEMENTARY)
   → Probability: 0.0 - 1.0
   → Risk level: safe/unsafe
   → Stored but NOT used for decision

3. Final decision = Rules result
   ML info shown for transparency only
```

---

## 📊 COMPLETE FLOW DIAGRAMS

### **Flow 1: General Drug Interaction Check**

```
User Request: Check Drug A + Drug B
    │
    ├─→ [Rules Check]
    │   └─→ Database lookup (drug_interactions table)
    │       ├─→ Found interaction? → Return severity
    │       └─→ Not found? → Continue to ML
    │
    ├─→ [ML Prediction]
    │   └─→ Load models (RF, XGB, LGB)
    │       ├─→ Extract features (hash encoding)
    │       ├─→ Get probabilities from each model
    │       ├─→ Ensemble (average)
    │       ├─→ Apply optimal threshold (0.2345)
    │       └─→ Predict severity
    │
    └─→ [Hybrid Decision]
        ├─→ IF rules = CONTRAINDICATED/MAJOR
        │   └─→ Use rules (safety first!)
        │
        ├─→ ELSE IF ML available
        │   └─→ Use ML prediction
        │
        └─→ ELSE
            └─→ Use rules result
```

---

### **Flow 2: Diabetic Patient Risk Check**

```
User Request: Check Drug X for Patient Y (eGFR=28)
    │
    ├─→ [Rules Assessment] (PRIMARY)
    │   └─→ Check specific drugs (Verapamil, Metformin, etc.)
    │       ├─→ Check drug classes (-pril, -sartan, etc.)
    │       ├─→ Check eGFR thresholds
    │       ├─→ Check complications
    │       └─→ Return: CONTRAINDICATED
    │
    ├─→ [ML Prediction] (SUPPLEMENTARY)
    │   └─→ Predict general interaction probability
    │       └─→ Return: Probability = 0.15 (Safe)
    │
    └─→ [Final Decision]
        └─→ ALWAYS use Rules result
            └─→ CONTRAINDICATED (ML ignored)
            └─→ ML info shown for transparency
```

---

### **Flow 3: Alternative Suggestions**

```
User Request: Find alternatives for Drug A + Drug B (interaction found)
    │
    ├─→ [Find Similar Drugs]
    │   └─→ Get drugs in same class as Drug A
    │       └─→ Calculate similarity score:
    │           ├─→ Same class: 40%
    │           ├─→ Similar indication: 30%
    │           ├─→ Similar mechanism: 20%
    │           └─→ Name similarity: 10%
    │
    ├─→ [Check Interactions] (Database lookup, NOT ML)
    │   └─→ For each similar drug:
    │       ├─→ Check if it interacts with Drug B
    │       └─→ Keep only if NO interaction or MINOR
    │
    └─→ [Return Alternatives]
        └─→ Sort by similarity score
        └─→ Return top 5 alternatives
```

**⚠️ Note**: This uses **database lookups**, NOT ML predictions!

---

## 🎯 KEY TAKEAWAYS

### **1. ML Integration Points**
- ✅ **General interaction check**: ML used when no rules found
- ✅ **Diabetic patient check**: ML supplementary only (never overrides)
- ❌ **Alternative suggestions**: NOT using ML (rule-based similarity)

### **2. ML Models Purpose**
- **Predict interaction probability** for drug pairs
- **Trained on TWOSIDES** (42M+ interactions)
- **Ensemble approach** (3 models averaged)
- **Optimal threshold** (0.2345) for better NPV

### **3. Suggestions Are NOT ML**
- **Rule-based similarity matching**
- **Database lookups** for interaction checks
- **Clinical knowledge** for diabetic alternatives
- **No ML predictions** used for suggestions

### **4. Safety Priority**
- **Rules always win** for high-risk (contraindicated/major)
- **ML supplementary** for diabetic patients
- **Database lookups** for suggestions (most reliable)

---

## 🔍 EXAMPLE SCENARIOS

### **Scenario 1: General Check - Unknown Pair**
```
Drugs: NewDrugA + NewDrugB
Rules: Not found in database
ML: Probability = 0.65 → "Major interaction"
Result: Major interaction (ML decision)
```

### **Scenario 2: General Check - Known High-Risk**
```
Drugs: Warfarin + Aspirin
Rules: MAJOR interaction found
ML: Probability = 0.20 → "Safe"
Result: MAJOR interaction (Rules override ML)
```

### **Scenario 3: Diabetic Patient - Low eGFR**
```
Patient: eGFR = 28
Drug: Verapamil
Rules: CONTRAINDICATED (eGFR < 30)
ML: Probability = 0.15 → "Safe"
Result: CONTRAINDICATED (Rules always win)
ML Info: Shown but ignored
```

### **Scenario 4: Alternative Suggestions**
```
Original: Metformin + Verapamil (interaction)
Find alternatives for Metformin:
  1. Get similar drugs: Glipizide, Glyburide
  2. Check each vs Verapamil (database lookup):
     - Glipizide + Verapamil → No interaction ✅
     - Glyburide + Verapamil → Minor interaction ✅
  3. Return: Glipizide, Glyburide
  (NOT using ML predictions!)
```

---

## 📝 SUMMARY

### **ML Integration**:
- ✅ **Integrated** in general interaction checks
- ✅ **Integrated** in diabetic patient checks (supplementary)
- ❌ **NOT used** for alternative suggestions

### **Suggestions**:
- ✅ **Rule-based similarity** matching
- ✅ **Database lookups** for interaction checks
- ✅ **Clinical knowledge** for diabetic alternatives
- ❌ **NOT ML predictions**

### **Priority**:
- 🥇 **Rules** (safety first!)
- 🥈 **ML** (supplementary/fallback)
- 🥉 **Database** (for suggestions)

---

**Made with ❤️ for patient safety**



