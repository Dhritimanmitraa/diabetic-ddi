# Project Title

**A Hybrid Rule-Based and Machine Learning System for Drug Safety Assessment in Diabetic Patients**

---

## Abstract

Diabetes mellitus affects millions of patients worldwide and requires complex medication management, often involving multiple drugs to control blood glucose, prevent complications, and manage comorbidities. Diabetic patients face unique risks when taking medications due to altered pharmacokinetics, renal impairment, and drug interactions that can lead to severe adverse events including hypoglycemia, hyperkalemia, nephrotoxicity, and lactic acidosis. Traditional drug interaction checkers fail to account for diabetes-specific factors such as kidney function (eGFR), diabetic complications, and glucose-altering drug effects, resulting in potentially fatal medication errors.

This project presents an intelligent clinical decision support system that combines rule-based clinical guidelines with machine learning models to assess drug safety specifically for diabetic patients. The system integrates evidence-based rules from clinical guidelines (ADA, AACE) with ensemble machine learning models (Random Forest, XGBoost, LightGBM) trained on over 2 million drug-drug interaction records from the TWOSIDES database. The hybrid approach prioritizes clinically validated rules for critical safety decisions while leveraging ML predictions for supplementary risk assessment and pattern recognition.

The system evaluates multiple patient-specific factors including estimated glomerular filtration rate (eGFR), serum potassium, liver function, diabetic complications (nephropathy, retinopathy, cardiovascular disease), and current medication regimens. It identifies fatal and contraindicated drug combinations, provides dose adjustment recommendations based on renal function, flags drugs that mask hypoglycemia symptoms, and suggests safer therapeutic alternatives. The rule engine covers over 200 drugs and drug classes with pattern-based matching for comprehensive coverage, while the ML component provides probabilistic risk scores calibrated using optimal threshold tuning to address class imbalance.

Evaluated on real-world diabetic patient scenarios, the system demonstrates high accuracy in identifying contraindicated medications (particularly in patients with chronic kidney disease), with rule-based decisions taking precedence over ML predictions for critical safety alerts. The framework provides actionable clinical recommendations, monitoring requirements, and alternative drug suggestions, making it suitable for integration into electronic health records and clinical pharmacy workflows to prevent medication-related harm in diabetic populations.

## Keywords

Diabetic Drug Safety, Clinical Decision Support System, Drug-Drug Interactions, Rule-Based Expert System, Machine Learning, Chronic Kidney Disease, Medication Safety, Pharmacovigilance, Clinical Guidelines

---

## Alternative Title Options

1. **A Hybrid Rule-Based and Machine Learning System for Drug Safety Assessment in Diabetic Patients**

2. **Intelligent Drug Risk Assessment for Diabetic Patients: A Clinical Decision Support System Combining Evidence-Based Rules and Machine Learning**

3. **Preventing Medication Errors in Diabetes: A Hybrid Clinical Decision Support System for Drug Safety Evaluation**

4. **Diabetic Drug Safety Prediction: Integrating Clinical Guidelines with Machine Learning for Personalized Medication Risk Assessment**

5. **A Clinical Decision Support Framework for Drug-Drug Interaction Assessment in Diabetic Patients with Chronic Kidney Disease**







