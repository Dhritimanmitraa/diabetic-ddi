"""
Seed script to populate the drugs table with common medications and known interactions.

Run:
    cd backend
    venv\Scripts\python -m scripts.seed_drugs
"""
import asyncio
import json
import sys
import os

# Ensure the backend root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import select, func
from app.database import engine, async_session, init_db
from app.models import Drug, DrugInteraction

# ---------------------------------------------------------------------------
# Drug data – covers common medications including diabetic, cardiovascular,
# analgesic, antibiotic, psychiatric, and other widely-used drug classes.
# ---------------------------------------------------------------------------
DRUGS = [
    # ── Cardiovascular / Anticoagulants ──
    {"name": "Warfarin", "generic_name": "Warfarin Sodium", "brand_names": json.dumps(["Coumadin", "Jantoven"]), "drug_class": "Anticoagulant", "description": "An anticoagulant (blood thinner) used to prevent blood clots.", "mechanism": "Inhibits vitamin K-dependent synthesis of clotting factors II, VII, IX, and X.", "indication": "Prevention and treatment of thromboembolic disorders and pulmonary embolism.", "is_approved": True},
    {"name": "Aspirin", "generic_name": "Acetylsalicylic Acid", "brand_names": json.dumps(["Bayer", "Ecotrin", "Bufferin"]), "drug_class": "NSAID / Antiplatelet", "description": "A nonsteroidal anti-inflammatory drug used for pain, fever, and antiplatelet therapy.", "mechanism": "Irreversibly inhibits cyclooxygenase (COX-1 and COX-2), reducing prostaglandin and thromboxane synthesis.", "indication": "Pain, fever, inflammation, and cardiovascular prevention.", "is_approved": True},
    {"name": "Clopidogrel", "generic_name": "Clopidogrel Bisulfate", "brand_names": json.dumps(["Plavix"]), "drug_class": "Antiplatelet Agent", "description": "An antiplatelet agent used to prevent blood clots in patients with cardiovascular disease.", "mechanism": "Irreversibly inhibits the P2Y12 component of ADP receptors on platelets.", "indication": "Acute coronary syndrome, recent MI, stroke, or peripheral arterial disease.", "is_approved": True},
    {"name": "Atorvastatin", "generic_name": "Atorvastatin Calcium", "brand_names": json.dumps(["Lipitor"]), "drug_class": "HMG-CoA Reductase Inhibitor (Statin)", "description": "A statin used to lower cholesterol and reduce the risk of cardiovascular disease.", "mechanism": "Competitively inhibits HMG-CoA reductase, the rate-limiting enzyme in cholesterol biosynthesis.", "indication": "Hyperlipidemia and cardiovascular risk reduction.", "is_approved": True},
    {"name": "Simvastatin", "generic_name": "Simvastatin", "brand_names": json.dumps(["Zocor"]), "drug_class": "HMG-CoA Reductase Inhibitor (Statin)", "description": "A statin medication used to control elevated cholesterol.", "mechanism": "Inhibits HMG-CoA reductase, decreasing hepatic cholesterol synthesis.", "indication": "Hypercholesterolemia and cardiovascular risk reduction.", "is_approved": True},
    {"name": "Lisinopril", "generic_name": "Lisinopril", "brand_names": json.dumps(["Prinivil", "Zestril"]), "drug_class": "ACE Inhibitor", "description": "An ACE inhibitor used to treat high blood pressure and heart failure.", "mechanism": "Inhibits angiotensin-converting enzyme, preventing conversion of angiotensin I to angiotensin II.", "indication": "Hypertension, heart failure, and post-myocardial infarction.", "is_approved": True},
    {"name": "Amlodipine", "generic_name": "Amlodipine Besylate", "brand_names": json.dumps(["Norvasc"]), "drug_class": "Calcium Channel Blocker", "description": "A calcium channel blocker used to treat hypertension and angina.", "mechanism": "Inhibits calcium ion influx across cell membranes of cardiac and vascular smooth muscle.", "indication": "Hypertension and chronic stable angina.", "is_approved": True},
    {"name": "Losartan", "generic_name": "Losartan Potassium", "brand_names": json.dumps(["Cozaar"]), "drug_class": "Angiotensin II Receptor Blocker (ARB)", "description": "An ARB used to treat high blood pressure and protect the kidneys from damage due to diabetes.", "mechanism": "Blocks the binding of angiotensin II to the AT1 receptor.", "indication": "Hypertension and diabetic nephropathy.", "is_approved": True},

    # ── Diabetic Medications ──
    {"name": "Metformin", "generic_name": "Metformin Hydrochloride", "brand_names": json.dumps(["Glucophage", "Fortamet", "Glumetza"]), "drug_class": "Biguanide", "description": "An oral antidiabetic used as first-line treatment for type 2 diabetes.", "mechanism": "Decreases hepatic glucose production and increases insulin sensitivity in peripheral tissues.", "indication": "Type 2 diabetes mellitus.", "is_approved": True},
    {"name": "Glipizide", "generic_name": "Glipizide", "brand_names": json.dumps(["Glucotrol"]), "drug_class": "Sulfonylurea", "description": "A sulfonylurea used to treat type 2 diabetes by stimulating insulin secretion.", "mechanism": "Stimulates insulin release from pancreatic beta cells by blocking ATP-sensitive potassium channels.", "indication": "Type 2 diabetes mellitus.", "is_approved": True},
    {"name": "Glyburide", "generic_name": "Glyburide", "brand_names": json.dumps(["Diabeta", "Micronase"]), "drug_class": "Sulfonylurea", "description": "A sulfonylurea used to lower blood sugar in patients with type 2 diabetes.", "mechanism": "Stimulates insulin release from pancreatic beta cells.", "indication": "Type 2 diabetes mellitus.", "is_approved": True},
    {"name": "Pioglitazone", "generic_name": "Pioglitazone Hydrochloride", "brand_names": json.dumps(["Actos"]), "drug_class": "Thiazolidinedione (TZD)", "description": "A TZD used to improve glycemic control in type 2 diabetes.", "mechanism": "Activates peroxisome proliferator-activated receptor gamma (PPARγ), improving insulin sensitivity.", "indication": "Type 2 diabetes mellitus.", "is_approved": True},
    {"name": "Sitagliptin", "generic_name": "Sitagliptin Phosphate", "brand_names": json.dumps(["Januvia"]), "drug_class": "DPP-4 Inhibitor", "description": "A DPP-4 inhibitor used to improve blood sugar control in type 2 diabetes.", "mechanism": "Inhibits dipeptidyl peptidase-4, increasing incretin levels which inhibit glucagon and stimulate insulin.", "indication": "Type 2 diabetes mellitus.", "is_approved": True},
    {"name": "Empagliflozin", "generic_name": "Empagliflozin", "brand_names": json.dumps(["Jardiance"]), "drug_class": "SGLT2 Inhibitor", "description": "An SGLT2 inhibitor used to lower blood sugar and reduce cardiovascular risk in diabetes.", "mechanism": "Inhibits sodium-glucose co-transporter 2 in the proximal renal tubules, reducing glucose reabsorption.", "indication": "Type 2 diabetes mellitus and heart failure.", "is_approved": True},
    {"name": "Insulin Glargine", "generic_name": "Insulin Glargine", "brand_names": json.dumps(["Lantus", "Basaglar", "Toujeo"]), "drug_class": "Long-acting Insulin", "description": "A long-acting insulin analog used for basal insulin replacement in diabetes.", "mechanism": "Forms microprecipitates in subcutaneous tissue, providing slow, peakless insulin release over 24 hours.", "indication": "Type 1 and Type 2 diabetes mellitus.", "is_approved": True},
    {"name": "Liraglutide", "generic_name": "Liraglutide", "brand_names": json.dumps(["Victoza", "Saxenda"]), "drug_class": "GLP-1 Receptor Agonist", "description": "A GLP-1 receptor agonist for glycemic control and cardiovascular risk reduction.", "mechanism": "Mimics GLP-1, enhancing glucose-dependent insulin secretion and suppressing glucagon.", "indication": "Type 2 diabetes mellitus and obesity.", "is_approved": True},

    # ── Analgesics / Anti-inflammatory ──
    {"name": "Ibuprofen", "generic_name": "Ibuprofen", "brand_names": json.dumps(["Advil", "Motrin"]), "drug_class": "NSAID", "description": "A nonsteroidal anti-inflammatory drug used for pain, fever, and inflammation.", "mechanism": "Inhibits cyclooxygenase (COX-1 and COX-2), reducing prostaglandin synthesis.", "indication": "Pain, fever, and inflammatory conditions.", "is_approved": True},
    {"name": "Acetaminophen", "generic_name": "Acetaminophen", "brand_names": json.dumps(["Tylenol", "Panadol"]), "drug_class": "Analgesic / Antipyretic", "description": "An analgesic and antipyretic used for pain relief and fever reduction.", "mechanism": "Inhibits prostaglandin synthesis in the central nervous system and acts on the heat-regulating center.", "indication": "Mild to moderate pain and fever.", "is_approved": True},
    {"name": "Naproxen", "generic_name": "Naproxen Sodium", "brand_names": json.dumps(["Aleve", "Naprosyn"]), "drug_class": "NSAID", "description": "A nonsteroidal anti-inflammatory drug used for pain and inflammation.", "mechanism": "Inhibits cyclooxygenase (COX-1 and COX-2), reducing prostaglandin synthesis.", "indication": "Pain, inflammation, arthritis, and menstrual cramps.", "is_approved": True},

    # ── Antibiotics ──
    {"name": "Amoxicillin", "generic_name": "Amoxicillin", "brand_names": json.dumps(["Amoxil", "Trimox"]), "drug_class": "Penicillin Antibiotic", "description": "A penicillin-type antibiotic used to treat bacterial infections.", "mechanism": "Inhibits bacterial cell wall synthesis by binding to penicillin-binding proteins.", "indication": "Bacterial infections of the ear, nose, throat, urinary tract, and skin.", "is_approved": True},
    {"name": "Azithromycin", "generic_name": "Azithromycin", "brand_names": json.dumps(["Zithromax", "Z-Pack"]), "drug_class": "Macrolide Antibiotic", "description": "A macrolide antibiotic used to treat a variety of bacterial infections.", "mechanism": "Binds to the 50S ribosomal subunit, inhibiting bacterial protein synthesis.", "indication": "Respiratory infections, skin infections, and sexually transmitted diseases.", "is_approved": True},
    {"name": "Ciprofloxacin", "generic_name": "Ciprofloxacin Hydrochloride", "brand_names": json.dumps(["Cipro"]), "drug_class": "Fluoroquinolone Antibiotic", "description": "A fluoroquinolone antibiotic used to treat various bacterial infections.", "mechanism": "Inhibits bacterial DNA gyrase and topoisomerase IV, preventing DNA replication.", "indication": "Urinary tract infections, respiratory infections, and gastrointestinal infections.", "is_approved": True},

    # ── Psychiatric / CNS ──
    {"name": "Sertraline", "generic_name": "Sertraline Hydrochloride", "brand_names": json.dumps(["Zoloft"]), "drug_class": "SSRI Antidepressant", "description": "A selective serotonin reuptake inhibitor used to treat depression and anxiety.", "mechanism": "Selectively inhibits the reuptake of serotonin in the central nervous system.", "indication": "Major depressive disorder, OCD, PTSD, panic disorder, and social anxiety.", "is_approved": True},
    {"name": "Fluoxetine", "generic_name": "Fluoxetine Hydrochloride", "brand_names": json.dumps(["Prozac"]), "drug_class": "SSRI Antidepressant", "description": "An SSRI antidepressant used to treat depression, OCD, and eating disorders.", "mechanism": "Selectively inhibits serotonin reuptake at the presynaptic neuronal membrane.", "indication": "Major depression, OCD, panic disorder, and bulimia nervosa.", "is_approved": True},
    {"name": "Alprazolam", "generic_name": "Alprazolam", "brand_names": json.dumps(["Xanax"]), "drug_class": "Benzodiazepine", "description": "A benzodiazepine used for the treatment of anxiety and panic disorders.", "mechanism": "Enhances the effect of GABA at the GABA-A receptor, producing sedative/anxiolytic effects.", "indication": "Anxiety disorders and panic disorder.", "is_approved": True},

    # ── Respiratory ──
    {"name": "Albuterol", "generic_name": "Albuterol Sulfate", "brand_names": json.dumps(["ProAir", "Ventolin"]), "drug_class": "Short-acting Beta-2 Agonist", "description": "A bronchodilator used to treat asthma and COPD.", "mechanism": "Stimulates beta-2 adrenergic receptors in the lungs, causing bronchial smooth muscle relaxation.", "indication": "Asthma and COPD.", "is_approved": True},
    {"name": "Montelukast", "generic_name": "Montelukast Sodium", "brand_names": json.dumps(["Singulair"]), "drug_class": "Leukotriene Receptor Antagonist", "description": "A leukotriene receptor antagonist used for asthma and allergies.", "mechanism": "Blocks the cysteinyl leukotriene receptor CysLT1, reducing inflammation.", "indication": "Asthma and allergic rhinitis.", "is_approved": True},

    # ── Gastrointestinal ──
    {"name": "Omeprazole", "generic_name": "Omeprazole", "brand_names": json.dumps(["Prilosec"]), "drug_class": "Proton Pump Inhibitor", "description": "A proton pump inhibitor used to reduce stomach acid production.", "mechanism": "Irreversibly blocks the hydrogen/potassium ATPase enzyme system (proton pump) in gastric parietal cells.", "indication": "GERD, peptic ulcers, and Zollinger-Ellison syndrome.", "is_approved": True},
    {"name": "Pantoprazole", "generic_name": "Pantoprazole Sodium", "brand_names": json.dumps(["Protonix"]), "drug_class": "Proton Pump Inhibitor", "description": "A proton pump inhibitor used to treat erosive esophagitis and GERD.", "mechanism": "Irreversibly inhibits the H+/K+ ATPase proton pump in gastric parietal cells.", "indication": "GERD and erosive esophagitis.", "is_approved": True},

    # ── Thyroid ──
    {"name": "Levothyroxine", "generic_name": "Levothyroxine Sodium", "brand_names": json.dumps(["Synthroid", "Levoxyl"]), "drug_class": "Thyroid Hormone", "description": "A synthetic thyroid hormone used to treat hypothyroidism.", "mechanism": "Replaces endogenous thyroxine (T4), restoring normal thyroid hormone levels.", "indication": "Hypothyroidism and thyroid hormone replacement.", "is_approved": True},

    # ── Corticosteroids ──
    {"name": "Prednisone", "generic_name": "Prednisone", "brand_names": json.dumps(["Deltasone", "Rayos"]), "drug_class": "Corticosteroid", "description": "A corticosteroid used for its anti-inflammatory and immunosuppressive properties.", "mechanism": "Binds to intracellular glucocorticoid receptors, modulating gene expression to reduce inflammation.", "indication": "Inflammatory and autoimmune conditions, allergic reactions, and asthma.", "is_approved": True},

    # ── Diuretics ──
    {"name": "Hydrochlorothiazide", "generic_name": "Hydrochlorothiazide", "brand_names": json.dumps(["Microzide"]), "drug_class": "Thiazide Diuretic", "description": "A thiazide diuretic used to treat hypertension and edema.", "mechanism": "Inhibits sodium reabsorption in the distal convoluted tubule of the kidney.", "indication": "Hypertension and edema.", "is_approved": True},
    {"name": "Furosemide", "generic_name": "Furosemide", "brand_names": json.dumps(["Lasix"]), "drug_class": "Loop Diuretic", "description": "A loop diuretic used to treat fluid retention and hypertension.", "mechanism": "Inhibits reabsorption of sodium and chloride in the ascending loop of Henle.", "indication": "Edema, heart failure, and hypertension.", "is_approved": True},

    # ── Additional common medications ──
    {"name": "Metoprolol", "generic_name": "Metoprolol Tartrate", "brand_names": json.dumps(["Lopressor", "Toprol-XL"]), "drug_class": "Beta-Blocker", "description": "A selective beta-1 blocker used to treat hypertension, angina, and heart failure.", "mechanism": "Selectively blocks beta-1 adrenergic receptors in the heart, reducing heart rate and cardiac output.", "indication": "Hypertension, angina, heart failure, and post-MI.", "is_approved": True},
    {"name": "Gabapentin", "generic_name": "Gabapentin", "brand_names": json.dumps(["Neurontin", "Gralise"]), "drug_class": "Anticonvulsant / Analgesic", "description": "An anticonvulsant used to treat seizures and neuropathic pain.", "mechanism": "Binds to the alpha-2-delta subunit of voltage-gated calcium channels, modulating neurotransmitter release.", "indication": "Epilepsy and neuropathic pain.", "is_approved": True},
    {"name": "Tramadol", "generic_name": "Tramadol Hydrochloride", "brand_names": json.dumps(["Ultram"]), "drug_class": "Opioid Analgesic", "description": "A centrally acting synthetic opioid analgesic for moderate to moderately severe pain.", "mechanism": "Binds to mu-opioid receptors and inhibits reuptake of norepinephrine and serotonin.", "indication": "Moderate to moderately severe pain.", "is_approved": True},
    {"name": "Rosuvastatin", "generic_name": "Rosuvastatin Calcium", "brand_names": json.dumps(["Crestor"]), "drug_class": "HMG-CoA Reductase Inhibitor (Statin)", "description": "A statin used to lower cholesterol and reduce cardiovascular risk.", "mechanism": "Inhibits HMG-CoA reductase, reducing hepatic cholesterol synthesis.", "indication": "Hyperlipidemia and cardiovascular risk reduction.", "is_approved": True},
    {"name": "Dapagliflozin", "generic_name": "Dapagliflozin", "brand_names": json.dumps(["Farxiga"]), "drug_class": "SGLT2 Inhibitor", "description": "An SGLT2 inhibitor for blood sugar control and heart failure.", "mechanism": "Inhibits SGLT2 in the kidney, reducing glucose reabsorption and increasing urinary glucose excretion.", "indication": "Type 2 diabetes mellitus, heart failure, and chronic kidney disease.", "is_approved": True},
    {"name": "Canagliflozin", "generic_name": "Canagliflozin", "brand_names": json.dumps(["Invokana"]), "drug_class": "SGLT2 Inhibitor", "description": "An SGLT2 inhibitor used for glycemic control in type 2 diabetes.", "mechanism": "Inhibits SGLT2, reducing renal glucose reabsorption.", "indication": "Type 2 diabetes mellitus.", "is_approved": True},
    {"name": "Glimepiride", "generic_name": "Glimepiride", "brand_names": json.dumps(["Amaryl"]), "drug_class": "Sulfonylurea", "description": "A sulfonylurea used to lower blood sugar in type 2 diabetes.", "mechanism": "Stimulates insulin secretion from pancreatic beta cells.", "indication": "Type 2 diabetes mellitus.", "is_approved": True},
]

# ---------------------------------------------------------------------------
# Known drug-drug interactions
# ---------------------------------------------------------------------------
INTERACTIONS = [
    # Warfarin interactions
    {"drug1": "Warfarin", "drug2": "Aspirin", "severity": "major", "description": "Concurrent use of warfarin and aspirin significantly increases the risk of bleeding.", "effect": "Increased risk of gastrointestinal and intracranial bleeding.", "mechanism": "Both drugs inhibit different aspects of hemostasis; aspirin inhibits platelet aggregation while warfarin inhibits coagulation factors.", "management": "Avoid combination unless specifically directed by a physician. Monitor INR closely if used together.", "source": "drugbank", "evidence_level": "established", "confidence_score": 0.95},
    {"drug1": "Warfarin", "drug2": "Ibuprofen", "severity": "major", "description": "NSAIDs increase the risk of bleeding when combined with warfarin.", "effect": "Increased anticoagulant effect and risk of GI bleeding.", "mechanism": "Ibuprofen inhibits platelet function and may displace warfarin from protein binding sites.", "management": "Avoid concurrent use. Use acetaminophen for pain relief instead.", "source": "drugbank", "evidence_level": "established", "confidence_score": 0.95},
    {"drug1": "Warfarin", "drug2": "Fluoxetine", "severity": "major", "description": "Fluoxetine inhibits CYP2C9, increasing warfarin levels and bleeding risk.", "effect": "Increased INR and bleeding risk.", "mechanism": "Fluoxetine inhibits cytochrome P450 2C9, the primary enzyme metabolizing warfarin.", "management": "Monitor INR closely. Consider dose adjustment of warfarin.", "source": "drugbank", "evidence_level": "established", "confidence_score": 0.90},
    {"drug1": "Warfarin", "drug2": "Omeprazole", "severity": "moderate", "description": "Omeprazole may increase warfarin concentrations by inhibiting CYP2C19.", "effect": "Slightly increased anticoagulant effect.", "mechanism": "Omeprazole inhibits CYP2C19 which may reduce R-warfarin metabolism.", "management": "Monitor INR when starting or stopping omeprazole.", "source": "drugbank", "evidence_level": "established", "confidence_score": 0.80},
    {"drug1": "Warfarin", "drug2": "Simvastatin", "severity": "moderate", "description": "Simvastatin may enhance the anticoagulant effect of warfarin.", "effect": "Increased INR and potential bleeding risk.", "mechanism": "Competition for CYP3A4 metabolism pathway.", "management": "Monitor INR when initiating or changing simvastatin dose.", "source": "drugbank", "evidence_level": "established", "confidence_score": 0.80},
    {"drug1": "Warfarin", "drug2": "Ciprofloxacin", "severity": "major", "description": "Ciprofloxacin significantly increases the anticoagulant effect of warfarin.", "effect": "Elevated INR and increased risk of serious bleeding.", "mechanism": "Ciprofloxacin inhibits CYP1A2 and alters gut flora, both affecting warfarin metabolism.", "management": "Monitor INR closely. Consider dose reduction of warfarin.", "source": "drugbank", "evidence_level": "established", "confidence_score": 0.92},

    # Metformin interactions
    {"drug1": "Metformin", "drug2": "Insulin Glargine", "severity": "moderate", "description": "Concurrent use increases the risk of hypoglycemia.", "effect": "Enhanced blood glucose-lowering effect, risk of hypoglycemia.", "mechanism": "Additive hypoglycemic effects from dual antidiabetic therapy.", "management": "Monitor blood glucose closely. Adjust insulin dose as needed.", "source": "clinical", "evidence_level": "established", "confidence_score": 0.85},
    {"drug1": "Metformin", "drug2": "Furosemide", "severity": "moderate", "description": "Furosemide may increase metformin plasma levels.", "effect": "Increased risk of lactic acidosis.", "mechanism": "Furosemide may increase metformin levels by competing for renal tubular secretion.", "management": "Monitor for signs of lactic acidosis. Consider dose adjustment.", "source": "clinical", "evidence_level": "established", "confidence_score": 0.80},
    {"drug1": "Metformin", "drug2": "Ciprofloxacin", "severity": "moderate", "description": "Ciprofloxacin may alter blood glucose levels when combined with metformin.", "effect": "Dysglycemia (hypoglycemia or hyperglycemia).", "mechanism": "Fluoroquinolones can alter insulin secretion and glucose homeostasis.", "management": "Monitor blood glucose closely during concurrent use.", "source": "clinical", "evidence_level": "established", "confidence_score": 0.75},

    # NSAID interactions
    {"drug1": "Ibuprofen", "drug2": "Aspirin", "severity": "moderate", "description": "Ibuprofen may interfere with the antiplatelet effect of low-dose aspirin.", "effect": "Reduced cardioprotective effect of aspirin.", "mechanism": "Ibuprofen competes with aspirin for the COX-1 binding site, blocking aspirin's irreversible platelet inhibition.", "management": "Take aspirin at least 30 minutes before or 8 hours after ibuprofen.", "source": "fda", "evidence_level": "established", "confidence_score": 0.90},
    {"drug1": "Ibuprofen", "drug2": "Lisinopril", "severity": "moderate", "description": "NSAIDs may reduce the antihypertensive effect of ACE inhibitors.", "effect": "Decreased blood pressure control and increased risk of renal impairment.", "mechanism": "NSAIDs inhibit prostaglandin synthesis, which can reduce renal blood flow and counteract ACE inhibitor effects.", "management": "Monitor blood pressure and renal function. Use lowest effective NSAID dose.", "source": "clinical", "evidence_level": "established", "confidence_score": 0.85},
    {"drug1": "Naproxen", "drug2": "Lisinopril", "severity": "moderate", "description": "NSAIDs may reduce the antihypertensive effect of ACE inhibitors.", "effect": "Decreased blood pressure control.", "mechanism": "Prostaglandin synthesis inhibition by NSAIDs counteracts ACE inhibitor effect.", "management": "Monitor blood pressure. Use lowest effective NSAID dose for shortest duration.", "source": "clinical", "evidence_level": "established", "confidence_score": 0.85},

    # Statin interactions
    {"drug1": "Simvastatin", "drug2": "Amlodipine", "severity": "major", "description": "Amlodipine increases simvastatin levels, raising the risk of rhabdomyolysis.", "effect": "Increased risk of myopathy and rhabdomyolysis.", "mechanism": "Amlodipine inhibits CYP3A4, increasing simvastatin plasma concentrations.", "management": "Do not exceed simvastatin 20mg daily when used with amlodipine.", "source": "fda", "evidence_level": "established", "confidence_score": 0.95},
    {"drug1": "Atorvastatin", "drug2": "Azithromycin", "severity": "moderate", "description": "Azithromycin may increase statin levels slightly.", "effect": "Potential increased risk of myopathy.", "mechanism": "Weak CYP3A4 inhibition by azithromycin.", "management": "Monitor for muscle pain or weakness during concurrent use.", "source": "clinical", "evidence_level": "theoretical", "confidence_score": 0.65},

    # Diabetic drug interactions
    {"drug1": "Glipizide", "drug2": "Fluoxetine", "severity": "moderate", "description": "Fluoxetine may increase the hypoglycemic effect of sulfonylureas.", "effect": "Increased risk of hypoglycemia.", "mechanism": "SSRIs may enhance insulin secretion and improve insulin sensitivity.", "management": "Monitor blood glucose closely and adjust sulfonylurea dose as needed.", "source": "clinical", "evidence_level": "established", "confidence_score": 0.80},
    {"drug1": "Glipizide", "drug2": "Aspirin", "severity": "moderate", "description": "High-dose aspirin may enhance the hypoglycemic effect of sulfonylureas.", "effect": "Increased risk of hypoglycemia.", "mechanism": "Aspirin displaces sulfonylureas from protein binding and may have intrinsic hypoglycemic effects.", "management": "Monitor blood glucose when high-dose aspirin is started or stopped.", "source": "clinical", "evidence_level": "established", "confidence_score": 0.75},
    {"drug1": "Empagliflozin", "drug2": "Furosemide", "severity": "moderate", "description": "Combined diuretic effects may cause volume depletion.", "effect": "Increased risk of dehydration, hypotension, and acute kidney injury.", "mechanism": "Both drugs promote fluid loss through different renal mechanisms.", "management": "Assess volume status and renal function. Adjust dosing as needed.", "source": "clinical", "evidence_level": "established", "confidence_score": 0.85},
    {"drug1": "Insulin Glargine", "drug2": "Pioglitazone", "severity": "moderate", "description": "TZDs used with insulin increase the risk of fluid retention and heart failure.", "effect": "Increased risk of edema, weight gain, and congestive heart failure.", "mechanism": "Pioglitazone causes fluid retention which is exacerbated by insulin.", "management": "Monitor for signs of heart failure. Consider dose reduction.", "source": "fda", "evidence_level": "established", "confidence_score": 0.90},

    # CNS interactions
    {"drug1": "Alprazolam", "drug2": "Sertraline", "severity": "moderate", "description": "Sertraline may increase alprazolam plasma levels.", "effect": "Increased sedation and CNS depression.", "mechanism": "Sertraline inhibits CYP3A4, reducing alprazolam metabolism.", "management": "Consider lower alprazolam dose. Monitor for excessive sedation.", "source": "clinical", "evidence_level": "established", "confidence_score": 0.80},
    {"drug1": "Tramadol", "drug2": "Sertraline", "severity": "major", "description": "Combined use increases the risk of serotonin syndrome and seizures.", "effect": "Risk of serotonin syndrome (agitation, hyperthermia, tachycardia) and lowered seizure threshold.", "mechanism": "Both drugs increase serotonin levels — tramadol inhibits reuptake, sertraline blocks SERT.", "management": "Avoid combination if possible. If used together, monitor closely for serotonin syndrome.", "source": "clinical", "evidence_level": "established", "confidence_score": 0.90},
    {"drug1": "Tramadol", "drug2": "Fluoxetine", "severity": "contraindicated", "description": "High risk of serotonin syndrome when tramadol is used with SSRIs.", "effect": "Serotonin syndrome: hyperthermia, rigidity, myoclonus, autonomic instability.", "mechanism": "Synergistic serotonergic activity from both agents.", "management": "AVOID this combination. Use alternative analgesic.", "source": "fda", "evidence_level": "established", "confidence_score": 0.95},

    # Prednisone interactions
    {"drug1": "Prednisone", "drug2": "Ibuprofen", "severity": "moderate", "description": "Combined use increases the risk of gastrointestinal bleeding.", "effect": "Increased GI bleeding and ulceration risk.", "mechanism": "Both agents damage the gastric mucosa through different mechanisms.", "management": "Use gastroprotective therapy (PPI). Monitor for GI symptoms.", "source": "clinical", "evidence_level": "established", "confidence_score": 0.85},
    {"drug1": "Prednisone", "drug2": "Metformin", "severity": "moderate", "description": "Corticosteroids may increase blood glucose levels, counteracting metformin.", "effect": "Hyperglycemia and reduced glycemic control.", "mechanism": "Prednisone stimulates gluconeogenesis and reduces insulin sensitivity.", "management": "Monitor blood glucose closely. Adjust metformin dose or add insulin if needed.", "source": "clinical", "evidence_level": "established", "confidence_score": 0.85},

    # Levothyroxine interactions
    {"drug1": "Levothyroxine", "drug2": "Omeprazole", "severity": "moderate", "description": "PPIs may reduce levothyroxine absorption.", "effect": "Subtherapeutic thyroid hormone levels.", "mechanism": "PPIs raise gastric pH, which may impair levothyroxine dissolution and absorption.", "management": "Separate administration by at least 4 hours. Monitor TSH levels.", "source": "clinical", "evidence_level": "established", "confidence_score": 0.80},

    # Clopidogrel interactions
    {"drug1": "Clopidogrel", "drug2": "Omeprazole", "severity": "major", "description": "Omeprazole significantly reduces the antiplatelet effect of clopidogrel.", "effect": "Reduced antiplatelet activity and increased risk of cardiovascular events.", "mechanism": "Omeprazole inhibits CYP2C19, blocking conversion of clopidogrel to its active metabolite.", "management": "Use pantoprazole instead of omeprazole. Avoid concurrent use.", "source": "fda", "evidence_level": "established", "confidence_score": 0.95},
    {"drug1": "Clopidogrel", "drug2": "Aspirin", "severity": "moderate", "description": "Dual antiplatelet therapy increases bleeding risk but may be therapeutically indicated.", "effect": "Increased risk of bleeding.", "mechanism": "Additive inhibition of platelet aggregation through different mechanisms.", "management": "Use only when clinically indicated (e.g., after stent placement). Monitor for signs of bleeding.", "source": "clinical", "evidence_level": "established", "confidence_score": 0.85},

    # ACE inhibitor + ARB
    {"drug1": "Lisinopril", "drug2": "Losartan", "severity": "major", "description": "Dual RAAS blockade increases risk of renal impairment, hyperkalemia, and hypotension.", "effect": "Hyperkalemia, acute kidney injury, and hypotension.", "mechanism": "Excessive blockade of the renin-angiotensin-aldosterone system.", "management": "Avoid concurrent use. Choose one agent from the ACE inhibitor or ARB class.", "source": "fda", "evidence_level": "established", "confidence_score": 0.95},

    # Potassium-sparing with ACE
    {"drug1": "Lisinopril", "drug2": "Hydrochlorothiazide", "severity": "minor", "description": "Often used therapeutically together; mild risk of electrolyte imbalance.", "effect": "Enhanced antihypertensive effect. Risk of hyponatremia in elderly.", "mechanism": "Complementary mechanisms for blood pressure reduction.", "management": "Monitor electrolytes, especially potassium and sodium. Adjust doses as needed.", "source": "clinical", "evidence_level": "established", "confidence_score": 0.70},
]


async def seed():
    """Seed the database with drugs and interactions."""
    await init_db()

    async with async_session() as session:
        # Check if drugs already exist
        count = await session.scalar(select(func.count(Drug.id)))
        if count and count > 0:
            print(f"Database already has {count} drugs. Skipping seed.")
            return

        print("Seeding drugs...")
        drug_objects = {}
        for d in DRUGS:
            drug = Drug(**d)
            session.add(drug)
            drug_objects[d["name"]] = drug

        # Flush to get IDs assigned
        await session.flush()

        print(f"  Added {len(DRUGS)} drugs.")

        # Now seed interactions
        print("Seeding drug interactions...")
        added = 0
        for ix in INTERACTIONS:
            d1 = drug_objects.get(ix["drug1"])
            d2 = drug_objects.get(ix["drug2"])
            if d1 and d2:
                interaction = DrugInteraction(
                    drug1_id=d1.id,
                    drug2_id=d2.id,
                    severity=ix["severity"],
                    description=ix["description"],
                    effect=ix["effect"],
                    mechanism=ix["mechanism"],
                    management=ix["management"],
                    source=ix["source"],
                    evidence_level=ix["evidence_level"],
                    confidence_score=ix["confidence_score"],
                )
                session.add(interaction)
                added += 1
            else:
                missing = ix["drug1"] if not d1 else ix["drug2"]
                print(f"  WARNING: Drug '{missing}' not found, skipping interaction.")

        await session.commit()
        print(f"  Added {added} drug interactions.")

        # Verify
        total_drugs = await session.scalar(select(func.count(Drug.id)))
        total_ix = await session.scalar(select(func.count(DrugInteraction.id)))
        print(f"\nDone! Database now has {total_drugs} drugs and {total_ix} interactions.")


if __name__ == "__main__":
    asyncio.run(seed())
