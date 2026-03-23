# Plan: Fix ML Pipeline, Hyperparameter Tuning & RAG System

## TL;DR
The codebase has all the right components (Bayesian optimization, ensemble models, ChromaDB RAG, rule-based system) but several critical disconnects prevent them from actually working together. The main issues are: (1) feature dimension mismatch between training (15-17 dims) and prediction (242 dims), (2) Bayesian optimizer exists but is dead code, (3) no retraining mechanism, (4) silent ML failure fallback, (5) fragmented ChromaDB collections. The rule-based system and RAG pipeline are functional.

## Current State Assessment

### Working Well
- Rule-based drug interaction system (PostgreSQL + hardcoded diabetic rules + JSON configs)
- Hybrid decision gate (rules override ML for contraindicated/major severity)
- ChromaDB embedding with sentence-transformers (all-MiniLM, 384D)
- LangChain RecursiveCharacterTextSplitter (500 chars, 50 overlap)
- Similarity search with distance thresholds
- LangGraph RAG orchestration with auto-fetch for unknown drugs

### Critical Issues
1. **Feature dimension mismatch**: Trainer produces 15-17 features, predictor expects 242 → models produce garbage predictions
2. **Bayesian optimizer is dead code**: BayesianOptimizer + Optuna TPE fully implementedbut never invoked
3. **No training endpoint/script for DDI models**: Only diabetic model has a training script (uses synthetic data)
4. **Models loaded from disk on every request**: No caching
5. **Silent ML failure**: `try/except` swallows errors, falls back to rules-only without user knowing
6. **No optimal threshold**: Falls back to hardcoded 0.5
7. **ChromaDB fragmentation**: 4 separate collections with potential redundancy

---

## Steps

### Phase 1: Fix Feature Engineering Alignment (CRITICAL — blocks all ML functionality)

1. **Unify feature extraction** — Create a single `DDIFeatureExtractor` class used by BOTH trainer and predictor
   - File: `backend/app/ml/feature_engineering.py` — refactor `DrugFeatureExtractor` to be the canonical source
   - File: `backend/app/ml/predictor.py` — remove `extract_features_simple()` (lines ~91-145), import from feature_engineering instead
   - Features should include: TF-IDF text similarity (mechanism/indication), categorical encoding (drug class), interaction frequency, molecular weight features, same_class binary
   - Target: consistent N-dimensional feature vector (whatever the trainer produces)

2. **Add PYTHONHASHSEED=0** to any startup scripts/configs to ensure deterministic hashing if hash features are kept
   - Files: `run_app.bat`, `ralph_loop.bat`, `ralph_loop.sh`

### Phase 2: Wire Up Bayesian Hyperparameter Tuning + Training Pipeline

3. **Create DDI training script** — `backend/scripts/train_ddi_models.py`
   - Load real interaction data from PostgreSQL `drug_interactions` table (fix the `NotImplementedError` in `load_data_from_db`)
   - Call `DDITrainer.train_all_models()` which invokes BayesianOptimizer
   - Save trained models + optimal threshold + feature extractor to `backend/models/`
   - File: `backend/app/ml/trainer.py` — implement `load_data_from_db()` (currently raises NotImplementedError at line ~62)

4. **Add training API endpoint** — `POST /ml/train` (admin-only)
   - File: `backend/app/routers/` — create `ml_router.py` or add to existing router
   - Triggers async training with progress reporting
   - Saves model artifacts + training results JSON

5. **Generate optimal threshold** during training
   - File: `backend/app/ml/trainer.py` — after training, compute optimal threshold via precision-recall curve or Youden's J statistic
   - Save to `backend/models/optimal_threshold.json`

### Phase 3: Fix Prediction Pipeline

6. **Add model caching** — load models once at app startup, not per-request
   - File: `backend/app/ml/predictor.py` — use a module-level singleton or FastAPI dependency
   - `get_predictor()` should cache the loaded predictor instance

7. **Fix silent ML failure** — return ML status in response
   - File: `backend/app/routers/interactions.py` — add `ml_available: bool` and `ml_error: Optional[str]` to response
   - Don't silently swallow exceptions; include ML status in API response

8. **Use trained feature extractor** at prediction time
   - File: `backend/app/ml/predictor.py` — load `feature_extractor.pkl` from models dir and use it instead of `extract_features_simple()`

### Phase 4: Consolidate & Verify RAG System

9. **Audit ChromaDB collections** — ensure no orphaned or redundant collections
   - File: `backend/app/prescription/knowledge_base.py` — verify `drug_knowledge` collection is populated on startup
   - File: `backend/app/prescription/langchain_rag_kb.py` — verify `medical_knowledge_rag` collection is populated
   - Consider merging `drug_knowledge` + `web_drug_knowledge` into a single unified collection if they serve the same purpose

10. **Add deduplication** to ChromaDB ingestion
    - File: `backend/app/prescription/langchain_rag_kb.py` — check if document already exists before adding (use metadata drug_name + chunk_index as ID)

11. **Verify embedding consistency** — ensure all collections use the same embedding function
    - All 4 services should use `DefaultEmbeddingFunction()` (already appears to be the case, but verify no overrides)

### Phase 5: Validate End-to-End

12. **Write integration test**: Train → Predict → Verify
    - File: `backend/tests/test_ml_pipeline.py`
    - Train models on a small sample of real DB data
    - Predict on known drug pairs
    - Assert feature dimensions match, predictions are non-trivial, severity thresholds work

13. **Write RAG test**: Index → Query → Verify
    - File: `backend/tests/test_rag_pipeline.py`
    - Index a known drug document
    - Query for that drug
    - Assert results contain the expected content with reasonable distance scores

14. **Manual verification**:
    - Call `POST /interactions/check` with a known interacting pair (e.g., Metformin + Warfarin)
    - Verify response includes both rule-based AND ml-based results
    - Verify `decision_source` field is populated correctly

---

## Relevant Files

- `backend/app/ml/feature_engineering.py` — Unify feature extraction (canonical source)
- `backend/app/ml/predictor.py` — Remove extract_features_simple(), add caching, load feature_extractor.pkl
- `backend/app/ml/trainer.py` — Implement load_data_from_db(), connect BayesianOptimizer, save optimal threshold
- `backend/app/ml/bayesian_optimizer.py` — Already implemented, just needs to be called
- `backend/app/ml/models.py` — Model definitions (RF, XGBoost, LightGBM, ensemble)
- `backend/app/routers/interactions.py` — Fix silent failure, add ML status to response
- `backend/scripts/train_ddi_models.py` — NEW: training script for DDI models
- `backend/app/prescription/langchain_rag_kb.py` — RAG deduplication, collection audit
- `backend/app/prescription/knowledge_base.py` — Collection population verification
- `backend/app/prescription/rag_service.py` — Verify collection name consistency
- `backend/tests/test_ml_pipeline.py` — NEW: ML integration test
- `backend/tests/test_rag_pipeline.py` — NEW: RAG integration test

## Verification

1. Run `python backend/scripts/train_ddi_models.py` — models train without errors, `training_results.json` shows optimization history, models saved to `backend/models/`
2. Run `pytest backend/tests/test_ml_pipeline.py` — feature dimensions match between train and predict, predictions are non-trivial (not all same class)
3. Run `pytest backend/tests/test_rag_pipeline.py` — documents indexed and retrieved with correct metadata
4. Call `POST /interactions/check {"drug1_name": "Metformin", "drug2_name": "Warfarin"}` — response has `ml_probability`, `ml_severity`, `decision_source` all populated (not null)
5. Check `backend/models/optimal_threshold.json` exists after training
6. Check `backend/models/training_results.json` contains `optimization_results` with trial history from Optuna

## Decisions

- Feature extraction will be unified around the Trainer's `DrugFeatureExtractor` (TF-IDF + categorical + statistical features) — NOT the predictor's hash-based approach, which is semantically meaningless
- Rules always override ML for contraindicated/major severity (existing behavior, preserved)
- ChromaDB collections remain separate for now (prescriptions need per-prescription isolation), but drug_knowledge and web_drug_knowledge should be merged
- Training uses real database data, not synthetic — synthetic data approach only acceptable for diabetic risk model

## Further Considerations

1. **Model retraining schedule**: Should training be triggered manually only, or on a periodic schedule (e.g., weekly cron)? Recommendation: Manual for now via admin endpoint, add scheduling later.
2. **Feature set expansion**: Current features are drug-metadata-based. Consider adding pharmacological features (CYP enzyme interactions, protein binding %) from external APIs for better ML accuracy.
3. **Embedding model upgrade**: Current all-MiniLM-L6-v2 (384D) is fast but not medical-domain specific. Consider PubMedBERT or BioSentVec for better medical similarity. Recommendation: Keep current model for now, benchmark alternatives later.
