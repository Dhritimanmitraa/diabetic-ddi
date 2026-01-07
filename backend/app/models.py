"""Database models for drugs and interactions."""

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    Float,
    ForeignKey,
    Table,
    Boolean,
    DateTime,
)
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database import Base


# Association table for drug categories
drug_categories = Table(
    "drug_categories",
    Base.metadata,
    Column("drug_id", Integer, ForeignKey("drugs.id"), primary_key=True),
    Column("category_id", Integer, ForeignKey("categories.id"), primary_key=True),
)


class Drug(Base):
    """Drug model storing drug information."""

    __tablename__ = "drugs"

    id = Column(Integer, primary_key=True, index=True)
    drugbank_id = Column(String(20), unique=True, index=True, nullable=True)
    name = Column(String(255), index=True, nullable=False)
    generic_name = Column(String(255), index=True, nullable=True)
    brand_names = Column(Text, nullable=True)  # JSON array of brand names
    description = Column(Text, nullable=True)
    drug_class = Column(String(255), nullable=True)
    mechanism = Column(Text, nullable=True)
    indication = Column(Text, nullable=True)
    pharmacology = Column(Text, nullable=True)

    # Chemical properties
    molecular_formula = Column(String(100), nullable=True)
    molecular_weight = Column(Float, nullable=True)
    smiles = Column(Text, nullable=True)

    # Status
    is_approved = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    categories = relationship(
        "Category", secondary=drug_categories, back_populates="drugs"
    )
    interactions_as_drug1 = relationship(
        "DrugInteraction",
        foreign_keys="DrugInteraction.drug1_id",
        back_populates="drug1",
    )
    interactions_as_drug2 = relationship(
        "DrugInteraction",
        foreign_keys="DrugInteraction.drug2_id",
        back_populates="drug2",
    )

    def __repr__(self):
        return f"<Drug(id={self.id}, name='{self.name}')>"


class Category(Base):
    """Drug category/therapeutic class."""

    __tablename__ = "categories"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, index=True, nullable=False)
    description = Column(Text, nullable=True)

    # Relationships
    drugs = relationship("Drug", secondary=drug_categories, back_populates="categories")

    def __repr__(self):
        return f"<Category(id={self.id}, name='{self.name}')>"


class DrugInteraction(Base):
    """Drug-Drug Interaction model."""

    __tablename__ = "drug_interactions"

    id = Column(Integer, primary_key=True, index=True)
    drug1_id = Column(Integer, ForeignKey("drugs.id"), nullable=False, index=True)
    drug2_id = Column(Integer, ForeignKey("drugs.id"), nullable=False, index=True)

    # Interaction details
    severity = Column(
        String(50), nullable=False, default="moderate"
    )  # minor, moderate, major, contraindicated
    description = Column(Text, nullable=True)
    effect = Column(Text, nullable=True)
    mechanism = Column(Text, nullable=True)
    management = Column(Text, nullable=True)

    # Source tracking
    source = Column(String(100), nullable=True)  # drugbank, twosides, openfda, etc.
    evidence_level = Column(
        String(50), nullable=True
    )  # established, theoretical, case_report

    # Confidence score (0-1)
    confidence_score = Column(Float, default=0.8)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    drug1 = relationship(
        "Drug", foreign_keys=[drug1_id], back_populates="interactions_as_drug1"
    )
    drug2 = relationship(
        "Drug", foreign_keys=[drug2_id], back_populates="interactions_as_drug2"
    )

    def __repr__(self):
        return f"<DrugInteraction(drug1_id={self.drug1_id}, drug2_id={self.drug2_id}, severity='{self.severity}')>"


class DrugSimilarity(Base):
    """Drug similarity scores for alternative suggestions."""

    __tablename__ = "drug_similarities"

    id = Column(Integer, primary_key=True, index=True)
    drug1_id = Column(Integer, ForeignKey("drugs.id"), nullable=False, index=True)
    drug2_id = Column(Integer, ForeignKey("drugs.id"), nullable=False, index=True)

    # Similarity metrics
    structural_similarity = Column(Float, default=0.0)  # Based on chemical structure
    therapeutic_similarity = Column(
        Float, default=0.0
    )  # Based on drug class/indication
    overall_similarity = Column(Float, default=0.0)

    # Relationships
    source_drug = relationship("Drug", foreign_keys=[drug1_id])
    similar_drug = relationship("Drug", foreign_keys=[drug2_id])


class ComparisonLog(Base):
    """Log of all drug comparison queries made by users."""

    __tablename__ = "comparison_logs"

    id = Column(Integer, primary_key=True, index=True)

    # Drug information
    drug1_name = Column(String(255), nullable=False, index=True)
    drug2_name = Column(String(255), nullable=False, index=True)
    drug1_id = Column(Integer, ForeignKey("drugs.id"), nullable=True)
    drug2_id = Column(Integer, ForeignKey("drugs.id"), nullable=True)

    # Result
    has_interaction = Column(Boolean, default=False)
    is_safe = Column(Boolean, default=True)
    severity = Column(
        String(50), nullable=True
    )  # minor, moderate, major, contraindicated

    # Additional info
    effect = Column(Text, nullable=True)
    safety_message = Column(Text, nullable=True)

    # ML Decision Audit Fields
    ml_probability = Column(Float, nullable=True)  # ML interaction probability
    ml_severity = Column(String(50), nullable=True)  # ML predicted severity
    ml_decision_source = Column(
        String(50), nullable=True
    )  # ml_primary, rule_override, rules_only
    ml_model_version = Column(String(100), nullable=True)  # Model version used
    rule_override_reason = Column(
        Text, nullable=True
    )  # Why rule overrode ML (if applicable)

    # Metadata
    ip_address = Column(String(50), nullable=True)
    user_agent = Column(String(500), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    # Relationships
    drug1 = relationship("Drug", foreign_keys=[drug1_id])
    drug2 = relationship("Drug", foreign_keys=[drug2_id])

    def __repr__(self):
        status = "INTERACTION" if self.has_interaction else "SAFE"
        return f"<ComparisonLog({self.drug1_name} + {self.drug2_name} = {status})>"


class MLPrediction(Base):
    """Log of ML model predictions for analysis."""

    __tablename__ = "ml_predictions"

    id = Column(Integer, primary_key=True, index=True)

    # Drug information
    drug1_name = Column(String(255), nullable=False, index=True)
    drug2_name = Column(String(255), nullable=False, index=True)
    drug1_id = Column(Integer, ForeignKey("drugs.id"), nullable=True)
    drug2_id = Column(Integer, ForeignKey("drugs.id"), nullable=True)

    # Prediction results
    interaction_probability = Column(Float, nullable=False)
    predicted_interaction = Column(Boolean, default=False)
    severity_prediction = Column(String(50), nullable=True)
    confidence = Column(Float, nullable=True)

    # Individual model predictions (JSON)
    model_predictions = Column(Text, nullable=True)  # JSON: {"random_forest": 0.8, ...}

    # Comparison with actual result (if available)
    actual_interaction = Column(Boolean, nullable=True)
    prediction_correct = Column(Boolean, nullable=True)

    # Metadata
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    def __repr__(self):
        return f"<MLPrediction({self.drug1_name} + {self.drug2_name} = {self.interaction_probability:.2f})>"


class ModelMetrics(Base):
    """Track model performance metrics over time."""

    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, index=True)

    # Model identification
    model_type = Column(
        String(50), nullable=False, index=True
    )  # random_forest, xgboost, lightgbm
    model_version = Column(String(50), nullable=True)

    # Performance metrics
    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    auc_roc = Column(Float, nullable=True)

    # Training info
    training_samples = Column(Integer, nullable=True)
    test_samples = Column(Integer, nullable=True)
    n_features = Column(Integer, nullable=True)

    # Hyperparameters (JSON)
    hyperparameters = Column(Text, nullable=True)

    # Timestamps
    trained_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<ModelMetrics({self.model_type}, AUC={self.auc_roc:.4f})>"


class OptimizationResult(Base):
    """Store hyperparameter optimization results."""

    __tablename__ = "optimization_results"

    id = Column(Integer, primary_key=True, index=True)

    # Model and method
    model_type = Column(String(50), nullable=False, index=True)
    optimization_method = Column(
        String(50), nullable=False
    )  # bayesian, grid_search, random_search

    # Results
    best_score = Column(Float, nullable=False)
    best_params = Column(Text, nullable=True)  # JSON

    # Efficiency metrics
    n_trials = Column(Integer, nullable=True)
    total_time_seconds = Column(Float, nullable=True)

    # Full trial history (JSON)
    trial_history = Column(Text, nullable=True)

    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<OptimizationResult({self.model_type}, {self.optimization_method}, score={self.best_score:.4f})>"


class TwosidesInteraction(Base):
    """TWOSIDES/OffSIDES mined interaction/effect record."""

    __tablename__ = "twosides_interactions"

    id = Column(Integer, primary_key=True, index=True)
    drug1_name = Column(String(255), nullable=False, index=True)
    drug2_name = Column(String(255), nullable=False, index=True)
    effect = Column(Text, nullable=True)  # reported effect / outcome
    severity = Column(
        String(50), nullable=True, index=True
    )  # mapped severity (fatal/contra/major/moderate/minor)
    source = Column(String(100), default="twosides")  # twosides or offsides
    evidence = Column(Text, nullable=True)  # optional evidence text / score
    raw_row = Column(Text, nullable=True)  # JSON of original row
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def __repr__(self):
        return f"<TwosidesInteraction({self.drug1_name}+{self.drug2_name} effect={self.effect})>"


class OffsidesEffect(Base):
    """OffSIDES mined single-drug side-effect signal."""

    __tablename__ = "offsides_effects"

    id = Column(Integer, primary_key=True, index=True)
    drug_name = Column(String(255), nullable=False, index=True)
    effect = Column(Text, nullable=True)
    severity = Column(
        String(50), nullable=True, index=True
    )  # optional mapped severity if we choose to classify
    source = Column(String(100), default="offsides")
    evidence = Column(Text, nullable=True)
    raw_row = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def __repr__(self):
        return f"<OffsidesEffect({self.drug_name} effect={self.effect})>"
