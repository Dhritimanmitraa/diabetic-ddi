"""
Retrain models with imbalanced-learn resampling to improve NPV.

This script uses SMOTE/ADASYN to balance the dataset before training,
which should significantly improve NPV (Negative Predictive Value).
"""
import subprocess
import sys
from pathlib import Path

def main():
    """Run training with different resampling methods."""
    
    script_path = Path(__file__).parent / "train_twosides_ml.py"
    
    print("=" * 80)
    print("RETRAINING WITH IMBALANCED-LEARN")
    print("=" * 80)
    print("\nThis will train models with SMOTE resampling to improve NPV.")
    print("NPV (Negative Predictive Value) is CRITICAL for safety - it tells")
    print("us how reliable 'safe' predictions are.\n")
    
    # Train with SMOTE (most common and effective)
    print("\n" + "=" * 80)
    print("TRAINING WITH SMOTE (50% balance)")
    print("=" * 80)
    
    cmd = [
        sys.executable,
        "-m", "scripts.train_twosides_ml",
        "--resampling", "smote",
        "--sampling-ratio", "0.5",
        "--max-samples", "500000",
        "--n-jobs", "-1"
    ]
    
    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    
    if result.returncode == 0:
        print("\n" + "=" * 80)
        print("✅ TRAINING COMPLETE!")
        print("=" * 80)
        print("\nNext steps:")
        print("1. Check the NPV (Negative Predictive Value) in the results above")
        print("2. If NPV < 70%, try:")
        print("   - python -m scripts.train_twosides_ml --resampling adasyn --sampling-ratio 0.6")
        print("   - python -m scripts.train_twosides_ml --resampling borderline --sampling-ratio 0.5")
        print("3. After training, run:")
        print("   - python -m scripts.find_optimal_threshold")
        print("   - python -m scripts.evaluate_models --find-optimal")
        print("\n⚠️  Remember: NPV should be > 70% for safe predictions!")
    else:
        print("\n❌ Training failed. Check errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()




