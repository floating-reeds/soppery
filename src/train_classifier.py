"""
Citation Hallucination Classifier (Guardrail)

Trains a lightweight classifier on mechanistic scores to produce
a real-time trust score for citations.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
import joblib


def load_scores(scores_paths: List[str]) -> pd.DataFrame:
    """Load and combine scores from multiple files."""
    all_scores = []
    
    for path in scores_paths:
        with open(path, 'r') as f:
            scores = json.load(f)
            all_scores.extend(scores)
    
    return pd.DataFrame(all_scores)


def prepare_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract feature matrix and labels from scores dataframe."""
    
    # Filter to only labeled data
    df = df[df['label'].isin(['real', 'fabricated', 'misattributed'])].copy()
    
    # Binary label: 1 = real, 0 = hallucinated (fabricated or misattributed)
    df['label_binary'] = (df['label'] == 'real').astype(int)
    
    # Feature columns
    feature_cols = [
        'ics_mean', 'ics_final',
        'pos_mean', 'pos_final',
        'pfs_mean', 'pfs_final',
        'bas_mean',
    ]
    
    # Add layer-wise variance features if per-layer scores are available
    for score_type in ['ics', 'pos', 'pfs', 'bas']:
        col = f'{score_type}_scores'
        if col in df.columns:
            df[f'{score_type}_std'] = df[col].apply(
                lambda x: np.std(x) if isinstance(x, list) and len(x) > 0 else 0
            )
            df[f'{score_type}_min'] = df[col].apply(
                lambda x: min(x) if isinstance(x, list) and len(x) > 0 else 0
            )
            df[f'{score_type}_max'] = df[col].apply(
                lambda x: max(x) if isinstance(x, list) and len(x) > 0 else 0
            )
            feature_cols.extend([f'{score_type}_std', f'{score_type}_min', f'{score_type}_max'])
    
    # Filter to available columns
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    X = df[feature_cols].values
    y = df['label_binary'].values
    
    # Handle NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return X, y, feature_cols


def train_classifiers(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Dict]:
    """Train and evaluate multiple classifiers."""
    
    results = {}
    
    classifiers = {
        'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    }
    
    for name, clf in classifiers.items():
        print(f"\nTraining {name}...")
        
        # Fit
        clf.fit(X_train, y_train)
        
        # Predict
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, 'predict_proba') else y_pred
        
        # Metrics
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0,
            'model': clf,
        }
        
        print(f"  Accuracy: {results[name]['accuracy']:.4f}")
        print(f"  Precision: {results[name]['precision']:.4f}")
        print(f"  Recall: {results[name]['recall']:.4f}")
        print(f"  F1: {results[name]['f1']:.4f}")
        print(f"  ROC-AUC: {results[name]['roc_auc']:.4f}")
    
    return results


class CitationGuardrail:
    """Real-time trust score for generated citations."""
    
    def __init__(self, model_path: str):
        """Load trained model and scaler."""
        self.model = joblib.load(model_path)
        scaler_path = model_path.replace('.pkl', '_scaler.pkl')
        self.scaler = joblib.load(scaler_path) if Path(scaler_path).exists() else None
        
        # Load feature names
        meta_path = model_path.replace('.pkl', '_meta.json')
        if Path(meta_path).exists():
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                self.feature_names = meta.get('feature_names', [])
        else:
            self.feature_names = []
    
    def compute_trust_score(self, features: Dict[str, float]) -> float:
        """
        Compute probability that citation is REAL.
        Returns trust score in [0, 1].
        """
        # Build feature vector
        if self.feature_names:
            X = np.array([[features.get(f, 0) for f in self.feature_names]])
        else:
            X = np.array([list(features.values())])
        
        # Scale if scaler exists
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        # Predict probability
        proba = self.model.predict_proba(X)[0, 1]
        return float(proba)
    
    def should_flag(self, trust_score: float, threshold: float = 0.5) -> bool:
        """Returns True if citation should be flagged as potentially hallucinated."""
        return trust_score < threshold
    
    def evaluate_citation(self, scores: Dict) -> Dict:
        """Full evaluation of a citation's trustworthiness."""
        trust = self.compute_trust_score(scores)
        return {
            'trust_score': trust,
            'is_flagged': self.should_flag(trust),
            'confidence': abs(trust - 0.5) * 2,  # 0 = uncertain, 1 = confident
            'recommendation': 'LIKELY REAL' if trust > 0.7 else (
                'UNCERTAIN' if trust > 0.3 else 'LIKELY HALLUCINATED'
            )
        }


def main():
    parser = argparse.ArgumentParser(description="Train citation hallucination classifier")
    parser.add_argument("--scores", type=str, nargs='+', required=True, 
                       help="Paths to score JSON files")
    parser.add_argument("--output", type=str, required=True,
                       help="Output directory for trained models")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()
    
    # Load data
    print("Loading scores...")
    df = load_scores(args.scores)
    print(f"Loaded {len(df)} citation scores")
    
    # Prepare features
    X, y, feature_names = prepare_features(df)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Label distribution: {np.bincount(y)}")
    
    if len(X) < 10:
        print("Not enough labeled data for training. Need at least 10 samples.")
        return
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train classifiers
    results = train_classifiers(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Select best model
    best_name = max(results, key=lambda k: results[k]['f1'])
    best_model = results[best_name]['model']
    print(f"\nBest model: {best_name} (F1={results[best_name]['f1']:.4f})")
    
    # Save model
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / "guardrail.pkl"
    scaler_path = output_dir / "guardrail_scaler.pkl"
    meta_path = output_dir / "guardrail_meta.json"
    
    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)
    
    with open(meta_path, 'w') as f:
        json.dump({
            'feature_names': feature_names,
            'best_model': best_name,
            'metrics': {k: {m: v for m, v in r.items() if m != 'model'} 
                       for k, r in results.items()},
        }, f, indent=2)
    
    print(f"\nSaved model to {model_path}")
    print(f"Saved scaler to {scaler_path}")
    print(f"Saved metadata to {meta_path}")
    
    # Feature importance (if available)
    if hasattr(best_model, 'feature_importances_'):
        print("\nFeature Importances:")
        importances = sorted(
            zip(feature_names, best_model.feature_importances_),
            key=lambda x: x[1], reverse=True
        )
        for name, imp in importances:
            print(f"  {name}: {imp:.4f}")


if __name__ == "__main__":
    main()
