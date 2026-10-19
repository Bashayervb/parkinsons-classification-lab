"""Reproducible educational benchmark; no clinical prediction claims."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
TARGET = 'Diagnosis'
IDENTIFIERS = ['PatientID', 'DoctorInCharge']
ASSESSMENTS_AND_SYMPTOMS = ['UPDRS', 'MoCA', 'FunctionalAssessment', 'Tremor', 'Rigidity',
    'Bradykinesia', 'PosturalInstability', 'SpeechProblems', 'SleepDisorders', 'Constipation']
CATEGORICAL = ['Gender', 'Ethnicity', 'EducationLevel']
ROOT = Path(__file__).resolve().parents[1]


def load_data(path):
    df = pd.read_csv(path)
    required = {TARGET, *IDENTIFIERS, *ASSESSMENTS_AND_SYMPTOMS, *CATEGORICAL}
    missing = required - set(df)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    if df.empty or df[TARGET].isna().any() or set(df[TARGET].unique()) != {0, 1}:
        raise ValueError('Diagnosis must contain both 0 and 1 with no missing labels')
    if df['PatientID'].isna().any() or df['PatientID'].duplicated().any():
        raise ValueError('PatientID must be present and unique; repeated subjects need grouped splitting')
    features = df.drop(columns=[TARGET, *IDENTIFIERS])
    if any(not pd.api.types.is_numeric_dtype(features[c]) for c in features):
        raise ValueError('Expected numeric feature encodings')
    if np.isinf(features.to_numpy(dtype=float)).any():
        raise ValueError('Infinite feature values are not supported')
    if features.duplicated().any():
        raise ValueError('Repeated feature rows require deduplication/grouping before splitting')
    return df


def feature_columns(df, scenario):
    exclude = [TARGET, *IDENTIFIERS]
    if scenario == 'without_assessments_symptoms':
        exclude += ASSESSMENTS_AND_SYMPTOMS
    elif scenario != 'all_non_identifier_features':
        raise ValueError('Unknown scenario')
    return [c for c in df if c not in exclude]


def make_pipeline(columns, model):
    cats = [c for c in CATEGORICAL if c in columns]
    numeric = [c for c in columns if c not in cats]
    preprocess = ColumnTransformer([
        ('numeric', Pipeline([('imputer', SimpleImputer(strategy='median')),
                              ('scale', StandardScaler())]), numeric),
        ('categorical', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
            ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cats)
    ])
    return Pipeline([('preprocess', preprocess), ('model', clone(model))])


def metric_values(y, scores):
    pred = np.asarray(scores) >= 0.5
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return { 'accuracy': float(accuracy_score(y, pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
        'precision': float(precision_score(y, pred, zero_division=0)),
        'recall': float(recall_score(y, pred, zero_division=0)),
        'specificity': float(tn / (tn + fp)),
        'f1': float(f1_score(y, pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y, scores)),
        'average_precision': float(average_precision_score(y, scores)),
        'confusion_matrix': [[int(tn), int(fp)], [int(fn), int(tp)]] }


def run(data_path, output_dir, model_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    df = load_data(data_path)
    y = df[TARGET].astype(int)
    train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.2,
        random_state=SEED, stratify=y)
    assert not set(train_idx).intersection(test_idx)
    # Fixed scenario definitions, split and candidate settings; do not tune on holdout results.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    candidates = {
        'dummy': DummyClassifier(strategy='prior'),
        'logistic_regression': LogisticRegression(max_iter=2000, random_state=SEED),
        'random_forest': RandomForestClassifier(n_estimators=200, max_depth=6,
            min_samples_leaf=5, random_state=SEED, n_jobs=1)
    }
    report = {
        'seed': SEED, 'input_sha256': hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
        'rows': len(df), 'columns': len(df.columns), 'missing_cells': int(df.isna().sum().sum()),
        'diagnosis_counts': {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        'training_rows': len(train_idx), 'test_rows': len(test_idx),
        'training_positive_rate': float(y.iloc[train_idx].mean()),
        'test_positive_rate': float(y.iloc[test_idx].mean()),
        'versions': {'sklearn': sklearn.__version__, 'pandas': pd.__version__, 'numpy': np.__version__},
        'selection_rule': 'Highest mean training-only five-fold average precision; exclude dummy from selection.',
        'threshold': 0.5, 'scenarios': {},
        'interpretation': 'Dataset-label classification only. Publisher identifies the source as synthetic educational data; timing of constructed features is not documented. No clinical validation or early-detection claim.'
    }
    cv_rows, prediction_rows = [], []
    for scenario in ['without_assessments_symptoms', 'all_non_identifier_features']:
        columns = feature_columns(df, scenario)
        Xtrain, Xtest = df.iloc[train_idx][columns], df.iloc[test_idx][columns]
        ytrain, ytest = y.iloc[train_idx], y.iloc[test_idx]
        summary = {}
        for name, model in candidates.items():
            pipe = make_pipeline(columns, model)
            result = cross_validate(pipe, Xtrain, ytrain, cv=cv, n_jobs=1,
                scoring={'ap': 'average_precision', 'roc': 'roc_auc', 'balanced': 'balanced_accuracy'},
                error_score='raise')
            summary[name] = {key: {'mean': float(result['test_' + key].mean()),
                'std': float(result['test_' + key].std())} for key in ['ap', 'roc', 'balanced']}
            for fold in range(5):
                cv_rows.append({'scenario': scenario, 'model': name, 'fold': fold + 1,
                    **{key: float(result['test_' + key][fold]) for key in ['ap', 'roc', 'balanced']}})
        selected = max([n for n in candidates if n != 'dummy'], key=lambda n: summary[n]['ap']['mean'])
        fitted = make_pipeline(columns, candidates[selected]).fit(Xtrain, ytrain)
        scores = fitted.predict_proba(Xtest)[:, 1]
        baseline = make_pipeline(columns, candidates['dummy']).fit(Xtrain, ytrain)
        baseline_scores = baseline.predict_proba(Xtest)[:, 1]
        report['scenarios'][scenario] = {'features': columns, 'feature_count': len(columns),
            'cross_validation': summary, 'selected_model': selected,
            'test': metric_values(ytest, scores), 'baseline_test': metric_values(ytest, baseline_scores)}
        joblib.dump({'pipeline': fitted, 'features': columns, 'scenario': scenario}, model_dir / f'{scenario}.joblib')
        for case, (truth, score) in enumerate(zip(ytest, scores), 1):
            prediction_rows.append({'case': case, 'scenario': scenario, 'dataset_label': int(truth),
                'model_score': float(score), 'predicted_label': int(score >= 0.5)})
    (output_dir / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    pd.DataFrame(cv_rows).to_csv(output_dir / 'cross_validation.csv', index=False)
    pd.DataFrame(prediction_rows).to_csv(output_dir / 'holdout_predictions.csv', index=False)
    render_figures(report, df, output_dir)
    write_summary(report, output_dir)
    print((output_dir / 'summary.md').read_text())
    return report


def render_figures(report, df, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.7))
    counts = df[TARGET].value_counts().sort_index()
    ax.bar(['Dataset label 0', 'Dataset label 1'], counts, color=['#87a6a1', '#24594f'])
    ax.set(ylabel='Records', title='Label distribution in the supplied CSV')
    for i, n in enumerate(counts): ax.text(i, n + 10, str(n), ha='center')
    fig.tight_layout();fig.savefig(output_dir / 'label_distribution.png', dpi=160);plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (scenario, result) in zip(axes, report['scenarios'].items()):
        matrix = np.array(result['test']['confusion_matrix'])
        ax.imshow(matrix, cmap='Greens');ax.set(xticks=[0,1], yticks=[0,1],
            xlabel='Predicted dataset label', ylabel='Reference dataset label',
            title='Without symptoms/assessments' if scenario.startswith('without') else 'All non-identifier features')
        for i in range(2):
            for j in range(2): ax.text(j, i, str(matrix[i,j]), ha='center', va='center', color='white' if matrix[i,j] > matrix.max() * 0.6 else 'black')
    fig.suptitle('Fixed held-out test set — threshold 0.5');fig.tight_layout()
    fig.savefig(output_dir / 'confusion_matrices.png', dpi=160);plt.close(fig)


def write_summary(report, output_dir):
    lines = ['# Experiment results', '',
        f"Input: {report['rows']:,} records; {report['missing_cells']} missing cells. Training: {report['training_rows']:,}; holdout: {report['test_rows']:,}.", '',
        'Models selected using training-only 5-fold average precision. The same holdout was used for two predeclared feature scenarios; it was not used to tune settings. Threshold: 0.5.', '',
        '| Scenario | Selected model | CV AP (mean ± SD) | Holdout AP | Balanced accuracy | Recall | Specificity |',
        '|---|---|---|---|---|---|---|']
    for scenario,r in report['scenarios'].items():
        ap=r['cross_validation'][r['selected_model']]['ap'];t=r['test']
        lines.append(f"| {scenario} | {r['selected_model']} | {ap['mean']:.3f} ± {ap['std']:.3f} | {t['average_precision']:.3f} | {t['balanced_accuracy']:.3f} | {t['recall']:.3f} | {t['specificity']:.3f} |")
    lines += ['',f"Holdout positive-label fraction: {report['test_positive_rate']:.3f}; this is also the constant-score baseline AP.", '',
        '## Limits', 'The source is a user-supplied CSV matching the schema of the linked Kaggle dataset. The publisher identifies this as synthetic educational data, not verified patient observations. Timing and construction of features/labels are not documented in sufficient detail to establish clinical validity. Numeric category codes are retained.',
        'Symptom/assessment features may be closely related to how Diagnosis was assigned. The feature comparison illustrates sensitivity to that choice; it does not establish clinical leakage or early detection. No external validation or clinical utility has been demonstrated.',
        'CV standard deviations describe fold variability, not confidence intervals. Scores are classifier outputs, not calibrated individual clinical risks. No causal effects can be inferred.', '',
        'PatientID, DoctorInCharge and Diagnosis never enter model inputs. Imputation, scaling and encoding are fitted inside each training fold.', '',
        'Generated by `python -m src.train`. Code was prepared with AI assistance and should be reviewed and understood before interview use.']
    (output_dir / 'summary.md').write_text('\n'.join(lines) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'data/parkinsons_disease_data.csv')
    parser.add_argument('--output', type=Path, default=ROOT/'reports')
    parser.add_argument('--models', type=Path, default=ROOT/'models')
    args = parser.parse_args();run(args.data, args.output, args.models)
