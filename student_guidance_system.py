import pandas as pd
import numpy as np
import os
import joblib
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sys
import matplotlib
matplotlib.use('Agg')          # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_PATH = 'student/student-mat.csv'
MODEL_SAVE_DIR = 'models'
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

def load_data():
    """Load student data from CSV"""
    print("\n" + "="*70)
    print(" STEP 1: LOADING DATA")
    print("="*70)
    
    paths = [
        DATA_PATH,
        './student/student-mat.csv',
        'uploads/student/student-mat.csv',
    ]
    
    df = None
    for path in paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, sep=';')
                print(f"\n Data loaded successfully!")
                print(f"   File: {path}")
                print(f"   Shape: {df.shape[0]} students, {df.shape[1]} features")
                return df
            except Exception as e:
                print(f" Error loading {path}: {e}")
    
    if df is None:
        print(f"\n Could not find data!")
        print(f"\nTried paths: {paths}")
        raise FileNotFoundError(f"Student data not found")
    
    return df

# ============================================================================
# STEP 2: EXPLORE DATA
# ============================================================================

def explore_data(df):
    """Explore the dataset"""
    print("\n" + "="*70)
    print(" STEP 2: DATA EXPLORATION")
    print("="*70)
    
    print(f"\n Dataset Info:")
    print(f"   Rows: {df.shape[0]}")
    print(f"   Columns: {df.shape[1]}")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    print(f"\n Column Types:")
    print(f"   Numeric: {len(numeric_cols)}")
    print(f"   Categorical: {len(categorical_cols)}")
    
    print(f"\n Missing Values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print(f"   None! ")
    else:
        print(missing[missing > 0])
    
    print(f"\n Target Variable (G3 - Final Grade):")
    print(f"   Mean: {df['G3'].mean():.2f}")
    print(f"   Min: {df['G3'].min()}")
    print(f"   Max: {df['G3'].max()}")
    print(f"   Std: {df['G3'].std():.2f}")
    
    return df

# ============================================================================
# STEP 3: PREPROCESS DATA (FIXED)
# ============================================================================

def preprocess_data(df):
    """Clean and preprocess the data - FIXED VERSION"""
    print("\n" + "="*70)
    print(" STEP 3: DATA PREPROCESSING")
    print("="*70)
    
    df_processed = df.copy()
    
    # Handle missing values
    print(f"\n Handling missing values...")
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
    
    for col in numeric_cols:
        if df_processed[col].isnull().sum() > 0:
            df_processed[col].fillna(df_processed[col].median(), inplace=True)
    
    for col in categorical_cols:
        if df_processed[col].isnull().sum() > 0:
            mode_value = df_processed[col].mode()
            if len(mode_value) > 0:
                df_processed[col].fillna(mode_value[0], inplace=True)
    
    # Encode categorical variables
    print(f" Encoding categorical variables...")
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        label_encoders[col] = le
    
    print(f"  Encoded {len(categorical_cols)} columns")
    
    # Remove outliers (IQR method)
    print(f" Removing outliers...")
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    rows_before = len(df_processed)
    
    for col in numeric_cols:
        Q1 = df_processed[col].quantile(0.25)
        Q3 = df_processed[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_processed = df_processed[(df_processed[col] >= lower_bound) & 
                                   (df_processed[col] <= upper_bound)]
    
    rows_removed = rows_before - len(df_processed)
    if rows_removed > 0:
        print(f"  Removed {rows_removed} outliers ({rows_removed/rows_before*100:.1f}%)")
    
    print(f"  Remaining: {len(df_processed)} rows")
    
    return df_processed, label_encoders

# ============================================================================
# STEP 4: FEATURE ENGINEERING
# ============================================================================

def engineer_features(df):
    """Create new meaningful features"""
    print("\n" + "="*70)
    print("  STEP 4: FEATURE ENGINEERING")
    print("="*70)
    
    df_engineered = df.copy()
    features_created = []
    
    if all(col in df_engineered.columns for col in ['G1', 'G2', 'G3']):
        print(f"\n Creating academic features...")
        df_engineered['grade_improvement'] = df_engineered['G3'] - df_engineered['G1']
        features_created.append('grade_improvement')
        df_engineered['grade_consistency'] = 1 - (abs(df_engineered['G2'] - df_engineered['G1']) / 
                                                   (df_engineered['G1'] + 1))
        features_created.append('grade_consistency')
        df_engineered['academic_trend'] = ((df_engineered['G2'] - df_engineered['G1']) + 
                                          (df_engineered['G3'] - df_engineered['G2']))
        features_created.append('academic_trend')
        df_engineered['avg_grade'] = (df_engineered['G1'] + df_engineered['G2'] + 
                                     df_engineered['G3']) / 3
        features_created.append('avg_grade')
    
    if 'studytime' in df_engineered.columns and 'traveltime' in df_engineered.columns:
        print(f" Creating commitment features...")
        df_engineered['study_commitment'] = df_engineered['studytime'] + df_engineered['traveltime']
        features_created.append('study_commitment')
    
    if 'failures' in df_engineered.columns and 'absences' in df_engineered.columns:
        print(f" Creating risk features...")
        df_engineered['risk_score'] = df_engineered['failures'] + (df_engineered['absences'] / 10)
        features_created.append('risk_score')
    
    if 'Medu' in df_engineered.columns and 'Fedu' in df_engineered.columns:
        print(f" Creating socioeconomic features...")
        df_engineered['parental_education'] = (df_engineered['Medu'] + df_engineered['Fedu']) / 2
        features_created.append('parental_education')
    
    print(f"\n  Created {len(features_created)} features: {', '.join(features_created)}")
    
    return df_engineered

# ============================================================================
# STEP 5: PREPARE DATA
# ============================================================================

def prepare_modeling_data(df):
    """Prepare features and target for ML"""
    print("\n" + "="*70)
    print(" STEP 5: PREPARING MODELING DATA")
    print("="*70)
    
    print(f"\n Creating target variable...")
    bins = [0, 10, 15, 20]
    labels = ['Low', 'Medium', 'High']
    df['performance_level'] = pd.cut(df['G3'], bins=bins, labels=labels, include_lowest=True)
    
    target_encoder = LabelEncoder()
    df['performance_encoded'] = target_encoder.fit_transform(df['performance_level'])
    
    print(f"  Classes: {dict(enumerate(target_encoder.classes_))}")
    print(f"  Distribution:")
    for label, count in df['performance_level'].value_counts().items():
        print(f"    {label}: {count} ({count/len(df)*100:.1f}%)")
    
    print(f"\n Selecting features...")
    exclude_cols = ['G3', 'performance_level', 'performance_encoded']
    X = df.drop(exclude_cols, axis=1, errors='ignore')
    y = df['performance_encoded']
    
    print(f"  Total features: {X.shape[1]}")
    print(f"  Total samples: {X.shape[0]}")
    
    feature_names = list(X.columns)
    
    return X, y, target_encoder, feature_names

# ============================================================================
# STEP 6: SPLIT AND SCALE
# ============================================================================

def split_scale_data(X, y):
    """Split data and scale features"""
    print("\n" + "="*70)
    print("✂️  STEP 6: SPLITTING AND SCALING DATA")
    print("="*70)
    
    print(f"\n Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"  Training: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Testing: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    print(f"\n Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler

# ============================================================================
# STEP 7: TRAIN MODELS
# ============================================================================

def train_models(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled):
    """Train multiple models"""
    print("\n" + "="*70)
    print(" STEP 7: TRAINING MODELS")
    print("="*70)
    
    models_info = {}
    
    # Model 1: Random Forest
    print(f"\n Training Random Forest...")
    try:
        rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        models_info['Random Forest'] = {
            'model': rf,
            'accuracy': accuracy_score(y_test, rf_pred),
            'f1': f1_score(y_test, rf_pred, average='weighted'),
            'predictions': rf_pred
        }
        print(f"    Accuracy: {models_info['Random Forest']['accuracy']:.4f} | F1: {models_info['Random Forest']['f1']:.4f}")
    except Exception as e:
        print(f"    Error: {e}")
    
    # Model 2: Gradient Boosting
    print(f"\n Training Gradient Boosting...")
    try:
        gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=RANDOM_STATE)
        gb.fit(X_train, y_train)
        gb_pred = gb.predict(X_test)
        models_info['Gradient Boosting'] = {
            'model': gb,
            'accuracy': accuracy_score(y_test, gb_pred),
            'f1': f1_score(y_test, gb_pred, average='weighted'),
            'predictions': gb_pred
        }
        print(f"    Accuracy: {models_info['Gradient Boosting']['accuracy']:.4f} | F1: {models_info['Gradient Boosting']['f1']:.4f}")
    except Exception as e:
        print(f"    Error: {e}")
    
    # Model 3: Logistic Regression
    print(f"\n Training Logistic Regression...")
    try:
        lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, multi_class='multinomial')
        lr.fit(X_train_scaled, y_train)
        lr_pred = lr.predict(X_test_scaled)
        models_info['Logistic Regression'] = {
            'model': lr,
            'accuracy': accuracy_score(y_test, lr_pred),
            'f1': f1_score(y_test, lr_pred, average='weighted'),
            'predictions': lr_pred
        }
        print(f"    Accuracy: {models_info['Logistic Regression']['accuracy']:.4f} | F1: {models_info['Logistic Regression']['f1']:.4f}")
    except Exception as e:
        print(f"    Error: {e}")
    
    # Model 4: SVM
    print(f"\n  Training Support Vector Machine...")
    try:
        svm = SVC(kernel='rbf', C=100, gamma='scale', probability=True, random_state=RANDOM_STATE)
        svm.fit(X_train_scaled, y_train)
        svm_pred = svm.predict(X_test_scaled)
        models_info['SVM'] = {
            'model': svm,
            'accuracy': accuracy_score(y_test, svm_pred),
            'f1': f1_score(y_test, svm_pred, average='weighted'),
            'predictions': svm_pred
        }
        print(f"    Accuracy: {models_info['SVM']['accuracy']:.4f} | F1: {models_info['SVM']['f1']:.4f}")
    except Exception as e:
        print(f"    Error: {e}")
    
    # Model 5: Voting Ensemble
    print(f"\n Training Voting Ensemble...")
    try:
        if 'Random Forest' in models_info and 'Gradient Boosting' in models_info and 'Logistic Regression' in models_info:
            voting_clf = VotingClassifier(
                estimators=[
                    ('rf', models_info['Random Forest']['model']),
                    ('gb', models_info['Gradient Boosting']['model']),
                    ('lr', models_info['Logistic Regression']['model'])
                ],
                voting='soft'
            )
            voting_clf.fit(X_train, y_train)
            voting_pred = voting_clf.predict(X_test)
            models_info['Voting Ensemble'] = {
                'model': voting_clf,
                'accuracy': accuracy_score(y_test, voting_pred),
                'f1': f1_score(y_test, voting_pred, average='weighted'),
                'predictions': voting_pred
            }
            print(f"    Accuracy: {models_info['Voting Ensemble']['accuracy']:.4f} | F1: {models_info['Voting Ensemble']['f1']:.4f}")
    except Exception as e:
        print(f"    Error: {e}")
    
    if not models_info:
        print("\n No models trained successfully!")
        return None, None
    
    best_model_name = max(models_info.items(), key=lambda x: x[1]['accuracy'])[0]
    print(f"\n🏆 BEST MODEL: {best_model_name}")
    print(f"   Accuracy: {models_info[best_model_name]['accuracy']:.4f}")
    print(f"   F1 Score: {models_info[best_model_name]['f1']:.4f}")
    
    return models_info, best_model_name

# ============================================================================
# STEP 8: SAVE MODELS
# ============================================================================

def save_models(models_info, best_model_name, scaler, target_encoder, label_encoders, feature_names):
    """Save all models and preprocessing objects"""
    print("\n" + "="*70)
    print(" STEP 8: SAVING MODELS")
    print("="*70)
    
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    print(f"\n Created directory: {MODEL_SAVE_DIR}/")
    
    best_model = models_info[best_model_name]['model']
    
    print(f"\n Saving models...")
    
    joblib.dump(best_model, os.path.join(MODEL_SAVE_DIR, 'best_model.pkl'))
    print(f"   models/best_model.pkl")
    
    all_models_dict = {name: info['model'] for name, info in models_info.items()}
    joblib.dump(all_models_dict, os.path.join(MODEL_SAVE_DIR, 'all_models.pkl'))
    print(f"   models/all_models.pkl")
    
    joblib.dump(scaler, os.path.join(MODEL_SAVE_DIR, 'scaler.pkl'))
    print(f"   models/scaler.pkl")
    
    joblib.dump(label_encoders, os.path.join(MODEL_SAVE_DIR, 'label_encoders.pkl'))
    print(f"   models/label_encoders.pkl")
    
    joblib.dump(target_encoder, os.path.join(MODEL_SAVE_DIR, 'target_encoder.pkl'))
    print(f"   models/target_encoder.pkl")
    
    joblib.dump(feature_names, os.path.join(MODEL_SAVE_DIR, 'feature_names.pkl'))
    print(f"   models/feature_names.pkl")
    
    #  FIX: Save metadata WITH f1 scores (were missing before)
    metadata = {
        'best_model': best_model_name,
        'best_accuracy': models_info[best_model_name]['accuracy'],
        'best_f1': models_info[best_model_name]['f1'],          # ← ADDED
        'n_features': len(feature_names),
        'feature_names': feature_names,
        'target_classes': list(target_encoder.classes_),
        'all_models_accuracy': {name: info['accuracy'] for name, info in models_info.items()},
        'all_models_f1': {name: info['f1'] for name, info in models_info.items()}  # ← ADDED
    }
    
    joblib.dump(metadata, os.path.join(MODEL_SAVE_DIR, 'metadata.pkl'))
    print(f"   models/metadata.pkl")
    
    return metadata

# ============================================================================
# STEP 9: GENERATE & SAVE VISUALIZATIONS
# ============================================================================

# ── shared style ─────────────────────────────────────────────────────────────
DARK_BG    = '#0a0e1a'
PANEL_BG   = '#141c30'
GOLD       = '#c9a84c'
GOLD_LIGHT = '#e8c96d'
SILVER     = '#a8b8d8'
SUCCESS    = '#3ecf8e'
DANGER     = '#ff5c7c'
ACCENT     = '#7c8cf8'
WHITE      = '#f0f4ff'
PALETTE    = [GOLD, SUCCESS, ACCENT, DANGER, '#ff8c42']

def _apply_dark_style(fig, axes_list):
    """Apply the shared dark academic theme to a figure."""
    fig.patch.set_facecolor(DARK_BG)
    for ax in axes_list:
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=SILVER, labelsize=10)
        ax.xaxis.label.set_color(SILVER)
        ax.yaxis.label.set_color(SILVER)
        ax.title.set_color(WHITE)
        for spine in ax.spines.values():
            spine.set_edgecolor('#1f2d50')


# ── 1. cluster_distribution.png ──────────────────────────────────────────────

def save_cluster_distribution(df_engineered, target_encoder, save_dir):
    """Bar chart: how many students fall in each performance cluster."""
    print(f"\n   Generating cluster_distribution.png ...")

    try:
        bins   = [0, 10, 15, 20]
        labels = ['Low', 'Medium', 'High']
        perf   = pd.cut(df_engineered['G3'], bins=bins, labels=labels, include_lowest=True)
        counts = perf.value_counts().reindex(labels)
        total  = counts.sum()

        fig, ax = plt.subplots(figsize=(8, 5))
        _apply_dark_style(fig, [ax])

        bar_colors = [DANGER, GOLD, SUCCESS]
        bars = ax.bar(labels, counts.values, color=bar_colors,
                      width=0.55, zorder=3, edgecolor=DARK_BG, linewidth=1.5)

        # value labels on top of each bar
        for bar, count in zip(bars, counts.values):
            pct = count / total * 100
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(counts.values) * 0.02,
                    f'{count}\n({pct:.1f}%)',
                    ha='center', va='bottom', color=WHITE,
                    fontsize=11, fontweight='bold')

        ax.set_ylim(0, max(counts.values) * 1.22)
        ax.set_title('Student Cluster Distribution\nby Performance Level',
                     fontsize=14, fontweight='bold', pad=18, color=WHITE)
        ax.set_xlabel('Performance Level', fontsize=11, labelpad=10)
        ax.set_ylabel('Number of Students',  fontsize=11, labelpad=10)
        ax.yaxis.grid(True, color='#1f2d50', linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)

        # legend patches
        patches = [mpatches.Patch(color=c, label=l)
                   for c, l in zip(bar_colors, labels)]
        ax.legend(handles=patches, facecolor=PANEL_BG, edgecolor='#1f2d50',
                  labelcolor=SILVER, fontsize=10, loc='upper right')

        plt.tight_layout(pad=2)
        path = os.path.join(save_dir, 'cluster_distribution.png')
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
        plt.close(fig)
        print(f"      Saved: {path}")

    except Exception as e:
        print(f"      cluster_distribution.png failed: {e}")


# ── 2. feature_importance.png ────────────────────────────────────────────────

def save_feature_importance(models_info, feature_names, save_dir):
    """Horizontal bar chart: top-15 feature importances from best tree model."""
    print(f"\n   Generating feature_importance.png ...")

    try:
        # pick the best tree-based model that has feature_importances_
        tree_models = {k: v for k, v in models_info.items()
                       if hasattr(v['model'], 'feature_importances_')}
        if not tree_models:
            print("       No tree model with feature_importances_ found, skipping.")
            return

        best_tree_name = max(tree_models, key=lambda k: tree_models[k]['accuracy'])
        importances    = tree_models[best_tree_name]['model'].feature_importances_
        indices        = np.argsort(importances)[::-1][:15]   # top 15
        top_features   = [feature_names[i] for i in indices]
        top_importance = importances[indices]

        fig, ax = plt.subplots(figsize=(9, 6))
        _apply_dark_style(fig, [ax])

        # gradient-like coloring: top features are more gold
        norm_vals  = top_importance / top_importance.max()
        bar_colors = [plt.matplotlib.colors.to_hex(
                          plt.matplotlib.colors.hsv_to_rgb(
                              [0.12, 0.4 + 0.55 * v, 0.95]))
                      for v in norm_vals]

        bars = ax.barh(range(len(top_features)), top_importance,
                       color=bar_colors, edgecolor=DARK_BG,
                       linewidth=1, height=0.68, zorder=3)

        # value annotations
        for bar, val in zip(bars, top_importance):
            ax.text(val + top_importance.max() * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f'{val:.3f}', va='center', ha='left',
                    color=SILVER, fontsize=9)

        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features, fontsize=10)
        ax.invert_yaxis()
        ax.set_title(f'Top 15 Feature Importances\n({best_tree_name})',
                     fontsize=14, fontweight='bold', pad=18, color=WHITE)
        ax.set_xlabel('Importance Score', fontsize=11, labelpad=10)
        ax.xaxis.grid(True, color='#1f2d50', linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)

        plt.tight_layout(pad=2)
        path = os.path.join(save_dir, 'feature_importance.png')
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
        plt.close(fig)
        print(f"      Saved: {path}")

    except Exception as e:
        print(f"      feature_importance.png failed: {e}")


# ── 3. model_comparison.png ──────────────────────────────────────────────────

def save_model_comparison(models_info, save_dir):
    """Grouped bar chart: Accuracy vs F1 for every trained model."""
    print(f"\n   Generating model_comparison.png ...")

    try:
        model_names = list(models_info.keys())
        accuracies  = [models_info[m]['accuracy'] for m in model_names]
        f1_scores   = [models_info[m]['f1']       for m in model_names]

        x      = np.arange(len(model_names))
        width  = 0.36
        # shorten long names for readability
        labels = [n.replace(' & ', '\n& ').replace('Gradient ', 'Gradient\n')
                  for n in model_names]

        fig, ax = plt.subplots(figsize=(10, 6))
        _apply_dark_style(fig, [ax])

        bars_acc = ax.bar(x - width / 2, accuracies, width,
                          label='Accuracy', color=GOLD,
                          edgecolor=DARK_BG, linewidth=1.2, zorder=3)
        bars_f1  = ax.bar(x + width / 2, f1_scores, width,
                          label='F1 Score (weighted)', color=ACCENT,
                          edgecolor=DARK_BG, linewidth=1.2, zorder=3)

        # value labels
        for bar in bars_acc:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.004,
                    f'{bar.get_height():.2%}',
                    ha='center', va='bottom', color=GOLD_LIGHT,
                    fontsize=9, fontweight='bold')

        for bar in bars_f1:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.004,
                    f'{bar.get_height():.2%}',
                    ha='center', va='bottom', color=SILVER,
                    fontsize=9, fontweight='bold')

        # highlight best model
        best_idx = int(np.argmax(accuracies))
        ax.axvspan(best_idx - 0.5, best_idx + 0.5,
                   color=GOLD, alpha=0.06, zorder=0)
        ax.text(best_idx, max(max(accuracies), max(f1_scores)) + 0.04,
                '🏆 Best', ha='center', color=GOLD_LIGHT,
                fontsize=10, fontweight='bold')

        # reference line at 80 %
        ax.axhline(0.80, color='#1f2d50', linewidth=1.2,
                   linestyle='--', zorder=1, label='80% baseline')

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylim(0, min(1.0, max(max(accuracies), max(f1_scores)) + 0.14))
        ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
        ax.set_title('Model Performance Comparison\nAccuracy vs F1 Score',
                     fontsize=14, fontweight='bold', pad=18, color=WHITE)
        ax.set_ylabel('Score', fontsize=11, labelpad=10)
        ax.yaxis.grid(True, color='#1f2d50', linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.legend(facecolor=PANEL_BG, edgecolor='#1f2d50',
                  labelcolor=SILVER, fontsize=10)

        plt.tight_layout(pad=2)
        path = os.path.join(save_dir, 'model_comparison.png')
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
        plt.close(fig)
        print(f"      Saved: {path}")

    except Exception as e:
        print(f"      model_comparison.png failed: {e}")


def generate_visualizations(df_engineered, models_info, feature_names, target_encoder, save_dir):
    """Entry point: generate all three charts and save to save_dir."""
    print("\n" + "="*70)
    print(" STEP 9: GENERATING VISUALIZATIONS")
    print("="*70)

    save_cluster_distribution(df_engineered, target_encoder, save_dir)
    save_feature_importance(models_info, feature_names, save_dir)
    save_model_comparison(models_info, save_dir)

    print(f"\n   All charts saved to: {save_dir}/")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main pipeline execution"""
    print("\n" + "="*70)
    print(" ML TRAINING PIPELINE - FIXED VERSION")
    print("="*70)
    print(f"Training on: {DATA_PATH}\n")
    
    try:
        df = load_data()
        explore_data(df)
        df_processed, label_encoders = preprocess_data(df)
        df_engineered = engineer_features(df_processed)
        X, y, target_encoder, feature_names = prepare_modeling_data(df_engineered)
        X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = split_scale_data(X, y)
        models_info, best_model_name = train_models(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled)
        
        if models_info is None:
            print("\n Training failed!")
            return False
        
        metadata = save_models(models_info, best_model_name, scaler, target_encoder, label_encoders, feature_names)

        # Step 9: charts
        generate_visualizations(df_engineered, models_info, feature_names, target_encoder, MODEL_SAVE_DIR)
        
        print("\n" + "="*70)
        print(" TRAINING COMPLETE!")
        print("="*70)
        print(f"""
 Training Summary:
   • Data: {len(df_engineered)} students
   • Features: {len(feature_names)}
   • Best Model: {best_model_name}
   • Accuracy: {metadata['best_accuracy']:.4f} ({metadata['best_accuracy']*100:.2f}%)
   • F1 Score:  {metadata['best_f1']:.4f} ({metadata['best_f1']*100:.2f}%)
""")
        
        print("All Model Accuracies:\n")
        for name in sorted(models_info.keys(), key=lambda x: models_info[x]['accuracy'], reverse=True):
            acc = models_info[name]['accuracy']
            f1 = models_info[name]['f1']
            print(f"   • {name:.<30} Acc: {acc:.2%}  F1: {f1:.2%}")
        
        print("\n" + "="*70)
        print("Ready to use ML predictions! 🎉")
        print("="*70 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)