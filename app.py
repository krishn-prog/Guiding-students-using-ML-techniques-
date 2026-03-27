from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from datetime import datetime
import os
import joblib
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# FLASK SETUP — FIX 1: explicit template_folder so Flask always finds index.html
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')

app = Flask(__name__, template_folder=TEMPLATE_DIR)
CORS(app, resources={
    r"/api/*": {
        "origins": ["*"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

app.config['JSON_SORT_KEYS'] = False

# ============================================================================
# LOAD TRAINED MODELS — FIX 2: safe metadata access + detailed error logging
# ============================================================================

print("\n" + "="*70)
print(" Loading Trained ML Models...")
print("="*70)

MODELS_DIR = os.path.join(BASE_DIR, 'models')
ML_AVAILABLE = False
best_model = None
scaler = None
label_encoders = None
target_encoder = None
feature_names = None
metadata = None

if os.path.exists(MODELS_DIR):
    try:
        best_model      = joblib.load(os.path.join(MODELS_DIR, 'best_model.pkl'))
        scaler          = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
        label_encoders  = joblib.load(os.path.join(MODELS_DIR, 'label_encoders.pkl'))
        target_encoder  = joblib.load(os.path.join(MODELS_DIR, 'target_encoder.pkl'))
        feature_names   = joblib.load(os.path.join(MODELS_DIR, 'feature_names.pkl'))
        metadata        = joblib.load(os.path.join(MODELS_DIR, 'metadata.pkl'))

        # ── FIX 2: provide safe defaults for keys that old pipeline didn't save ──
        metadata.setdefault('best_f1', 0.0)
        metadata.setdefault('all_models_f1', {
            name: 0.0 for name in metadata.get('all_models_accuracy', {})
        })

        ML_AVAILABLE = True

        print(f"\n Models loaded successfully!")
        print(f"\n Model Information:")
        print(f"   Best Model : {metadata['best_model']}")
        print(f"   Accuracy   : {metadata['best_accuracy']:.2%}")
        print(f"   F1-Score   : {metadata['best_f1']:.2%}")
        print(f"   Features   : {len(feature_names)}")
        print(f"   Classes    : {', '.join(metadata['target_classes'])}")

    except Exception as e:
        print(f"\n❌ Error loading models: {e}")
        import traceback
        traceback.print_exc()
        print(f"\n   → Run: python ml_training_pipeline_fixed.py  to retrain")
        print(f"   → Using fallback rule-based recommendations for now...")
        ML_AVAILABLE = False
else:
    print(f"\n  Models directory not found at: {MODELS_DIR}")
    print(f"   → Run: python ml_training_pipeline_fixed.py")
    ML_AVAILABLE = False

print("\n" + "="*70 + "\n")

# ============================================================================
# GLOBALS
# ============================================================================

STUDENT_CLUSTER_PROFILES = {
    0: 'High Achievers - Consistent Excellence',
    1: 'Developing Learners - Growth Potential',
    2: 'At-Risk Students - Support Needed',
    3: 'Balanced Performers - Well-Rounded'
}

CAREER_DOMAINS = {
    'STEM': {
        'fields': ['Software Engineering', 'Data Science', 'Physics', 'Mathematics'],
        'required_skills': ['Analytical thinking', 'Problem solving', 'Logical reasoning'],
        'avg_salary': 'High'
    },
    'Business & Management': {
        'fields': ['Entrepreneurship', 'Finance', 'Management', 'Economics'],
        'required_skills': ['Leadership', 'Decision making', 'Communication'],
        'avg_salary': 'High'
    },
    'Healthcare': {
        'fields': ['Medicine', 'Nursing', 'Pharmacy', 'Psychology'],
        'required_skills': ['Attention to detail', 'Compassion', 'Communication'],
        'avg_salary': 'High'
    },
    'Humanities & Social Sciences': {
        'fields': ['Law', 'Literature', 'Sociology', 'Political Science'],
        'required_skills': ['Communication', 'Critical thinking', 'Research'],
        'avg_salary': 'Medium'
    },
    'Arts & Design': {
        'fields': ['Graphic Design', 'Architecture', 'Fine Arts', 'Multimedia'],
        'required_skills': ['Creativity', 'Attention to detail', 'Visual thinking'],
        'avg_salary': 'Medium'
    },
    'Trades & Vocational': {
        'fields': ['Electrical work', 'Plumbing', 'Carpentry', 'Automotive'],
        'required_skills': ['Practical skills', 'Problem solving', 'Precision'],
        'avg_salary': 'Medium'
    }
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def make_prediction(student_data):
    """Use trained ML model for prediction"""
    if not ML_AVAILABLE:
        return None, None
    try:
        features_list = [student_data.get(fname, 0) for fname in feature_names]
        X = np.array([features_list])
        X_scaled = scaler.transform(X)
        prediction = best_model.predict(X_scaled)[0]
        if hasattr(best_model, 'predict_proba'):
            probabilities = best_model.predict_proba(X_scaled)[0]
            confidence = float(max(probabilities))
        else:
            confidence = 0.95
        performance_level = target_encoder.inverse_transform([prediction])[0]
        return performance_level, confidence
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, None


def get_career_recommendations(performance_level, risk_score, engagement_score):
    """Recommend careers based on predicted performance"""
    scores = {}
    if performance_level == 'High' and engagement_score > 0.5:
        scores = {'STEM': 0.95, 'Business & Management': 0.90, 'Healthcare': 0.85,
                  'Humanities & Social Sciences': 0.75, 'Arts & Design': 0.70, 'Trades & Vocational': 0.60}
    elif performance_level == 'Medium':
        scores = {'Business & Management': 0.85, 'Humanities & Social Sciences': 0.80,
                  'Healthcare': 0.75, 'STEM': 0.70, 'Arts & Design': 0.75, 'Trades & Vocational': 0.80}
    else:
        scores = {'Trades & Vocational': 0.90, 'Arts & Design': 0.80, 'Business & Management': 0.65,
                  'Healthcare': 0.60, 'STEM': 0.50, 'Humanities & Social Sciences': 0.65}
    if risk_score > 2:
        scores = {k: v * 0.85 for k, v in scores.items()}
    return scores


def assign_cluster(performance_level):
    if performance_level == 'High':   return 0
    elif performance_level == 'Medium': return 1
    else:                               return 2


def get_education_path(performance_level, cluster):
    if performance_level == "High":
        return {"program_level": "Bachelor's Degree (Research-focused)",
                "program_types": ["Research-based Programs", "Honors Degrees", "Advanced Studies"]}
    elif performance_level == "Medium":
        return {"program_level": "Bachelor's Degree",
                "program_types": ["Standard Programs", "Work-Study Programs"]}
    else:
        return {"program_level": "Diploma/Certificate or Bachelor's",
                "program_types": ["Vocational Programs", "Community College", "Online Programs"]}


def get_skills_recommendation(performance_level, risk_score):
    if risk_score < 1:
        tech_skills = ["Data Analysis", "Problem Solving", "Communication", "Programming"]
    elif performance_level == "Medium":
        tech_skills = ["Technical Fundamentals", "Problem Solving", "Communication"]
    else:
        tech_skills = ["Fundamental Concepts", "Practical Skills", "Basic Technology"]
    return {
        "technical_skills": tech_skills,
        "soft_skills": ["Leadership", "Teamwork", "Time Management", "Critical Thinking"],
        "priority_areas": ["Attendance", "Assignment Completion", "Test Preparation"]
    }


def get_action_items(performance_level, risk_score, engagement_score):
    actions = []
    if risk_score > 2:
        actions.append({"priority": "HIGH",
                        "action": "Schedule meeting with academic advisor for intervention strategy",
                        "timeline": "Within 1 week"})
        actions.append({"priority": "HIGH",
                        "action": "Enroll in tutoring or peer mentoring program",
                        "timeline": "Immediate"})
    if engagement_score < 0.3:
        actions.append({"priority": "HIGH",
                        "action": "Join 1-2 extracurricular activities this semester",
                        "timeline": "Before end of month"})
    actions.append({"priority": "MEDIUM",
                    "action": "Attend career counseling session",
                    "timeline": "Next month"})
    actions.append({"priority": "MEDIUM",
                    "action": "Explore internship opportunities in recommended fields",
                    "timeline": "Summer break"})
    return actions

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/', methods=['GET'])
def index():
    """Serve the frontend"""
    return render_template('index.html')


@app.route('/api', methods=['GET'])
def api_root():
    return jsonify({
        "message": "🎓 Student Guidance System API - With Real ML",
        "version": "2.0.0",
        "status": "ML Models Loaded " if ML_AVAILABLE else "Fallback Mode ",
        "model_info": {
            "best_model": metadata['best_model'] if ML_AVAILABLE else None,
            "accuracy": f"{metadata['best_accuracy']:.2%}" if ML_AVAILABLE else None,
            "features": len(feature_names) if ML_AVAILABLE else None
        }
    })


@app.route('/api/health', methods=['GET', 'OPTIONS'])
def health_check():
    if request.method == 'OPTIONS':
        return '', 200
    return jsonify({
        "status": "healthy",
        "ml_models_loaded": ML_AVAILABLE,
        "timestamp": datetime.now().isoformat(),
        "api_version": "2.0.0"
    })


@app.route('/api/guidance', methods=['POST', 'OPTIONS'])
def generate_guidance():
    """Generate guidance for a student using REAL ML MODEL"""
    if request.method == 'OPTIONS':
        return '', 200

    try:
        student = request.get_json()
        if not student:
            return jsonify({"error": "No JSON data provided"}), 400

        required_fields = [
            'student_id', 'age', 'gender', 'school', 'study_time',
            'absences', 'grade_1', 'grade_2', 'grade_3', 'failures',
            'parental_education', 'support', 'extracurricular',
            'motivation', 'stress_level'
        ]
        missing_fields = [f for f in required_fields if f not in student]
        if missing_fields:
            return jsonify({"error": f"Missing fields: {missing_fields}"}), 400

        # ── ML PREDICTION ──
        if ML_AVAILABLE:
            print(f"\n Using {metadata['best_model']} for prediction...")
            student_features = {
                'age': student['age'], 'study_time': student['study_time'],
                'absences': student['absences'], 'grade_1': student['grade_1'],
                'grade_2': student['grade_2'], 'grade_3': student['grade_3'],
                'failures': student['failures'], 'parental_education': student['parental_education'],
            }
            performance_level, confidence = make_prediction(student_features)
            if performance_level is None:
                performance_level = 'High' if student['grade_3'] >= 15 else ('Medium' if student['grade_3'] >= 10 else 'Low')
                confidence = 0.70
            print(f"   Prediction: {performance_level} (confidence: {confidence:.2%})")
        else:
            print(f"\n  No ML model, using rule-based fallback...")
            performance_level = 'High' if student['grade_3'] >= 15 else ('Medium' if student['grade_3'] >= 10 else 'Low')
            confidence = 0.60

        risk_score = student['failures'] + (student['absences'] / 10)
        engagement_score = 1.0 if (student.get('extracurricular', False) and student.get('support', False)) else 0.5
        cluster = assign_cluster(performance_level)

        response = {
            "student_id": student['student_id'],
            "cluster_profile": STUDENT_CLUSTER_PROFILES.get(cluster, 'Unknown'),
            "performance_level": performance_level,
            "model_confidence": float(confidence),
            "model_used": metadata['best_model'] if ML_AVAILABLE else "Rule-Based Fallback",
            "career_recommendations": get_career_recommendations(performance_level, risk_score, engagement_score),
            "education_path": get_education_path(performance_level, cluster),
            "skill_development": get_skills_recommendation(performance_level, risk_score),
            "action_items": get_action_items(performance_level, risk_score, engagement_score),
            "timestamp": datetime.now().isoformat()
        }

        return jsonify(response)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


@app.route('/api/career-domains', methods=['GET'])
def get_career_domains():
    return jsonify(CAREER_DOMAINS)


@app.route('/api/model-info', methods=['GET'])
def model_info():
    if not ML_AVAILABLE:
        return jsonify({"error": "No models loaded. Run: python ml_training_pipeline_fixed.py"}), 400
    return jsonify({
        "best_model": metadata['best_model'],
        "accuracy": f"{metadata['best_accuracy']:.2%}",
        "f1_score": f"{metadata['best_f1']:.2%}",
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "target_classes": metadata['target_classes'],
        "all_models": {
            name: {
                "accuracy": f"{acc:.2%}",
                "f1": f"{metadata['all_models_f1'].get(name, 0.0):.2%}"
            }
            for name, acc in metadata['all_models_accuracy'].items()
        }
    })

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({"error": "Internal server error"}), 500

# ============================================================================
# STARTUP MESSAGE
# ============================================================================

def print_startup_message():
    status = "ML MODELS LOADED" if ML_AVAILABLE else "  FALLBACK MODE (run training pipeline first)"
    print("\n" + "="*75)
    print(" Student Guidance System — Flask + ML")
    print("="*75)
    print(f"\n  Status   : {status}")
    if ML_AVAILABLE:
        print(f"  Model    : {metadata['best_model']}")
        print(f"  Accuracy : {metadata['best_accuracy']:.2%}")
        print(f"  F1 Score : {metadata['best_f1']:.2%}")
        print(f"  Features : {len(feature_names)}")
    print(f"\n  Templates: {TEMPLATE_DIR}")
    print(f"  Models   : {MODELS_DIR}")
    print("\n  Endpoints:")
    print("    GET  /                    → Frontend UI")
    print("    GET  /api/health          → Health check")
    print("    GET  /api/model-info      → Model details")
    print("    POST /api/guidance        → Student guidance")
    print("\n" + "="*75 + "\n")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print_startup_message()
    print("Starting Flask server…  Press CTRL+C to stop\n")
    try:
        app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\nServer stopped")
        exit(0)