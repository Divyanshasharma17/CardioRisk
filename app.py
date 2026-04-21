"""
app.py — CardioRisk Flask application
Features: auth, prediction form, PDF export, risk factor breakdown,
          risk trend data, email summary, prediction history page.
"""

import logging
import os
import re
import sys
from dotenv import load_dotenv
load_dotenv()  # loads variables from .env into os.environ

from flask import (Flask, request, jsonify, render_template,
                   redirect, url_for, flash, make_response)
from flask_login import (LoginManager, login_user, logout_user,
                         login_required, current_user)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import (configure_logging, validate_patient_data,
                   build_error_response, build_success_response)
from model import load_artifacts, predict_risk
from data_processing import prepare_patient_input
from database import init_db, SessionLocal, save_prediction, PatientRecord, User
from report import get_risk_factors, generate_pdf_report
from mailer import init_mail, send_prediction_email

# ── Logging ───────────────────────────────────────────────────────────────────
configure_logging()
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=os.path.join(PROJECT_ROOT, "templates"),
    static_folder=os.path.join(PROJECT_ROOT, "static"),
)
app.secret_key = os.environ.get("SECRET_KEY", "cardiorisk-dev-key")

# Mail config from environment variables
app.config["MAIL_USERNAME"] = os.environ.get("MAIL_USERNAME", "")
app.config["MAIL_PASSWORD"] = os.environ.get("MAIL_PASSWORD", "")

# ── Extensions ────────────────────────────────────────────────────────────────
login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message = "Please log in to access this page."
init_mail(app)


@login_manager.user_loader
def load_user(user_id):
    db = SessionLocal()
    user = db.query(User).filter(User.id == int(user_id)).first()
    db.close()
    return user


# ── ML artifacts ──────────────────────────────────────────────────────────────
logger.info("Loading model artifacts...")
try:
    MODEL, SCALER = load_artifacts()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

try:
    init_db()
    logger.info("Database initialized")
except Exception as e:
    logger.error(f"Failed to init DB: {e}")
    raise

logger.info("CardioRisk API ready")


# ── Auth ──────────────────────────────────────────────────────────────────────

@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username         = request.form.get("username", "").strip()
        email            = request.form.get("email", "").strip()
        password         = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not all([username, email, password, confirm_password]):
            flash("All fields are required.", "error")
            return render_template("register.html")
        if len(username) < 3:
            flash("Username must be at least 3 characters.", "error")
            return render_template("register.html")
        if password != confirm_password:
            flash("Passwords do not match.", "error")
            return render_template("register.html")

        pw_ok = (len(password) >= 8 and re.search(r"[A-Z]", password)
                 and re.search(r"[a-z]", password) and re.search(r"\d", password)
                 and re.search(r"[!@#$%^&*]", password))
        if not pw_ok:
            flash("Password must be 8+ chars with uppercase, lowercase, number and special character.", "error")
            return render_template("register.html")

        db = SessionLocal()
        if db.query(User).filter(User.username == username).first():
            flash("Username already taken.", "error")
            db.close()
            return render_template("register.html")
        if db.query(User).filter(User.email == email).first():
            flash("Email already registered.", "error")
            db.close()
            return render_template("register.html")

        user = User(username=username, email=email)
        user.set_password(password)
        db.add(user)
        db.commit()
        db.close()
        flash("Account created! Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        db = SessionLocal()
        user = db.query(User).filter(User.username == username).first()
        db.close()
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for("dashboard"))
        flash("Invalid username or password.", "error")
    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


# ── UI pages ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return redirect(url_for("dashboard"))


@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html", username=current_user.username)


@app.route("/history")
@login_required
def history():
    """Full prediction history page for the current user."""
    db = SessionLocal()
    records = (
        db.query(PatientRecord)
        .filter(PatientRecord.user_id == current_user.id)
        .order_by(PatientRecord.created_at.desc())
        .all()
    )
    db.close()
    return render_template("history.html", records=records, username=current_user.username)


@app.route("/predict-form", methods=["GET", "POST"])
@login_required
def predict_form():
    result = None
    risk_factors = []

    if request.method == "POST":
        try:
            patient_data = {
                "age":         int(request.form["age"]),
                "gender":      int(request.form["gender"]),
                "height":      float(request.form["height"]),
                "weight":      float(request.form["weight"]),
                "ap_hi":       int(request.form["ap_hi"]),
                "ap_lo":       int(request.form["ap_lo"]),
                "cholesterol": int(request.form["cholesterol"]),
                "gluc":        int(request.form["gluc"]),
                "smoke":       int(request.form["smoke"]),
                "alco":        int(request.form["alco"]),
                "active":      int(request.form["active"]),
            }
        except (KeyError, ValueError) as e:
            flash(f"Invalid input: {e}", "error")
            return render_template("predict_form.html", result=None, risk_factors=[])

        is_valid, error_msg = validate_patient_data(patient_data)
        if not is_valid:
            flash(error_msg, "error")
            return render_template("predict_form.html", result=None, risk_factors=[])

        X = prepare_patient_input(patient_data, SCALER)
        result = predict_risk(MODEL, X)
        risk_factors = get_risk_factors(MODEL, SCALER, patient_data)

        # Save to DB
        db = SessionLocal()
        save_prediction(db, patient_data, result, user_id=current_user.id)
        db.close()

        # Generate PDF and save to temp file for download
        pdf_bytes = generate_pdf_report(patient_data, result, risk_factors, current_user.username)
        import tempfile, uuid
        pdf_filename = f"cardiorisk_{current_user.id}_{uuid.uuid4().hex}.pdf"
        pdf_path = os.path.join(tempfile.gettempdir(), pdf_filename)
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)

        # Store just the filename in session (tiny, fits in cookie)
        from flask import session as flask_session
        flask_session["last_pdf_path"] = pdf_path
        logger.info(f"PDF saved to temp: {pdf_path}")

        # Send email (non-blocking, best-effort)
        if app.config.get("MAIL_USERNAME"):
            sent = send_prediction_email(
                to_email=current_user.email,
                username=current_user.username,
                patient_data=patient_data,
                result=result,
                pdf_bytes=pdf_bytes
            )
            if sent:
                flash("Prediction email sent to your inbox.", "success")
                logger.info(f"Email sent to {current_user.email}")
            else:
                logger.warning(f"Email failed for {current_user.email}")
        else:
            logger.info("MAIL_USERNAME not set — skipping email")

    return render_template("predict_form.html", result=result, risk_factors=risk_factors)


@app.route("/download-report")
@login_required
def download_report():
    """Download the most recently generated PDF report."""
    from flask import session as flask_session
    pdf_path = flask_session.get("last_pdf_path")

    if not pdf_path or not os.path.exists(pdf_path):
        flash("No report available. Run a prediction first.", "error")
        return redirect(url_for("predict_form"))

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    response = make_response(pdf_bytes)
    response.headers["Content-Type"] = "application/pdf"
    response.headers["Content-Disposition"] = "attachment; filename=CardioRisk_Report.pdf"
    return response


# ── API ───────────────────────────────────────────────────────────────────────

@app.route("/health")
def health_check():
    return jsonify({"status": "ok", "service": "CardioRisk API"}), 200


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    patient_data = request.get_json(silent=True)
    if patient_data is None:
        return jsonify(build_error_response("Request body must be valid JSON", 400)), 400

    is_valid, error_msg = validate_patient_data(patient_data)
    if not is_valid:
        return jsonify(build_error_response(error_msg, 422)), 422

    try:
        X = prepare_patient_input(patient_data, SCALER)
        result = predict_risk(MODEL, X)
        risk_factors = get_risk_factors(MODEL, SCALER, patient_data)
    except Exception as exc:
        logger.exception("Prediction error")
        return jsonify(build_error_response(str(exc), 500)), 500

    try:
        db = SessionLocal()
        save_prediction(db, patient_data, result, user_id=current_user.id)
        db.close()
    except Exception as exc:
        logger.warning(f"DB write failed: {exc}")

    return jsonify(build_success_response({**result, "risk_factors": risk_factors})), 200


@app.route("/records")
@login_required
def get_records():
    """Current user's last 50 records — used by dashboard charts."""
    try:
        db = SessionLocal()
        records = (
            db.query(PatientRecord)
            .filter(PatientRecord.user_id == current_user.id)
            .order_by(PatientRecord.created_at.desc())
            .limit(50).all()
        )
        db.close()
        return jsonify({"status": 200, "data": [r.to_dict() for r in records]}), 200
    except Exception as exc:
        return jsonify(build_error_response(str(exc), 500)), 500


@app.route("/trend-data")
@login_required
def trend_data():
    """Returns time-series risk probability data for the trend chart."""
    try:
        db = SessionLocal()
        records = (
            db.query(PatientRecord)
            .filter(PatientRecord.user_id == current_user.id)
            .order_by(PatientRecord.created_at.asc())
            .all()
        )
        db.close()
        data = [
            {"date": r.created_at.strftime("%Y-%m-%d %H:%M"), "prob": r.risk_prob,
             "level": r.risk_level}
            for r in records
        ]
        return jsonify({"status": 200, "data": data}), 200
    except Exception as exc:
        return jsonify(build_error_response(str(exc), 500)), 500


# ── Error handlers ────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return jsonify(build_error_response("Endpoint not found", 404)), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify(build_error_response("Method not allowed", 405)), 405


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
else:
    # This runs under gunicorn — catch startup errors explicitly
    import traceback
    try:
        pass  # app is already initialized above
    except Exception as e:
        traceback.print_exc()
