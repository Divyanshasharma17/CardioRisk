# CardioRisk

A production-ready REST API for predicting cardiovascular disease risk from patient health data, built with Python, Flask, and scikit-learn.

---

## Project Structure

```
CardioRisk/
├── app.py               # Flask app — routes and startup
├── model.py             # ML model training, evaluation, prediction
├── data_processing.py   # Data loading, cleaning, preprocessing
├── utils.py             # Input validation, logging helpers
├── database.py          # SQLAlchemy ORM — patient record persistence
├── data/
│   └── cardio_sample.csv
├── templates/
│   └── dashboard.html   # Web dashboard
├── static/
│   └── style.css
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/your-username/CardioRisk.git
cd CardioRisk

python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the API

```bash
python app.py
```

The server starts at `http://localhost:5000`. On first run, the model trains automatically on the sample dataset and saves artifacts (`*.pkl`) to disk.

---

## API Reference

### `GET /health`

Health check endpoint.

**Response**
```json
{ "status": "ok", "service": "CardioRisk API" }
```

---

### `POST /predict`

Predict cardiovascular risk for a patient.

**Request body**

| Field       | Type  | Range     | Description                          |
|-------------|-------|-----------|--------------------------------------|
| age         | int   | 1–120     | Age in years                         |
| gender      | int   | 1–2       | 1 = Female, 2 = Male                 |
| height      | float | 100–250   | Height in cm                         |
| weight      | float | 30–250    | Weight in kg                         |
| ap_hi       | int   | 50–300    | Systolic blood pressure              |
| ap_lo       | int   | 30–200    | Diastolic blood pressure             |
| cholesterol | int   | 1–3       | 1=Normal, 2=Above Normal, 3=High     |
| gluc        | int   | 1–3       | 1=Normal, 2=Above Normal, 3=High     |
| smoke       | int   | 0–1       | Smoker flag                          |
| alco        | int   | 0–1       | Alcohol consumption flag             |
| active      | int   | 0–1       | Physical activity flag               |

**Example request**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 55, "gender": 1, "height": 165, "weight": 80,
    "ap_hi": 140, "ap_lo": 90, "cholesterol": 3, "gluc": 1,
    "smoke": 0, "alco": 0, "active": 1
  }'
```

**Success response**
```json
{
  "status": 200,
  "data": {
    "risk_label": 1,
    "risk_probability": 0.7812,
    "risk_level": "High"
  }
}
```

**Error response (validation)**
```json
{ "error": "Field 'ap_hi' must be between 50 and 300, got 999", "status": 422 }
```

---

### `GET /records`

Returns the last 50 stored prediction records.

---

### `GET /dashboard`

Opens the web dashboard for visualising prediction statistics and history.

---

## Model Details

- Algorithm: Logistic Regression (scikit-learn)
- Preprocessing: StandardScaler (zero mean, unit variance)
- Evaluation metrics: Accuracy, F1-Score, Confusion Matrix
- Artifacts auto-saved as `cardiorisk_model.pkl` and `cardiorisk_scaler.pkl`

---

## Tech Stack

- **Flask** — REST API
- **scikit-learn** — ML model
- **pandas / numpy** — data processing
- **SQLAlchemy + SQLite** — persistence
- **Chart.js** — dashboard visualisations
