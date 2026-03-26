"""
database.py
-----------
SQLAlchemy ORM setup. PatientRecord is scoped to a user via user_id FK.
"""

from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker
from werkzeug.security import generate_password_hash, check_password_hash
import os as _os

_DB_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "cardiorisk.db")
DATABASE_URL = f"sqlite:///{_DB_PATH}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    """Registered user account."""

    __tablename__ = "users"

    id            = Column(Integer, primary_key=True, index=True)
    username      = Column(String(80), unique=True, nullable=False)
    email         = Column(String(120), unique=True, nullable=False)
    password_hash = Column(String(256), nullable=False)
    created_at    = Column(DateTime, default=datetime.utcnow)

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

    # Flask-Login required properties
    @property
    def is_authenticated(self): return True

    @property
    def is_active(self): return True

    @property
    def is_anonymous(self): return False

    def get_id(self): return str(self.id)


class PatientRecord(Base):
    """Prediction record — always linked to the user who made it."""

    __tablename__ = "patient_records"

    id            = Column(Integer, primary_key=True, index=True)
    user_id       = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    age           = Column(Integer, nullable=False)
    gender        = Column(Integer, nullable=False)
    height        = Column(Float, nullable=False)
    weight        = Column(Float, nullable=False)
    ap_hi         = Column(Integer, nullable=False)
    ap_lo         = Column(Integer, nullable=False)
    cholesterol   = Column(Integer, nullable=False)
    gluc          = Column(Integer, nullable=False)
    smoke         = Column(Integer, nullable=False)
    alco          = Column(Integer, nullable=False)
    active        = Column(Integer, nullable=False)
    risk_label    = Column(Integer, nullable=False)
    risk_prob     = Column(Float, nullable=False)
    risk_level    = Column(String(20), nullable=False)
    created_at    = Column(DateTime, default=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "id":          self.id,
            "user_id":     self.user_id,
            "age":         self.age,
            "gender":      self.gender,
            "height":      self.height,
            "weight":      self.weight,
            "ap_hi":       self.ap_hi,
            "ap_lo":       self.ap_lo,
            "cholesterol": self.cholesterol,
            "gluc":        self.gluc,
            "smoke":       self.smoke,
            "alco":        self.alco,
            "active":      self.active,
            "risk_label":  self.risk_label,
            "risk_prob":   self.risk_prob,
            "risk_level":  self.risk_level,
            "created_at":  self.created_at.isoformat() if self.created_at else None,
        }


def init_db() -> None:
    """Create all tables if they don't already exist."""
    Base.metadata.create_all(bind=engine)


def save_prediction(db, patient_data: dict, result: dict, user_id: int) -> PatientRecord:
    """
    Persist a prediction result linked to a specific user.

    Args:
        db: Active SQLAlchemy session.
        patient_data: Raw patient input dict.
        result: Prediction result from model.predict_risk().
        user_id: ID of the logged-in user.
    """
    record = PatientRecord(
        user_id=user_id,
        age=patient_data["age"],
        gender=patient_data["gender"],
        height=patient_data["height"],
        weight=patient_data["weight"],
        ap_hi=patient_data["ap_hi"],
        ap_lo=patient_data["ap_lo"],
        cholesterol=patient_data["cholesterol"],
        gluc=patient_data["gluc"],
        smoke=patient_data["smoke"],
        alco=patient_data["alco"],
        active=patient_data["active"],
        risk_label=result["risk_label"],
        risk_prob=result["risk_probability"],
        risk_level=result["risk_level"],
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record
