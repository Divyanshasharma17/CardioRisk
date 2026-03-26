"""
mailer.py
---------
Sends prediction summary emails using Flask-Mail.
Configure MAIL_* environment variables before use.
"""

from flask_mail import Mail, Message
import logging

logger = logging.getLogger(__name__)
mail = Mail()


def init_mail(app) -> None:
    """Attach Flask-Mail to the app instance."""
    app.config.setdefault("MAIL_SERVER",   "smtp.gmail.com")
    app.config.setdefault("MAIL_PORT",     587)
    app.config.setdefault("MAIL_USE_TLS",  True)
    app.config.setdefault("MAIL_USERNAME", "")   # set via env var MAIL_USERNAME
    app.config.setdefault("MAIL_PASSWORD", "")   # set via env var MAIL_PASSWORD
    app.config.setdefault("MAIL_DEFAULT_SENDER", app.config.get("MAIL_USERNAME", ""))
    mail.init_app(app)


def send_prediction_email(to_email: str, username: str,
                          patient_data: dict, result: dict,
                          pdf_bytes: bytes) -> bool:
    """
    Send a prediction summary email with the PDF report attached.

    Args:
        to_email: Recipient email address.
        username: User's display name.
        patient_data: Raw patient input dict.
        result: Prediction result dict.
        pdf_bytes: Generated PDF as bytes.

    Returns:
        True if sent successfully, False otherwise.
    """
    risk_emoji = {"High": "🔴", "Moderate": "🟡", "Low": "🟢"}.get(result["risk_level"], "")

    subject = f"CardioRisk Report — {risk_emoji} {result['risk_level']} Risk"

    html_body = f"""
    <div style="font-family: Arial, sans-serif; max-width: 600px; margin: auto;">
      <div style="background: linear-gradient(135deg,#c0392b,#e74c3c);
                  padding: 24px; text-align: center; border-radius: 8px 8px 0 0;">
        <h1 style="color:white; margin:0;">❤ CardioRisk</h1>
        <p style="color:rgba(255,255,255,0.85); margin:4px 0 0;">Prediction Report</p>
      </div>

      <div style="background:#fff; padding:24px; border:1px solid #e2e8f0;
                  border-top:none; border-radius:0 0 8px 8px;">
        <p>Hi <b>{username}</b>,</p>
        <p>Your cardiovascular risk prediction is ready. Here's a summary:</p>

        <table style="width:100%; border-collapse:collapse; margin:16px 0;">
          <tr style="background:#f7fafc;">
            <td style="padding:10px; border:1px solid #e2e8f0;"><b>Risk Level</b></td>
            <td style="padding:10px; border:1px solid #e2e8f0;">{risk_emoji} {result['risk_level']}</td>
          </tr>
          <tr>
            <td style="padding:10px; border:1px solid #e2e8f0;"><b>Probability</b></td>
            <td style="padding:10px; border:1px solid #e2e8f0;">{result['risk_probability']*100:.1f}%</td>
          </tr>
          <tr style="background:#f7fafc;">
            <td style="padding:10px; border:1px solid #e2e8f0;"><b>Diagnosis</b></td>
            <td style="padding:10px; border:1px solid #e2e8f0;">
              {"Cardiovascular disease risk detected" if result['risk_label'] == 1 else "No significant risk detected"}
            </td>
          </tr>
        </table>

        <p>The full detailed report with risk factor breakdown is attached as a PDF.</p>

        <p style="color:#718096; font-size:12px; margin-top:24px;">
          This is an automated message from CardioRisk. This report is for
          informational purposes only — please consult a qualified medical
          professional for diagnosis and treatment.
        </p>
      </div>
    </div>
    """

    try:
        msg = Message(subject=subject, recipients=[to_email], html=html_body)
        msg.attach(
            filename="CardioRisk_Report.pdf",
            content_type="application/pdf",
            data=pdf_bytes
        )
        mail.send(msg)
        logger.info(f"✅ Email sent successfully to {to_email}")
        return True
    except Exception as exc:
        logger.warning(f"❌ Email failed to {to_email} — reason: {exc}")
        return False
