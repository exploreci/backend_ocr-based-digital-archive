from fastapi_mail import FastMail, MessageSchema, ConnectionConfig

conf = ConnectionConfig(
    MAIL_USERNAME = "email-kamu@gmail.com",
    MAIL_PASSWORD = "app-password-anda", # 16 digit dari Google
    MAIL_FROM = "email-kamu@gmail.com",
    MAIL_PORT = 587,
    MAIL_SERVER = "smtp.gmail.com",
    MAIL_STARTTLS = True,
    MAIL_SSL_TLS = False,
    USE_CREDENTIALS = True
)

async def send_reset_password_email(email_to: str, token: str):
    # Link ini yang akan diklik user di email
    # Sesuaikan dengan route di Flutter/Web kamu
    reset_link = f"http://yourapp.com/reset-password?token={token}"
    
    html = f"""
    <p>Halo,</p>
    <p>Kami menerima permintaan untuk reset password akun Anda.</p>
    <p>Klik link di bawah ini untuk melanjutkan:</p>
    <a href="{reset_link}">Reset Password Sekarang</a>
    <p>Link ini akan kadaluwarsa dalam 15 menit.</p>
    """

    message = MessageSchema(
        subject="Reset Password Smart Web",
        recipients=[email_to],
        body=html,
        subtype="html"
    )

    fm = FastMail(conf)
    await fm.send_message(message)