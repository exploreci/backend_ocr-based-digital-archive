from sqlalchemy import Column, String, DateTime, ForeignKey
from datetime import datetime, timedelta
import secrets

class PasswordResetToken(Base):
    __tablename__ = "password_reset_tokens"
    
    email = Column(String(100), primary_key=True)
    token = Column(String(100), unique=True, nullable=False)
    expires_at = Column(DateTime, nullable=False)

    def is_expired(self):
        return datetime.utcnow() > self.expires_at