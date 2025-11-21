"""Database models for contact form rate limiting."""

from sqlalchemy import Column, String, DateTime, Index
from datetime import datetime

# Import Base from auth database to use the same declarative base
from src.shared.auth.database import Base


class ContactRateLimit(Base):
    """Rate limiting for contact form submissions."""
    __tablename__ = "contact_rate_limits"

    id = Column(String, primary_key=True)  # IP address or email
    identifier_type = Column(String, nullable=False)  # 'ip' or 'email'
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    __table_args__ = (
        Index('idx_contact_rate_limit_identifier_created', 'id', 'identifier_type', 'created_at'),
    )

