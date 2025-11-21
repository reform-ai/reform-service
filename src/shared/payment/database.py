"""Database models for payment and token system."""

from sqlalchemy import Column, String, Boolean, DateTime, Integer, ForeignKey, Numeric, CheckConstraint, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

# Import Base from auth database to use the same declarative base
from src.shared.auth.database import Base


class TokenTransaction(Base):
    """Token transaction model - tracks all token credits and debits."""
    __tablename__ = "token_transactions"

    id = Column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    token_type = Column(String, nullable=False)  # 'free' or 'purchased'
    amount = Column(Integer, nullable=False)  # Positive for credits, negative for debits
    source = Column(String, nullable=False)  # e.g., 'signup_bonus', 'stripe_purchase', 'subscription_monthly', 'analysis_usage', 'referral_bonus'
    expires_at = Column(DateTime, nullable=True)  # For free tokens, NULL for purchased tokens
    stripe_payment_intent_id = Column(String, nullable=True)  # Links to Stripe payment (for future use)
    stripe_subscription_id = Column(String, nullable=True)  # For subscription-based tokens (for future use)
    meta_data = Column('metadata', JSONB, nullable=True)  # Flexible storage for additional info (column name is 'metadata' in DB)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    __table_args__ = (
        CheckConstraint("token_type IN ('free', 'purchased')", name="check_token_type"),
        Index('idx_token_transactions_stripe_payment_intent', 'stripe_payment_intent_id', postgresql_where=Column('stripe_payment_intent_id').isnot(None)),
    )


class Subscription(Base):
    """Subscription model - tracks user subscription status."""
    __tablename__ = "subscriptions"

    id = Column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    stripe_subscription_id = Column(String, unique=True, nullable=True)  # Stripe subscription ID (for future use)
    stripe_customer_id = Column(String, nullable=False)  # Stripe customer ID (for future use)
    status = Column(String, nullable=False)  # 'active', 'canceled', 'past_due', 'unpaid', 'trialing'
    tier = Column(String, nullable=False)  # e.g., 'basic', 'pro'
    tokens_per_period = Column(Integer, nullable=False)  # Tokens allocated per billing period
    current_period_start = Column(DateTime, nullable=True)
    current_period_end = Column(DateTime, nullable=True)
    cancel_at_period_end = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (
        CheckConstraint("status IN ('active', 'canceled', 'past_due', 'unpaid', 'trialing')", name="check_subscription_status"),
        Index('idx_subscriptions_stripe_subscription_id', 'stripe_subscription_id', postgresql_where=Column('stripe_subscription_id').isnot(None)),
        Index('idx_subscriptions_status', 'status'),
    )


class Payment(Base):
    """Payment model - audit trail for payments."""
    __tablename__ = "payments"

    id = Column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    stripe_payment_intent_id = Column(String, unique=True, nullable=False)  # Stripe payment intent ID (for future use)
    amount = Column(Numeric(10, 2), nullable=False)  # Amount in dollars/cents
    currency = Column(String(3), default='usd', nullable=False)
    status = Column(String, nullable=False)  # 'succeeded', 'failed', 'pending', 'canceled'
    payment_type = Column(String, nullable=False)  # 'one_time' or 'subscription'
    tokens_granted = Column(Integer, nullable=True)  # Number of tokens granted for this payment
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        CheckConstraint("status IN ('succeeded', 'failed', 'pending', 'canceled')", name="check_payment_status"),
        CheckConstraint("payment_type IN ('one_time', 'subscription')", name="check_payment_type"),
        Index('idx_payments_stripe_payment_intent_id', 'stripe_payment_intent_id'),
        Index('idx_payments_status', 'status'),
    )

