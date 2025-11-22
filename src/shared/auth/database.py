"""Database setup and configuration."""

from sqlalchemy import create_engine, Column, String, Boolean, DateTime, Integer, ForeignKey, Index
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime, date
import os
import uuid

# Load environment variables from .env file (for local development)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, skip (fine for production)

# PostgreSQL database configuration
# DATABASE_URL should be set as an environment variable (e.g., from Heroku)
DATABASE_URL = os.environ.get("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError(
        "DATABASE_URL environment variable is required. "
        "Please set it to your PostgreSQL connection string."
    )

# Heroku uses postgres:// but SQLAlchemy 2.0+ requires postgresql://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# PostgreSQL connection pool configuration
engine_kwargs = {
    "pool_pre_ping": True,  # Verify connections before using
    "pool_recycle": 3600,  # Recycle connections after 1 hour
}

engine = create_engine(DATABASE_URL, **engine_kwargs)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class User(Base):
    """User model for authentication."""
    __tablename__ = "users"

    id = Column(String, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    full_name = Column(String, nullable=False)
    username = Column(String, unique=True, index=True, nullable=True)  # Optional username field
    is_verified = Column(Boolean, default=False, nullable=False)
    is_public = Column(Boolean, default=True, nullable=False)  # Privacy setting for social feed
    is_pt = Column(Boolean, default=False, nullable=False)  # Personal Trainer attribute
    is_admin = Column(Boolean, default=False, nullable=False)  # Admin role for system administration
    technical_level = Column(String, nullable=True)  # beginner, novice, intermediate, advanced, elite
    favorite_exercise = Column(String, nullable=True)  # Favorite exercise (dropdown selection later)
    community_preference = Column(String, nullable=True)  # share_to_similar_levels, share_to_pt, compete_with_someone
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_login = Column(DateTime, nullable=True)
    # Token system: users get 10 tokens per day
    tokens_remaining = Column(Integer, default=10, nullable=False)
    last_token_reset = Column(DateTime, default=datetime.utcnow, nullable=False)
    # Token activation: when user first activates their token system (one-time, lifetime)
    token_activation_date = Column(DateTime, nullable=True)  # Null until user clicks "Get 10 Free Tokens"
    # Payment-related columns (prepared for future Stripe integration)
    stripe_customer_id = Column(String, unique=True, nullable=True)  # Stripe customer ID (for future use)
    subscription_status = Column(String, nullable=True)  # 'none', 'active', 'canceled', 'past_due' (for future use)
    subscription_tier = Column(String, nullable=True)  # e.g., 'basic', 'pro' (for future use)


class EmailVerificationToken(Base):
    """Email verification token model for account verification."""
    __tablename__ = "email_verification_tokens"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    token = Column(String, unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=False, index=True)
    used_at = Column(DateTime, nullable=True)  # Set when token is used for verification

    __table_args__ = (
        Index('idx_email_verification_tokens_user_id', 'user_id'),
        Index('idx_email_verification_tokens_token', 'token'),
        Index('idx_email_verification_tokens_expires_at', 'expires_at'),
    )


class AnonymousAnalysis(Base):
    """Track analyses by anonymous (logged-out) users by IP address."""
    __tablename__ = "anonymous_analyses"

    ip_address = Column(String, primary_key=True, index=True)
    analysis_count = Column(Integer, default=0, nullable=False)
    first_analysis_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_analysis_at = Column(DateTime, default=datetime.utcnow, nullable=False)


def init_db():
    """Initialize database tables."""
    import logging
    from sqlalchemy.exc import IntegrityError, ProgrammingError
    
    try:
        # Use checkfirst=True to avoid errors if tables already exist
        Base.metadata.create_all(bind=engine, checkfirst=True)
        logging.info("Database tables initialized successfully")
    except (IntegrityError, ProgrammingError) as e:
        # Ignore duplicate type/constraint errors - these happen when types already exist
        # This is safe because checkfirst=True should prevent duplicate table creation
        error_str = str(e)
        if "pg_type_typname_nsp_index" in error_str or "duplicate key" in error_str.lower():
            logging.info("Database types already exist, skipping type creation (safe to ignore)")
        else:
            logging.warning(f"Database integrity/programming error (may be safe to ignore): {error_str}")
    except Exception as e:
        # Log error - this is important for debugging
        error_str = str(e)
        if "pg_type_typname_nsp_index" in error_str or "duplicate key" in error_str.lower():
            logging.info("Database types already exist, skipping type creation (safe to ignore)")
        else:
            logging.error(f"Database initialization error: {error_str}")
        # Don't raise - let the fallback in routes handle it
        # This allows the app to start even if DB init fails


def get_db():
    """Dependency to get database session."""
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def reset_daily_tokens_if_needed(user: User, db) -> None:
    """Reset user's tokens to 10 if it's a new day. Does NOT commit - caller must commit."""
    try:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        try:
            last_reset = user.last_token_reset
        except Exception as e:
            raise
        
        # Check if it's a new day (compare dates, not times)
        if last_reset:
            # Make both timezone-aware for comparison
            if last_reset.tzinfo is None:
                last_reset = last_reset.replace(tzinfo=timezone.utc)
            
            # Reset if it's a different day
            if last_reset.date() < now.date():
                user.tokens_remaining = 10
                user.last_token_reset = now
                # Don't commit here - let the caller commit after all changes
    except Exception as e:
        import logging
        logging.error(f"Error in reset_daily_tokens_if_needed: {str(e)}")
        raise


def reset_daily_anonymous_limit_if_needed(anonymous_record: AnonymousAnalysis) -> None:
    """Reset anonymous analysis count to 0 if it's a new day. Does NOT commit - caller must commit."""
    try:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        
        if anonymous_record and anonymous_record.last_analysis_at:
            last_analysis = anonymous_record.last_analysis_at
            
            # Make both timezone-aware for comparison
            if last_analysis.tzinfo is None:
                last_analysis = last_analysis.replace(tzinfo=timezone.utc)
            
            # Reset if it's a different day (new day = reset count to 0)
            if last_analysis.date() < now.date():
                anonymous_record.analysis_count = 0
                # Don't commit here - let the caller commit after all changes
    except Exception as e:
        import logging
        logging.error(f"Error in reset_daily_anonymous_limit_if_needed: {str(e)}")
        # Don't raise - allow request to continue even if reset fails


def calculate_token_cost(file_size_bytes: int) -> float:
    """
    Calculate token cost for an analysis.
    Base cost: 1 token
    Additional cost: 0.2 tokens per 50MB of file size
    Recommended: 0.2 tokens per 50MB (so a 500MB file uses 2 additional tokens)
    """
    base_cost = 1.0
    size_mb = file_size_bytes / (1024 * 1024)  # Convert bytes to MB
    additional_cost = (size_mb / 50.0) * 0.2  # 0.2 tokens per 50MB
    return base_cost + additional_cost

