"""Database setup and configuration."""

from sqlalchemy import create_engine, Column, String, Boolean, DateTime, Integer
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime, date
import os

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
    is_verified = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_login = Column(DateTime, nullable=True)
    # Token system: users get 10 tokens per day
    tokens_remaining = Column(Integer, default=10, nullable=False)
    last_token_reset = Column(DateTime, default=datetime.utcnow, nullable=False)


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
    try:
        # Use checkfirst=True to avoid errors if tables already exist
        Base.metadata.create_all(bind=engine, checkfirst=True)
        logging.info("Database tables initialized successfully")
    except Exception as e:
        # Log error - this is important for debugging
        logging.error(f"Database initialization error: {str(e)}")
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
    import logging
    try:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        try:
            last_reset = user.last_token_reset
            logging.info(f"[RESET_TOKENS] last_reset: {last_reset}, user.tokens_remaining: {user.tokens_remaining}")
        except Exception as e:
            logging.error(f"[RESET_TOKENS] *** ERROR accessing user.last_token_reset or user.tokens_remaining: {str(e)} ***")
            if "invalid state" in str(e).lower() or "detached" in str(e).lower():
                logging.error(f"[RESET_TOKENS] *** INVALID STATE ERROR DETECTED when accessing user attributes ***")
            raise
        
        # Check if it's a new day (compare dates, not times)
        if last_reset:
            # Make both timezone-aware for comparison
            if last_reset.tzinfo is None:
                last_reset = last_reset.replace(tzinfo=timezone.utc)
            
            # Reset if it's a different day
            if last_reset.date() < now.date():
                logging.info(f"[RESET_TOKENS] New day detected, resetting tokens from {user.tokens_remaining} to 10")
                user.tokens_remaining = 10
                user.last_token_reset = now
                logging.info(f"[RESET_TOKENS] Tokens reset complete, user.tokens_remaining: {user.tokens_remaining}")
                # Don't commit here - let the caller commit after all changes
    except Exception as e:
        logging.error(f"[RESET_TOKENS] *** ERROR in reset_daily_tokens_if_needed: {str(e)} ***")
        if "invalid state" in str(e).lower() or "detached" in str(e).lower():
            logging.error(f"[RESET_TOKENS] *** INVALID STATE ERROR DETECTED in reset_daily_tokens_if_needed ***")
        import traceback
        logging.error(f"[RESET_TOKENS] Traceback: {traceback.format_exc()}")
        raise


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

