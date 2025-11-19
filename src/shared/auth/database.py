"""Database setup and configuration."""

from sqlalchemy import create_engine, Column, String, Boolean, DateTime, Integer
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import NullPool
from datetime import datetime, date
import os

# Use SQLite for simplicity (can be upgraded to PostgreSQL later)
# On Heroku, use /tmp for SQLite database (ephemeral but writable)
# If DATABASE_URL is set and is PostgreSQL, use that instead
default_db_url = "sqlite:///./reform.db"
if os.environ.get("DYNO"):  # Heroku
    # Use /tmp for SQLite on Heroku (ephemeral filesystem)
    default_db_url = "sqlite:////tmp/reform.db"
elif os.environ.get("DATABASE_URL") and not os.environ.get("DATABASE_URL").startswith("sqlite"):
    # If DATABASE_URL is set and is not SQLite (e.g., PostgreSQL), use it
    default_db_url = os.environ.get("DATABASE_URL")

DATABASE_URL = os.environ.get("DATABASE_URL", default_db_url)

# SQLite-specific connection args
connect_args = {}
if "sqlite" in DATABASE_URL:
    connect_args = {"check_same_thread": False}
    # Ensure directory exists for SQLite
    if DATABASE_URL.startswith("sqlite:///"):
        db_path = DATABASE_URL.replace("sqlite:///", "")
        if db_path != ":memory:":
            db_dir = os.path.dirname(db_path)
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)

# Use pool_pre_ping to check connections before using them (helps with SQLite on Heroku)
# Use pool_recycle for connection recycling (only for non-SQLite databases)
# For SQLite, use a single connection pool to avoid concurrency issues
engine_kwargs = {
    "connect_args": connect_args,
    "pool_pre_ping": True,  # Verify connections before using
}
if "sqlite" in DATABASE_URL:
    # SQLite-specific: use a single connection to avoid "invalid state" errors
    # This prevents concurrent access issues on Heroku's ephemeral filesystem
    engine_kwargs["poolclass"] = None  # Use NullPool for SQLite (single connection)
    engine_kwargs["pool_pre_ping"] = False  # Not needed for single connection
else:
    # Only set pool_recycle for non-SQLite databases (PostgreSQL, etc.)
    engine_kwargs["pool_recycle"] = 3600  # Recycle connections after 1 hour

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
    """Reset user's tokens to 10 if it's a new day."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    last_reset = user.last_token_reset
    
    # Check if it's a new day (compare dates, not times)
    if last_reset:
        # Make both timezone-aware for comparison
        if last_reset.tzinfo is None:
            last_reset = last_reset.replace(tzinfo=timezone.utc)
        
        # Reset if it's a different day
        if last_reset.date() < now.date():
            user.tokens_remaining = 10
            user.last_token_reset = now
            db.commit()


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

