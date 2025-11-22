"""Authentication utilities: password hashing and JWT token generation."""

import bcrypt
from jose import JWTError, jwt
from datetime import datetime, timedelta
import os
import uuid

# JWT settings
SECRET_KEY = os.environ.get("SECRET_KEY")
if not SECRET_KEY:
    import logging
    logging.warning(
        "SECRET_KEY environment variable is not set. "
        "JWT token operations will fail. "
        "Please set SECRET_KEY to a secure random string."
    )
ALGORITHM = "HS256"
# Access tokens are short-lived for security (15 minutes)
ACCESS_TOKEN_EXPIRE_MINUTES = 15
# Refresh tokens are long-lived (30 days)
REFRESH_TOKEN_EXPIRE_DAYS = 30


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    # Bcrypt has a 72-byte limit, so we truncate if necessary
    password_bytes = password.encode('utf-8')
    if len(password_bytes) > 72:
        # Truncate to 72 bytes, handling multi-byte characters
        truncated = password_bytes[:72]
        # Remove any incomplete trailing bytes
        while truncated and truncated[-1] & 0x80 and not (truncated[-1] & 0x40):
            truncated = truncated[:-1]
        password_bytes = truncated
    
    # Hash with bcrypt
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    # Bcrypt has a 72-byte limit, so we truncate if necessary
    password_bytes = plain_password.encode('utf-8')
    if len(password_bytes) > 72:
        # Truncate to 72 bytes, handling multi-byte characters
        truncated = password_bytes[:72]
        # Remove any incomplete trailing bytes
        while truncated and truncated[-1] & 0x80 and not (truncated[-1] & 0x40):
            truncated = truncated[:-1]
        password_bytes = truncated
    
    # Verify with bcrypt
    hashed_bytes = hashed_password.encode('utf-8')
    return bcrypt.checkpw(password_bytes, hashed_bytes)


def create_access_token(data: dict, expires_delta: timedelta = None) -> str:
    """Create a JWT access token (short-lived)."""
    if not SECRET_KEY:
        raise ValueError(
            "SECRET_KEY environment variable is required for token creation. "
            "Please set it to a secure random string (e.g., generate with: python -c 'import secrets; print(secrets.token_urlsafe(32))')"
        )
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "type": "access"})  # Mark as access token
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def create_refresh_token(data: dict) -> str:
    """Create a JWT refresh token (long-lived)."""
    if not SECRET_KEY:
        raise ValueError(
            "SECRET_KEY environment variable is required for token creation."
        )
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})  # Mark as refresh token
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str, token_type: str = None) -> dict:
    """
    Verify and decode a JWT token.
    
    Args:
        token: JWT token string
        token_type: Optional token type to verify ("access" or "refresh")
    
    Returns:
        Decoded token payload or None if invalid
    """
    if not SECRET_KEY:
        import logging
        logging.error("SECRET_KEY is not set. Cannot verify token.")
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        # If token_type is specified, verify it matches
        if token_type and payload.get("type") != token_type:
            return None
        return payload
    except JWTError:
        return None


def generate_user_id() -> str:
    """Generate a unique user ID."""
    return str(uuid.uuid4())

