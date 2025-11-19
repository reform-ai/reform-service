"""Authentication utilities: password hashing and JWT token generation."""

import bcrypt
from jose import JWTError, jwt
from datetime import datetime, timedelta
import os
import uuid

# JWT settings
SECRET_KEY = os.environ.get("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30 * 24 * 60  # 30 days


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
    """Create a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> dict:
    """Verify and decode a JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None


def generate_user_id() -> str:
    """Generate a unique user ID."""
    return str(uuid.uuid4())

