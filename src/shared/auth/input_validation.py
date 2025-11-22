"""
Input validation and sanitization utilities.
Protects against SQL injection, XSS, and other injection attacks.
"""

import re
import html
from typing import Optional
from fastapi import HTTPException, status


# Maximum lengths for different input types
MAX_NAME_LENGTH = 100
MAX_EMAIL_LENGTH = 255
MAX_USERNAME_LENGTH = 30
MIN_USERNAME_LENGTH = 3
MAX_NOTES_LENGTH = 1000
MAX_FULL_NAME_LENGTH = 200


def sanitize_text(text: str, max_length: Optional[int] = None, allow_html: bool = False) -> str:
    """
    Sanitize text input to prevent XSS attacks.
    
    Args:
        text: Input text to sanitize
        max_length: Maximum allowed length (None for no limit)
        allow_html: If False, HTML entities are escaped (default: False)
    
    Returns:
        Sanitized text
    """
    if not text:
        return ""
    
    # Strip whitespace
    text = text.strip()
    
    # Escape HTML to prevent XSS (unless HTML is explicitly allowed)
    if not allow_html:
        text = html.escape(text)
    
    # Truncate if too long
    if max_length and len(text) > max_length:
        text = text[:max_length]
    
    return text


def validate_name(name: str, field_name: str = "Name") -> str:
    """
    Validate and sanitize a name field (first name, last name, full name).
    
    Args:
        name: Name to validate
        field_name: Field name for error messages
    
    Returns:
        Sanitized name
    
    Raises:
        HTTPException if validation fails
    """
    if not name or not name.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{field_name} cannot be empty"
        )
    
    # Sanitize and check length
    sanitized = sanitize_text(name, max_length=MAX_NAME_LENGTH)
    
    if len(sanitized) < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{field_name} must be at least 1 character"
        )
    
    if len(sanitized) > MAX_NAME_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{field_name} must be no more than {MAX_NAME_LENGTH} characters"
        )
    
    # Check for potentially dangerous patterns
    # Block script tags, javascript:, data:, etc.
    dangerous_patterns = [
        r'<script',
        r'javascript:',
        r'on\w+\s*=',  # onclick=, onerror=, etc.
        r'data:text/html',
        r'vbscript:',
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, sanitized, re.IGNORECASE):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"{field_name} contains invalid characters"
            )
    
    return sanitized


def validate_email(email: str) -> str:
    """
    Validate email address format and length.
    Note: Pydantic's EmailStr already validates format, but we add length check.
    
    Args:
        email: Email to validate
    
    Returns:
        Normalized email (lowercase)
    
    Raises:
        HTTPException if validation fails
    """
    if not email or not email.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email cannot be empty"
        )
    
    email = email.strip().lower()
    
    if len(email) > MAX_EMAIL_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Email must be no more than {MAX_EMAIL_LENGTH} characters"
        )
    
    # Basic email format check (Pydantic EmailStr does more thorough validation)
    if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid email format"
        )
    
    return email


def validate_username(username: str) -> str:
    """
    Validate and sanitize username.
    Usernames should be alphanumeric with underscores only, lowercase.
    
    Args:
        username: Username to validate
    
    Returns:
        Sanitized username (lowercase)
    
    Raises:
        HTTPException if validation fails
    """
    if not username or not username.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username cannot be empty"
        )
    
    username = username.strip().lower()
    
    if len(username) < MIN_USERNAME_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Username must be at least {MIN_USERNAME_LENGTH} characters"
        )
    
    if len(username) > MAX_USERNAME_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Username must be no more than {MAX_USERNAME_LENGTH} characters"
        )
    
    # Only allow alphanumeric and underscores
    if not re.match(r'^[a-z0-9_]+$', username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username can only contain lowercase letters, numbers, and underscores"
        )
    
    # Check for reserved usernames
    RESERVED_USERNAMES = {
        'admin', 'administrator', 'adm', 'ad',
        'dev', 'developer', 'devops',
        'test', 'testing', 'tester',
        'root', 'system', 'service',
        'api', 'app', 'web',
        'support', 'help', 'info',
        'null', 'undefined', 'none',
        'reform', 'reformgym', 'reformfit'
    }
    
    if username in RESERVED_USERNAMES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This username is reserved and cannot be used"
        )
    
    return username


def validate_notes(notes: Optional[str]) -> Optional[str]:
    """
    Validate and sanitize analysis notes.
    Allows some safe characters but prevents XSS.
    
    Args:
        notes: Notes to validate (can be None)
    
    Returns:
        Sanitized notes or None
    
    Raises:
        HTTPException if validation fails
    """
    if not notes:
        return None
    
    if not isinstance(notes, str):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Notes must be a string"
        )
    
    # Strip whitespace
    notes = notes.strip()
    
    if not notes:
        return None
    
    # Check length
    if len(notes) > MAX_NOTES_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Notes must be no more than {MAX_NOTES_LENGTH} characters"
        )
    
    # Sanitize to prevent XSS - escape HTML
    sanitized = html.escape(notes)
    
    # Check for dangerous patterns even after escaping
    dangerous_patterns = [
        r'javascript:',
        r'data:text/html',
        r'vbscript:',
        r'on\w+\s*=',  # Event handlers
    ]
    
    original_lower = notes.lower()
    for pattern in dangerous_patterns:
        if re.search(pattern, original_lower):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Notes contain invalid content"
            )
    
    return sanitized


def validate_password(password: str) -> str:
    """
    Validate password strength and length.
    
    Args:
        password: Password to validate
    
    Returns:
        Validated password
    
    Raises:
        HTTPException if validation fails
    """
    if not password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password cannot be empty"
        )
    
    if len(password) < 8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 8 characters"
        )
    
    # Bcrypt has a 72-byte limit
    password_bytes = password.encode('utf-8')
    if len(password_bytes) > 72:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password is too long (maximum 72 bytes). Please use a shorter password."
        )
    
    return password


def validate_full_name(first_name: str, last_name: str) -> str:
    """
    Validate and combine first and last name into full name.
    
    Args:
        first_name: First name
        last_name: Last name
    
    Returns:
        Validated full name
    """
    first_name = validate_name(first_name, "First name")
    last_name = validate_name(last_name, "Last name")
    
    full_name = f"{first_name} {last_name}".strip()
    
    if len(full_name) > MAX_FULL_NAME_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Full name must be no more than {MAX_FULL_NAME_LENGTH} characters"
        )
    
    return full_name

