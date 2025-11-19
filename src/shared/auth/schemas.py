"""Pydantic schemas for authentication requests and responses."""

from pydantic import BaseModel, EmailStr
from typing import Optional


class SignupRequest(BaseModel):
    """Signup request schema."""
    email: EmailStr
    password: str
    full_name: str


class LoginRequest(BaseModel):
    """Login request schema."""
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    """Token response schema."""
    access_token: str
    token_type: str = "bearer"
    user_id: str
    email: str
    full_name: str


class UserResponse(BaseModel):
    """User response schema."""
    id: str
    email: str
    full_name: str
    is_verified: bool
    created_at: Optional[str] = None
    tokens_remaining: Optional[int] = None  # Number of tokens remaining today


class ChangePasswordRequest(BaseModel):
    """Change password request schema."""
    current_password: str
    new_password: str

