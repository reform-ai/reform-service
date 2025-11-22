"""Authentication routes: signup and login endpoints."""

import secrets
import logging
import time
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from datetime import datetime, timedelta, timezone
from typing import List, Dict
from src.shared.auth.database import get_db, User, EmailVerificationToken
from src.shared.auth.auth import (
    hash_password,
    verify_password,
    create_access_token,
    generate_user_id
)
from src.shared.auth.schemas import (
    SignupRequest,
    LoginRequest,
    TokenResponse,
    UserResponse,
    ChangePasswordRequest,
    UpdateUsernameRequest,
    UpdateProfileRequest
)
from src.shared.auth.dependencies import get_current_user
from src.shared.auth.email_utils import send_verification_email

router = APIRouter(prefix="/api/auth", tags=["auth"])

# In-memory rate limiting for verification emails
# Format: {user_id: timestamp_of_last_email}
_verification_email_rate_limit: Dict[str, float] = {}
RATE_LIMIT_SECONDS = 300  # 5 minutes


def generate_verification_token() -> str:
    """Generate a secure random token for email verification."""
    return secrets.token_urlsafe(32)


def create_verification_token(db: Session, user_id: str) -> str:
    """
    Create a new verification token for a user, invalidating old unused tokens.
    Returns the token string.
    """
    import os
    
    # Invalidate all existing unused tokens for this user
    db.query(EmailVerificationToken).filter(
        EmailVerificationToken.user_id == user_id,
        EmailVerificationToken.used_at.is_(None)
    ).delete()
    
    # Generate new token
    token = generate_verification_token()
    
    # Calculate expiration (default 1 hour, configurable)
    expiry_hours = int(os.environ.get("VERIFICATION_TOKEN_EXPIRY_HOURS", "1"))
    expires_at = datetime.now(timezone.utc) + timedelta(hours=expiry_hours)
    
    # Create new token record
    verification_token = EmailVerificationToken(
        user_id=user_id,
        token=token,
        expires_at=expires_at
    )
    
    db.add(verification_token)
    db.commit()
    
    return token


def check_rate_limit(user_id: str) -> bool:
    """
    Check if user can send verification email (rate limiting).
    Returns True if allowed, False if rate limited.
    """
    current_time = time.time()
    
    if user_id in _verification_email_rate_limit:
        last_sent = _verification_email_rate_limit[user_id]
        if current_time - last_sent < RATE_LIMIT_SECONDS:
            return False
    
    # Update rate limit
    _verification_email_rate_limit[user_id] = current_time
    return True


@router.post("/signup", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def signup(request: SignupRequest, db: Session = Depends(get_db)):
    """Create a new user account."""
    import logging
    # Ensure database tables exist (fallback if startup init failed)
    try:
        from src.shared.auth.database import Base, engine
        Base.metadata.create_all(bind=engine, checkfirst=True)
    except Exception as e:
        logging.error(f"Database table creation error: {str(e)}")
        # Don't fail silently - if tables can't be created, we have a problem
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database initialization failed. Please try again later."
        )
    
    try:
        # Check if user already exists
        existing_user = db.query(User).filter(User.email == request.email).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Validate password length
        if len(request.password) < 8:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password must be at least 8 characters"
            )
        
        # Bcrypt has a 72-byte limit, warn if password is too long
        password_bytes = request.password.encode('utf-8')
        if len(password_bytes) > 72:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password is too long (maximum 72 bytes). Please use a shorter password."
            )
        
        # Create new user
        user_id = generate_user_id()
        password_hash = hash_password(request.password)
        
        # Combine first_name and last_name into full_name for database storage
        full_name = f"{request.first_name.strip()} {request.last_name.strip()}".strip()

        new_user = User(
            id=user_id,
            email=request.email,
            password_hash=password_hash,
            full_name=full_name,
            is_verified=False,  # Email verification to be implemented later
            created_at=datetime.utcnow(),
            tokens_remaining=10,  # Keep for backward compatibility, but use transaction system
            last_token_reset=datetime.utcnow()
        )
        
        db.add(new_user)
        db.commit()
        # Note: Tokens are NOT granted at signup. User must click "Get 10 Free Tokens" button on profile page.
        # No need to refresh - we have all the values we need
        
        # Send verification email automatically after signup
        try:
            verification_token = create_verification_token(db, user_id)
            email_sent = send_verification_email(
                user_email=request.email,
                user_name=full_name,
                verification_token=verification_token
            )
            if not email_sent:
                logging.warning(f"Failed to send verification email to {request.email} during signup")
        except Exception as e:
            # Don't fail signup if email sending fails - user can request email later
            logging.error(f"Error sending verification email during signup: {str(e)}", exc_info=True)
        
        # Create access token
        access_token = create_access_token(data={"sub": user_id, "email": request.email})
        
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            user_id=user_id,
            email=request.email,
            full_name=full_name
        )
    except HTTPException:
        # Re-raise HTTP exceptions (validation errors, etc.)
        raise
    except Exception as e:
        # Log database errors for debugging
        logging.error(f"Signup error: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create account. Please try again later."
        )


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    """Authenticate user and return access token."""
    import logging
    
    try:
        # Find user by email
        user = db.query(User).filter(User.email == request.email).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        # Verify password
        if not verify_password(request.password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        # Update last login (if column exists)
        try:
            user.last_login = datetime.utcnow()
        except AttributeError:
            # Column doesn't exist yet - skip update
            pass
        db.commit()
        
        # Create access token
        try:
            access_token = create_access_token(data={"sub": user.id, "email": user.email})
        except ValueError as e:
            # SECRET_KEY is missing or invalid
            logging.error(f"Failed to create access token: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Server configuration error. Please contact support."
            )
        
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            user_id=user.id,
            email=user.email,
            full_name=user.full_name
        )
    except HTTPException:
        # Re-raise HTTP exceptions (authentication failures, etc.)
        raise
    except Exception as e:
        # Log unexpected errors for debugging
        logging.error(f"Login error: {str(e)}", exc_info=True)
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed. Please try again later."
        )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get current user information."""
    try:
        # Calculate token balance from transaction system
        from src.shared.payment.token_utils import calculate_token_balance
        balance = calculate_token_balance(db, current_user.id)
        
        return UserResponse(
            id=current_user.id,
            email=current_user.email,
            full_name=current_user.full_name,
            username=current_user.username,
            is_verified=current_user.is_verified,
            is_pt=current_user.is_pt,
            technical_level=getattr(current_user, 'technical_level', None),
            favorite_exercise=getattr(current_user, 'favorite_exercise', None),
            community_preference=getattr(current_user, 'community_preference', None),
            created_at=current_user.created_at.isoformat() if current_user.created_at else None,
            tokens_remaining=balance.total  # Return total from transaction system
        )
    except HTTPException:
        raise
    except Exception as e:
        import logging
        logging.error(f"Error in get_current_user_info: {str(e)}", exc_info=True)
        raise


@router.post("/change-password", status_code=status.HTTP_200_OK)
async def change_password(
    request: ChangePasswordRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Change user password."""
    # Verify current password
    if not verify_password(request.current_password, current_user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Current password is incorrect"
        )
    
    # Validate new password length
    if len(request.new_password) < 8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="New password must be at least 8 characters"
        )
    
    # Check new password is different
    if verify_password(request.new_password, current_user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="New password must be different from current password"
        )
    
    # Bcrypt has a 72-byte limit
    password_bytes = request.new_password.encode('utf-8')
    if len(password_bytes) > 72:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="New password is too long (maximum 72 bytes)"
        )
    
    # Hash and update password
    new_password_hash = hash_password(request.new_password)
    current_user.password_hash = new_password_hash
    db.commit()
    
    return {"message": "Password changed successfully"}


@router.post("/update-username", status_code=status.HTTP_200_OK)
async def update_username(
    request: UpdateUsernameRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update user username."""
    import re
    
    # Reserved usernames that cannot be used
    RESERVED_USERNAMES = {
        'admin', 'administrator', 'adm', 'ad',
        'dev', 'developer', 'devops',
        'test', 'testing', 'tester',
        'root', 'system', 'service',
        'api', 'app', 'web',
        'support', 'help', 'info',
        'null', 'undefined', 'none'
    }
    
    username = request.username.strip().lower()
    
    # Validation: username must be 3-30 characters, alphanumeric and underscores only
    if len(username) < 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username must be at least 3 characters"
        )
    
    if len(username) > 30:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username must be at most 30 characters"
        )
    
    # Check if username contains only alphanumeric characters and underscores
    if not re.match(r'^[a-z0-9_]+$', username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username can only contain lowercase letters, numbers, and underscores"
        )
    
    # Check if username is reserved
    if username in RESERVED_USERNAMES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This username is reserved and cannot be used"
        )
    
    # Check if username is already taken by another user
    existing_user = db.query(User).filter(
        User.username == username,
        User.id != current_user.id
    ).first()
    
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already taken"
        )
    
    # Update username
    current_user.username = username
    db.commit()
    
    return {"message": "Username updated successfully", "username": username}


@router.post("/update-profile", status_code=status.HTTP_200_OK)
async def update_profile(
    request: UpdateProfileRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update user profile attributes (technical level, favorite exercise, community preference)."""
    # Validate technical_level if provided
    if request.technical_level is not None:
        valid_levels = ['beginner', 'novice', 'intermediate', 'advanced', 'elite']
        if request.technical_level not in valid_levels:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid technical level. Must be one of: {', '.join(valid_levels)}"
            )
        current_user.technical_level = request.technical_level
    
    # Validate community_preference if provided
    if request.community_preference is not None:
        valid_preferences = ['share_to_similar_levels', 'share_to_pt', 'compete_with_someone']
        if request.community_preference not in valid_preferences:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid community preference. Must be one of: {', '.join(valid_preferences)}"
            )
        current_user.community_preference = request.community_preference
    
    # Update favorite_exercise if provided (no validation yet, will add dropdown later)
    if request.favorite_exercise is not None:
        current_user.favorite_exercise = request.favorite_exercise.strip() if request.favorite_exercise else None
    
    db.commit()
    
    return {
        "message": "Profile updated successfully",
        "technical_level": current_user.technical_level,
        "favorite_exercise": current_user.favorite_exercise,
        "community_preference": current_user.community_preference
    }


@router.post("/send-verification-email", status_code=status.HTTP_200_OK)
async def send_verification_email_endpoint(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Send verification email to the current user.
    Rate limited to 1 email per 5 minutes per user.
    """
    # Check if already verified
    if current_user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email is already verified"
        )
    
    # Check rate limit
    if not check_rate_limit(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Please wait before requesting another verification email. You can request a new email in {RATE_LIMIT_SECONDS // 60} minutes."
        )
    
    try:
        # Create new verification token
        verification_token = create_verification_token(db, current_user.id)
        
        # Send email
        email_sent = send_verification_email(
            user_email=current_user.email,
            user_name=current_user.full_name,
            verification_token=verification_token
        )
        
        if not email_sent:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to send verification email. Please try again later."
            )
        
        return {"message": "Verification email sent successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error sending verification email: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send verification email. Please try again later."
        )


@router.get("/verify-email", status_code=status.HTTP_200_OK)
async def verify_email(
    token: str = Query(..., description="Verification token from email"),
    db: Session = Depends(get_db)
):
    """
    Verify user email using token from verification email.
    Public endpoint (no authentication required).
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Verification token is required"
        )
    
    try:
        # Find token in database
        verification_token = db.query(EmailVerificationToken).filter(
            EmailVerificationToken.token == token
        ).first()
        
        if not verification_token:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid verification token"
            )
        
        # Check if token has been used
        if verification_token.used_at is not None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="This verification token has already been used"
            )
        
        # Check if token has expired
        now = datetime.now(timezone.utc)
        if verification_token.expires_at.tzinfo is None:
            # Make timezone-aware if it's not
            expires_at = verification_token.expires_at.replace(tzinfo=timezone.utc)
        else:
            expires_at = verification_token.expires_at
        
        if now > expires_at:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Verification token has expired. Please request a new verification email."
            )
        
        # Get user
        user = db.query(User).filter(User.id == verification_token.user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Check if already verified
        if user.is_verified:
            # Mark token as used even if already verified (idempotent)
            verification_token.used_at = now
            db.commit()
            return {"message": "Email is already verified"}
        
        # Verify user and mark token as used
        user.is_verified = True
        verification_token.used_at = now
        db.commit()
        
        return {"message": "Email verified successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error verifying email: {str(e)}", exc_info=True)
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to verify email. Please try again later."
        )


@router.get("/verification-status", status_code=status.HTTP_200_OK)
async def get_verification_status(
    current_user: User = Depends(get_current_user)
):
    """
    Get the email verification status of the current user.
    """
    return {"is_verified": current_user.is_verified}


@router.get("/admin/users", response_model=List[UserResponse])
async def list_all_users(db: Session = Depends(get_db)):
    """Temporary admin endpoint to list all users. Remove in production."""
    users = db.query(User).all()
    return [
        UserResponse(
            id=user.id,
            email=user.email,
            full_name=user.full_name,
            username=user.username,
            is_verified=user.is_verified,
            is_pt=getattr(user, 'is_pt', False),
            technical_level=getattr(user, 'technical_level', None),
            favorite_exercise=getattr(user, 'favorite_exercise', None),
            community_preference=getattr(user, 'community_preference', None),
            created_at=user.created_at.isoformat() if user.created_at else None
        )
        for user in users
    ]


