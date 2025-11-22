"""Authentication routes: signup, login, and email verification endpoints."""

import os
import secrets
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import List, Dict

from fastapi import APIRouter, Depends, HTTPException, status, Query, Response, Request
from sqlalchemy.orm import Session
from src.shared.auth.rate_limit_utils import check_rate_limit as check_general_rate_limit, get_client_ip

from src.shared.auth.database import get_db, User, EmailVerificationToken
from src.shared.auth.auth import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
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
from src.shared.auth.input_validation import (
    validate_name,
    validate_email,
    validate_username,
    validate_password,
    validate_full_name,
    validate_notes,
    sanitize_text
)

router = APIRouter(prefix="/api/auth", tags=["auth"])

# Rate limiting will be applied via decorators that access app.state.limiter
# The limiter is initialized in app.py

# Email verification rate limiting (in-memory, additional to slowapi)
# Format: {user_id: timestamp_of_last_email}
_verification_email_rate_limit: Dict[str, float] = {}
RATE_LIMIT_SECONDS = 300  # 5 minutes between verification email requests


def generate_verification_token() -> str:
    """Generate a secure random token for email verification."""
    return secrets.token_urlsafe(32)


def create_verification_token(db: Session, user_id: str) -> str:
    """
    Create a new verification token for a user, invalidating old unused tokens.
    
    Args:
        db: Database session
        user_id: User ID to create token for
    
    Returns:
        The verification token string
    """
    # Invalidate all existing unused tokens for this user
    db.query(EmailVerificationToken).filter(
        EmailVerificationToken.user_id == user_id,
        EmailVerificationToken.used_at.is_(None)
    ).delete()
    
    # Generate new token
    token = generate_verification_token()
    
    # Calculate expiration (default 1 hour, configurable)
    # Use timezone-naive UTC datetime (datetime.utcnow()) to match database storage
    # This ensures consistent timezone handling regardless of database timezone setting
    expiry_hours = int(os.environ.get("VERIFICATION_TOKEN_EXPIRY_HOURS", "1"))
    expires_at = datetime.utcnow() + timedelta(hours=expiry_hours)
    
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
async def signup(
    request: Request,
    signup_data: SignupRequest,
    db: Session = Depends(get_db)
):
    """Create a new user account."""
    import logging
    
    # Apply rate limiting (3 signups per hour per IP)
    client_ip = get_client_ip(request)
    check_general_rate_limit(client_ip, "signup", max_requests=3, window_seconds=3600)
    
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
        # Validate and sanitize all inputs
        email = validate_email(signup_data.email)
        password = validate_password(signup_data.password)
        first_name = validate_name(signup_data.first_name, "First name")
        last_name = validate_name(signup_data.last_name, "Last name")
        full_name = validate_full_name(first_name, last_name)
        
        # Check if user already exists
        existing_user = db.query(User).filter(User.email == email).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Create new user
        user_id = generate_user_id()
        password_hash = hash_password(password)

        new_user = User(
            id=user_id,
            email=email,
            password_hash=password_hash,
            full_name=full_name,
            is_verified=False,  # User must verify email to use social features
            created_at=datetime.utcnow(),
            tokens_remaining=10,  # Keep for backward compatibility, but use transaction system
            last_token_reset=datetime.utcnow()
        )
        
        db.add(new_user)
        db.commit()
        # Note: Tokens are NOT granted at signup. User must click "Get 10 Free Tokens" button on profile page.
        
        # Send verification email automatically after signup
        # Don't fail signup if email sending fails - user can request email later
        try:
            verification_token = create_verification_token(db, user_id)
            email_sent = send_verification_email(
                user_email=email,
                user_name=full_name,
                verification_token=verification_token
            )
            if not email_sent:
                logging.warning(f"Failed to send verification email to {email} during signup")
        except Exception as e:
            # Don't fail signup if email sending fails - user can request email later
            logging.error(f"Error sending verification email during signup: {str(e)}", exc_info=True)
        
        # Create access and refresh tokens
        token_data = {"sub": user_id, "email": email}
        access_token = create_access_token(data=token_data)
        refresh_token = create_refresh_token(data=token_data)
        
        # Create response with user data (tokens sent via httpOnly cookies)
        response = TokenResponse(
            access_token=None,  # Not in body, sent via cookie
            token_type="bearer",
            user_id=user_id,
            email=email,
            full_name=full_name
        )
        
        # Set httpOnly cookies with tokens
        from src.shared.auth.auth import ACCESS_TOKEN_EXPIRE_MINUTES, REFRESH_TOKEN_EXPIRE_DAYS
        from datetime import timedelta
        access_max_age = int(timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES).total_seconds())
        refresh_max_age = int(timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS).total_seconds())
        
        # Determine if we're in production (HTTPS) or development (HTTP)
        is_production = os.environ.get("DYNO") or os.environ.get("ENVIRONMENT") == "production"
        
        # Create response object
        from fastapi.responses import JSONResponse
        json_response = JSONResponse(content=response.dict(), status_code=status.HTTP_201_CREATED)
        
        # Set access token cookie (short-lived)
        json_response.set_cookie(
            key="access_token",
            value=access_token,
            max_age=access_max_age,
            httponly=True,
            secure=is_production,
            samesite="lax",
            path="/"
        )
        
        # Set refresh token cookie (long-lived)
        json_response.set_cookie(
            key="refresh_token",
            value=refresh_token,
            max_age=refresh_max_age,
            httponly=True,
            secure=is_production,
            samesite="lax",
            path="/"
        )
        
        return json_response
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
async def login(
    request: Request,
    login_data: LoginRequest,
    db: Session = Depends(get_db)
):
    """Authenticate user and return access token."""
    import logging
    
    # Apply rate limiting (5 login attempts per minute per IP)
    client_ip = get_client_ip(request)
    check_general_rate_limit(client_ip, "login", max_requests=5, window_seconds=60)
    
    try:
        # Find user by email
        user = db.query(User).filter(User.email == login_data.email).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        # Verify password
        if not verify_password(login_data.password, user.password_hash):
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
        
        # Create access and refresh tokens
        try:
            token_data = {"sub": user.id, "email": user.email}
            access_token = create_access_token(data=token_data)
            refresh_token = create_refresh_token(data=token_data)
        except ValueError as e:
            # SECRET_KEY is missing or invalid
            logging.error(f"Failed to create tokens: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Server configuration error. Please contact support."
            )
        
        # Create response with user data (tokens sent via httpOnly cookies)
        response = TokenResponse(
            access_token=None,  # Not in body, sent via cookie
            token_type="bearer",
            user_id=user.id,
            email=user.email,
            full_name=user.full_name
        )
        
        # Set httpOnly cookies with tokens
        from src.shared.auth.auth import ACCESS_TOKEN_EXPIRE_MINUTES, REFRESH_TOKEN_EXPIRE_DAYS
        from datetime import timedelta
        access_max_age = int(timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES).total_seconds())
        refresh_max_age = int(timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS).total_seconds())
        
        # Determine if we're in production (HTTPS) or development (HTTP)
        is_production = os.environ.get("DYNO") or os.environ.get("ENVIRONMENT") == "production"
        
        # Create response object
        from fastapi.responses import JSONResponse
        json_response = JSONResponse(content=response.dict())
        
        # Set access token cookie (short-lived)
        json_response.set_cookie(
            key="access_token",
            value=access_token,
            max_age=access_max_age,
            httponly=True,
            secure=is_production,
            samesite="lax",
            path="/"
        )
        
        # Set refresh token cookie (long-lived)
        json_response.set_cookie(
            key="refresh_token",
            value=refresh_token,
            max_age=refresh_max_age,
            httponly=True,
            secure=is_production,
            samesite="lax",
            path="/"
        )
        
        return json_response
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


@router.post("/refresh", status_code=status.HTTP_200_OK)
async def refresh_token_endpoint(request: Request, db: Session = Depends(get_db)):
    """
    Refresh access token using refresh token.
    Returns new access token in httpOnly cookie.
    """
    # Apply rate limiting (10 refresh requests per minute per IP)
    client_ip = get_client_ip(request)
    check_general_rate_limit(client_ip, "refresh", max_requests=10, window_seconds=60)
    
    # Get refresh token from cookie
    refresh_token = request.cookies.get("refresh_token")
    
    if not refresh_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token not found"
        )
    
    # Verify refresh token
    from src.shared.auth.auth import verify_token, create_access_token
    payload = verify_token(refresh_token, token_type="refresh")
    
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token"
        )
    
    # Get user ID from token
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload"
        )
    
    # Verify user exists
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    # Create new access token
    try:
        access_token = create_access_token(data={"sub": user.id, "email": user.email})
    except ValueError as e:
        logging.error(f"Failed to create access token: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server configuration error. Please contact support."
        )
    
    # Set new access token cookie
    from src.shared.auth.auth import ACCESS_TOKEN_EXPIRE_MINUTES
    from datetime import timedelta
    access_max_age = int(timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES).total_seconds())
    is_production = os.environ.get("DYNO") or os.environ.get("ENVIRONMENT") == "production"
    
    from fastapi.responses import JSONResponse
    response = JSONResponse(content={"message": "Token refreshed successfully"})
    response.set_cookie(
        key="access_token",
        value=access_token,
        max_age=access_max_age,
        httponly=True,
        secure=is_production,
        samesite="lax",
        path="/"
    )
    
    return response


@router.post("/logout", status_code=status.HTTP_200_OK)
async def logout():
    """
    Logout endpoint that clears the httpOnly cookies.
    No authentication required - just clears the cookies.
    """
    from fastapi.responses import JSONResponse
    response = JSONResponse(content={"message": "Logged out successfully"})
    
    # Determine if we're in production (HTTPS) or development (HTTP)
    is_production = os.environ.get("DYNO") or os.environ.get("ENVIRONMENT") == "production"
    
    # Clear both cookies by setting them with max_age=0
    response.set_cookie(
        key="access_token",
        value="",
        max_age=0,
        httponly=True,
        secure=is_production,
        samesite="lax",
        path="/"
    )
    response.set_cookie(
        key="refresh_token",
        value="",
        max_age=0,
        httponly=True,
        secure=is_production,
        samesite="lax",
        path="/"
    )
    return response


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
    request: Request,
    password_data: ChangePasswordRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Change user password."""
    # Apply rate limiting (5 password changes per hour per user)
    check_general_rate_limit(current_user.id, "change_password", max_requests=5, window_seconds=3600)
    
    # Verify current password
    if not verify_password(password_data.current_password, current_user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Current password is incorrect"
        )
    
    # Validate new password
    new_password = validate_password(password_data.new_password)
    
    # Check new password is different
    if verify_password(new_password, current_user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="New password must be different from current password"
        )
    
    # Hash and update password
    new_password_hash = hash_password(new_password)
    current_user.password_hash = new_password_hash
    db.commit()
    
    return {"message": "Password changed successfully"}


@router.post("/update-username", status_code=status.HTTP_200_OK)
async def update_username(
    username_data: UpdateUsernameRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update user username."""
    # Validate and sanitize username
    username = validate_username(username_data.username)
    
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
    
    # Update favorite_exercise if provided - validate and sanitize
    if request.favorite_exercise is not None:
        if request.favorite_exercise.strip():
            # Sanitize to prevent XSS
            favorite_exercise = sanitize_text(request.favorite_exercise.strip(), max_length=100)
            current_user.favorite_exercise = favorite_exercise
        else:
            current_user.favorite_exercise = None
    
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
        # Both expires_at and now are timezone-naive UTC times
        now = datetime.utcnow()
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
            verification_token.used_at = datetime.utcnow()
            db.commit()
            return {"message": "Email is already verified"}
        
        # Verify user and mark token as used
        user.is_verified = True
        verification_token.used_at = datetime.utcnow()
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


