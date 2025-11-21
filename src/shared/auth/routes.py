"""Authentication routes: signup and login endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import datetime
from typing import List
from src.shared.auth.database import get_db, User
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

router = APIRouter(prefix="/api/auth", tags=["auth"])


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
        db.flush()  # Flush to get user ID before adding tokens
        
        # Give 10 monthly tokens as signup bonus (expires in 30 days)
        from datetime import timedelta, timezone
        from src.shared.payment.token_utils import add_tokens
        
        expires_at = datetime.now(timezone.utc) + timedelta(days=30)
        add_tokens(
            db=db,
            user_id=user_id,
            amount=10,
            token_type='free',
            source='monthly_allotment',
            expires_at=expires_at,
            metadata={'signup_bonus': True, 'allotment_period': 'monthly'}  # This parameter name is fine, it gets mapped to meta_data in the model
        )
        
        db.commit()
        # No need to refresh - we have all the values we need
        
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
        
        # Update last login
        user.last_login = datetime.utcnow()
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


