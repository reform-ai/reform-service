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
    ChangePasswordRequest
)
from src.shared.auth.dependencies import get_current_user

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/signup", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def signup(request: SignupRequest, db: Session = Depends(get_db)):
    """Create a new user account."""
    # Ensure database tables exist (fallback if startup init failed)
    try:
        from src.shared.auth.database import Base, engine
        Base.metadata.create_all(bind=engine, checkfirst=True)
    except Exception:
        pass  # Tables might already exist, continue anyway
    
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

    new_user = User(
        id=user_id,
        email=request.email,
        password_hash=password_hash,
        full_name=request.full_name,
        is_verified=False,  # Email verification to be implemented later
        created_at=datetime.utcnow(),
        tokens_remaining=10,  # New users start with 10 tokens
        last_token_reset=datetime.utcnow()
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    # Create access token
    access_token = create_access_token(data={"sub": user_id, "email": request.email})
    
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        user_id=user_id,
        email=request.email,
        full_name=request.full_name
    )


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    """Authenticate user and return access token."""
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
    access_token = create_access_token(data={"sub": user.id, "email": user.email})
    
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        user_id=user.id,
        email=user.email,
        full_name=user.full_name
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get current user information."""
    # Reset tokens if it's a new day
    from src.shared.auth.database import reset_daily_tokens_if_needed
    reset_daily_tokens_if_needed(current_user, db)
    db.refresh(current_user)
    
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        full_name=current_user.full_name,
        is_verified=current_user.is_verified,
        created_at=current_user.created_at.isoformat() if current_user.created_at else None,
        tokens_remaining=current_user.tokens_remaining
    )


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


@router.get("/admin/users", response_model=List[UserResponse])
async def list_all_users(db: Session = Depends(get_db)):
    """Temporary admin endpoint to list all users. Remove in production."""
    users = db.query(User).all()
    return [
        UserResponse(
            id=user.id,
            email=user.email,
            full_name=user.full_name,
            is_verified=user.is_verified,
            created_at=user.created_at.isoformat() if user.created_at else None
        )
        for user in users
    ]


