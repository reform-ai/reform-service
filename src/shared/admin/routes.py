"""Admin routes for managing users and system settings."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from src.shared.auth.database import get_db, User
from src.shared.admin.dependencies import verify_admin
from pydantic import BaseModel

router = APIRouter(prefix="/api/admin", tags=["admin"])


class VerifyPTResponse(BaseModel):
    """Response schema for PT verification."""
    username: str
    is_pt: bool
    message: str


@router.post("/users/{username}/verify-pt", response_model=VerifyPTResponse, status_code=status.HTTP_200_OK)
async def verify_pt_status(
    username: str,
    admin_user: User = Depends(verify_admin),  # Verify admin secret AND user is_admin=True
    db: Session = Depends(get_db)
):
    """
    Verify a user's Personal Trainer status.
    
    Requires:
    1. X-Admin-Secret header with valid admin secret
    2. Authenticated user with is_admin=True
    
    Sets the user's is_pt field to True.
    """
    # Find user by username
    user = db.query(User).filter(User.username == username).first()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with username '{username}' not found"
        )
    
    # Update is_pt status
    user.is_pt = True
    db.commit()
    db.refresh(user)
    
    return VerifyPTResponse(
        username=user.username,
        is_pt=user.is_pt,
        message=f"User '{username}' has been verified as a Personal Trainer"
    )

