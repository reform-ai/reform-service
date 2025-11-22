"""Admin authentication dependencies."""

import os
from fastapi import Depends, HTTPException, status, Header
from typing import Optional
from src.shared.auth.database import User
from src.shared.auth.dependencies import get_current_user


def verify_admin(
    x_admin_secret: Optional[str] = Header(None, alias="X-Admin-Secret"),
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Verify admin access: requires both admin secret AND user must have is_admin=True.
    
    This ensures only authenticated admin users with the correct secret can access admin endpoints.
    
    Returns the current user if both checks pass.
    Raises HTTPException if either check fails.
    """
    # Check 1: Verify admin secret
    admin_secret = os.environ.get("ADMIN_SECRET")
    
    if not admin_secret:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Admin authentication not configured"
        )
    
    if not x_admin_secret:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin secret required. Provide X-Admin-Secret header."
        )
    
    if x_admin_secret != admin_secret:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid admin secret"
        )
    
    # Check 2: Verify user has admin role
    if not getattr(current_user, 'is_admin', False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required. Your account does not have admin privileges."
        )
    
    return current_user

