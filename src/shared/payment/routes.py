"""Payment and token system routes."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Dict
from src.shared.auth.database import get_db
from src.shared.auth.dependencies import get_current_user
from src.shared.auth.database import User
from src.shared.payment.token_utils import (
    calculate_token_balance,
    get_token_transactions,
    TokenBalance
)

router = APIRouter(prefix="/api/tokens", tags=["tokens"])


@router.get("/balance")
async def get_token_balance(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict:
    """
    Get detailed token balance breakdown for current user.
    Returns breakdown by source type.
    """
    from datetime import datetime, timezone, timedelta
    from sqlalchemy import func, and_, or_
    from src.shared.payment.database import TokenTransaction
    
    now = datetime.now(timezone.utc)
    
    # Check and grant monthly tokens if 30 days have passed since last grant
    from src.shared.payment.token_utils import grant_monthly_tokens_if_needed
    grant_monthly_tokens_if_needed(db, current_user.id)
    db.commit()  # Commit the monthly grant if it happened
    
    # Calculate overall balance
    balance = calculate_token_balance(db, current_user.id)
    
    # Get breakdown by source
    # Monthly allotment tokens - match debits to credits by grant period
    # Get all valid monthly credit transactions (not expired)
    valid_monthly_credits = db.query(TokenTransaction).filter(
        and_(
            TokenTransaction.user_id == current_user.id,
            TokenTransaction.token_type == 'free',
            TokenTransaction.source == 'monthly_allotment',
            TokenTransaction.amount > 0,
            or_(
                TokenTransaction.expires_at.is_(None),
                TokenTransaction.expires_at > now
            )
        )
    ).all()
    
    monthly_credits = sum(t.amount for t in valid_monthly_credits)
    
    # For each valid credit, find debits that were created during that credit's validity period
    monthly_debits = 0
    for credit in valid_monthly_credits:
        credit_created = credit.created_at
        credit_expires = credit.expires_at or (now + timedelta(days=365))  # If no expiration, use far future
        
        # Find debits created during this credit's validity period
        debits_for_this_credit = db.query(func.coalesce(func.sum(func.abs(TokenTransaction.amount)), 0)).filter(
            and_(
                TokenTransaction.user_id == current_user.id,
                TokenTransaction.token_type == 'free',
                TokenTransaction.amount < 0,
                TokenTransaction.created_at >= credit_created,
                TokenTransaction.created_at <= credit_expires,
                or_(
                    TokenTransaction.source == 'monthly_allotment',
                    TokenTransaction.meta_data['deducted_from'].astext == 'monthly_allotment'
                )
            )
        ).scalar() or 0
        
        monthly_debits += min(debits_for_this_credit, credit.amount)  # Can't deduct more than the credit amount
    
    monthly_tokens = max(0, monthly_credits - monthly_debits)
    
    # Get earliest expiration date for monthly allotment tokens
    monthly_expires_at = None
    if monthly_tokens > 0:
        earliest_expiry = db.query(func.min(TokenTransaction.expires_at)).filter(
            and_(
                TokenTransaction.user_id == current_user.id,
                TokenTransaction.token_type == 'free',
                TokenTransaction.source == 'monthly_allotment',
                TokenTransaction.amount > 0,
                TokenTransaction.expires_at.isnot(None),
                TokenTransaction.expires_at > now
            )
        ).scalar()
        monthly_expires_at = earliest_expiry.isoformat() if earliest_expiry else None
    
    # Other free tokens: promotional + signup_bonus + referral_bonus + any other sources (except monthly_allotment)
    # This combines promotional with other free token sources for display
    other_free_sources = ['promotional', 'signup_bonus', 'referral_bonus']
    other_free_credits = 0
    other_free_debits = 0
    
    for source in other_free_sources:
        credits = db.query(func.coalesce(func.sum(TokenTransaction.amount), 0)).filter(
            and_(
                TokenTransaction.user_id == current_user.id,
                TokenTransaction.token_type == 'free',
                TokenTransaction.source == source,
                TokenTransaction.amount > 0,
                or_(
                    TokenTransaction.expires_at.is_(None),
                    TokenTransaction.expires_at > now
                )
            )
        ).scalar() or 0
        
        debits = db.query(func.coalesce(func.sum(func.abs(TokenTransaction.amount)), 0)).filter(
            and_(
                TokenTransaction.user_id == current_user.id,
                TokenTransaction.token_type == 'free',
                TokenTransaction.source == source,
                TokenTransaction.amount < 0
            )
        ).scalar() or 0
        
        other_free_credits += credits
        other_free_debits += debits
    
    # Also get any free tokens with unknown/other sources (not monthly_allotment)
    known_sources = ['monthly_allotment'] + other_free_sources
    unknown_free_credits = db.query(func.coalesce(func.sum(TokenTransaction.amount), 0)).filter(
        and_(
            TokenTransaction.user_id == current_user.id,
            TokenTransaction.token_type == 'free',
            TokenTransaction.amount > 0,
            ~TokenTransaction.source.in_(known_sources),
            or_(
                TokenTransaction.expires_at.is_(None),
                TokenTransaction.expires_at > now
            )
        )
    ).scalar() or 0
    
    unknown_free_debits = db.query(func.coalesce(func.sum(func.abs(TokenTransaction.amount)), 0)).filter(
        and_(
            TokenTransaction.user_id == current_user.id,
            TokenTransaction.token_type == 'free',
            TokenTransaction.amount < 0,
            ~TokenTransaction.source.in_(known_sources)
        )
    ).scalar() or 0
    
    # Combine all other free tokens (promotional + signup_bonus + referral_bonus + unknown)
    other_free_tokens = max(0, other_free_credits + unknown_free_credits - other_free_debits - unknown_free_debits)
    
    # Get earliest expiration date for other free tokens
    other_free_expires_at = None
    if other_free_tokens > 0:
        # Check all other free sources for earliest expiration
        earliest_expiry = None
        for source in other_free_sources:
            expiry = db.query(func.min(TokenTransaction.expires_at)).filter(
                and_(
                    TokenTransaction.user_id == current_user.id,
                    TokenTransaction.token_type == 'free',
                    TokenTransaction.source == source,
                    TokenTransaction.amount > 0,
                    TokenTransaction.expires_at.isnot(None),
                    TokenTransaction.expires_at > now
                )
            ).scalar()
            if expiry and (earliest_expiry is None or expiry < earliest_expiry):
                earliest_expiry = expiry
        
        # Also check unknown sources (not in known_sources list)
        unknown_expiry = db.query(func.min(TokenTransaction.expires_at)).filter(
            and_(
                TokenTransaction.user_id == current_user.id,
                TokenTransaction.token_type == 'free',
                TokenTransaction.amount > 0,
                ~TokenTransaction.source.in_(known_sources),
                TokenTransaction.expires_at.isnot(None),
                TokenTransaction.expires_at > now
            )
        ).scalar()
        if unknown_expiry and (earliest_expiry is None or unknown_expiry < earliest_expiry):
            earliest_expiry = unknown_expiry
        
        other_free_expires_at = earliest_expiry.isoformat() if earliest_expiry else None
    
    # Purchased tokens
    purchased_tokens = balance.purchased_tokens
    
    return {
        "total": balance.total,
        "breakdown": {
            "monthly_allotment": monthly_tokens,
            "other_free": other_free_tokens,  # Includes promotional + signup_bonus + referral_bonus + others
            "purchased": purchased_tokens
        },
        "expiration_dates": {
            "monthly_allotment": monthly_expires_at,
            "other_free": other_free_expires_at,
            "purchased": None  # Purchased tokens never expire
        },
        "free_total": balance.free_tokens,
        "purchased_total": balance.purchased_tokens
    }


@router.get("/transactions")
async def get_token_transactions_history(
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict:
    """
    Get token transaction history for current user.
    """
    transactions = get_token_transactions(db, current_user.id, limit, offset)
    
    return {
        "transactions": [
            {
                "id": str(t.id),
                "token_type": t.token_type,
                "amount": t.amount,
                "source": t.source,
                "expires_at": t.expires_at.isoformat() if t.expires_at else None,
                "created_at": t.created_at.isoformat() if t.created_at else None,
                "metadata": t.meta_data
            }
            for t in transactions
        ],
        "limit": limit,
        "offset": offset
    }

