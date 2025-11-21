"""Contact routes for sending messages to support."""

import os
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, HTTPException, status, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from sqlalchemy import and_, func

from src.shared.contact.schemas import ContactRequest, ContactResponse
from src.shared.contact.database import ContactRateLimit
from src.shared.auth.database import get_db
from src.shared.auth.dependencies import get_current_user

router = APIRouter(prefix="/api/contact", tags=["contact"])
security = HTTPBearer(auto_error=False)

# Rate limiting configuration
RATE_LIMIT_MAX_REQUESTS = 3  # Max 3 messages per hour
RATE_LIMIT_WINDOW = timedelta(hours=1)
RATE_LIMIT_CLEANUP_AGE = timedelta(days=1)  # Clean up entries older than 1 day


def get_client_ip(request: Request) -> str:
    """Get client IP address for rate limiting."""
    # Check for forwarded IP (from proxy/load balancer)
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        # Take the first IP in the chain
        return forwarded.split(",")[0].strip()
    # Fallback to direct connection
    return request.client.host if request.client else "unknown"


def check_rate_limit(db: Session, ip_address: str, email: Optional[str] = None) -> bool:
    """
    Check if IP address or email has exceeded rate limit.
    Uses database-backed rate limiting that works across multiple dynos.
    
    Args:
        db: Database session
        ip_address: Client IP address
        email: Optional email address (for additional rate limiting by email)
    
    Returns:
        True if within rate limit, False if exceeded
    """
    now = datetime.utcnow()
    window_start = now - RATE_LIMIT_WINDOW
    
    # Clean up old entries (older than RATE_LIMIT_CLEANUP_AGE)
    cleanup_threshold = now - RATE_LIMIT_CLEANUP_AGE
    try:
        db.query(ContactRateLimit).filter(
            ContactRateLimit.created_at < cleanup_threshold
        ).delete()
        db.commit()
    except Exception as e:
        logging.warning(f"Failed to cleanup old rate limit entries: {str(e)}")
        db.rollback()
    
    # Check rate limit by IP address
    ip_count = db.query(func.count(ContactRateLimit.id)).filter(
        and_(
            ContactRateLimit.id == ip_address,
            ContactRateLimit.identifier_type == 'ip',
            ContactRateLimit.created_at >= window_start
        )
    ).scalar() or 0
    
    if ip_count >= RATE_LIMIT_MAX_REQUESTS:
        return False
    
    # Also check rate limit by email if provided (stricter limit)
    if email:
        email_count = db.query(func.count(ContactRateLimit.id)).filter(
            and_(
                ContactRateLimit.id == email.lower(),
                ContactRateLimit.identifier_type == 'email',
                ContactRateLimit.created_at >= window_start
            )
        ).scalar() or 0
        
        if email_count >= RATE_LIMIT_MAX_REQUESTS:
            return False
    
    # Record this request
    try:
        # Record IP address
        ip_record = ContactRateLimit(
            id=ip_address,
            identifier_type='ip',
            created_at=now
        )
        db.add(ip_record)
        
        # Also record email if provided
        if email:
            email_record = ContactRateLimit(
                id=email.lower(),
                identifier_type='email',
                created_at=now
            )
            db.add(email_record)
        
        db.commit()
    except Exception as e:
        logging.error(f"Failed to record rate limit: {str(e)}")
        db.rollback()
        # Don't fail the request if rate limit recording fails
        # But log it for monitoring
    
    return True


def send_email(name: str, email: str, subject: str, message: str) -> bool:
    """Send email to support@reformgym.fit using SMTP."""
    try:
        # Get email configuration from environment variables
        smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
        smtp_port = int(os.environ.get("SMTP_PORT", "587"))
        smtp_user = os.environ.get("SMTP_USER")
        smtp_password = os.environ.get("SMTP_PASSWORD")
        support_email = os.environ.get("SUPPORT_EMAIL", "support@reformgym.fit")
        
        if not smtp_user or not smtp_password:
            logging.error("SMTP credentials not configured")
            return False
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = smtp_user
        msg['To'] = support_email
        msg['Reply-To'] = email  # Allow support to reply directly to user
        msg['Subject'] = f"Contact Form: {subject}"
        
        # Create email body
        body = f"""
New contact form submission from ReformGym website:

Name: {name}
Email: {email}
Subject: {subject}

Message:
{message}

---
This message was sent from the ReformGym contact form.
Reply directly to this email to respond to {name} ({email}).
"""
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()  # Enable encryption
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
        
        logging.info(f"Contact form email sent successfully from {email}")
        return True
        
    except Exception as e:
        logging.error(f"Failed to send contact form email: {str(e)}", exc_info=True)
        return False


@router.post("/submit", response_model=ContactResponse, status_code=status.HTTP_200_OK)
async def submit_contact_form(
    contact_data: ContactRequest,
    request: Request,
    db: Session = Depends(get_db),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Submit contact form message to support@reformgym.fit.
    
    Features:
    - Database-backed rate limiting: Max 3 messages per hour per IP address and email
    - Works across multiple server instances/dynos
    - Input validation and sanitization
    - Secure email sending via SMTP
    - Optional: If user is logged in, their account info is included
    """
    # Ensure contact rate limit table exists (fallback if startup init failed)
    try:
        from src.shared.contact.database import Base as ContactBase
        from src.shared.auth.database import engine
        ContactBase.metadata.create_all(bind=engine, checkfirst=True)
    except Exception as e:
        logging.warning(f"Contact table creation check failed: {str(e)}")
        # Continue anyway - rate limiting will fail gracefully
    
    # Rate limiting check (by IP and email)
    client_ip = get_client_ip(request)
    if not check_rate_limit(db, client_ip, contact_data.email):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Please wait before sending another message. (Max {RATE_LIMIT_MAX_REQUESTS} messages per hour per IP address or email)"
        )
    
    # Optional: Get user info if logged in
    user_info = None
    if credentials:
        try:
            # Try to get current user (optional - don't fail if not authenticated)
            user = get_current_user(credentials, db)
            if user:
                user_info = f"User ID: {user.id}, Username: {user.username or 'N/A'}"
        except:
            # User not authenticated or invalid token - that's fine, continue as anonymous
            pass
    
    # Send email
    email_sent = send_email(
        name=contact_data.name,
        email=contact_data.email,
        subject=contact_data.subject,
        message=contact_data.message + (f"\n\n---\n{user_info}" if user_info else "")
    )
    
    if not email_sent:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send message. Please try again later or contact support directly."
        )
    
    return ContactResponse(
        success=True,
        message="Your message has been sent successfully! We'll get back to you soon."
    )

