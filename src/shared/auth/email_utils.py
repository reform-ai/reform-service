"""Email utilities for authentication features (verification, password reset, etc.)."""

import os
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional


def send_verification_email(user_email: str, user_name: str, verification_token: str) -> bool:
    """
    Send email verification email to user.
    
    Args:
        user_email: User's email address
        user_name: User's full name for personalization
        verification_token: Verification token to include in link
    
    Returns:
        True if email sent successfully, False otherwise
    """
    try:
        # Get email configuration from environment variables
        smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
        smtp_port = int(os.environ.get("SMTP_PORT", "587"))
        smtp_user = os.environ.get("SMTP_USER")
        smtp_password = os.environ.get("SMTP_PASSWORD")
        
        # Verification email settings
        verification_from = os.environ.get("VERIFICATION_EMAIL_FROM", "donotreply@reformgym.fit")
        base_url = os.environ.get("VERIFICATION_LINK_BASE_URL", "https://reformgym.fit")
        
        if not smtp_user or not smtp_password:
            logging.error("SMTP credentials not configured")
            return False
        
        # Build verification URL
        verification_url = f"{base_url}/verify-email?token={verification_token}"
        
        # Create message
        msg = MIMEMultipart('alternative')
        msg['From'] = verification_from
        msg['To'] = user_email
        msg['Subject'] = "Verify your Reform account"
        
        # Plain text version
        text_body = f"""
Hi {user_name},

Welcome to Reform! Please verify your email address to complete your account setup.

Click the link below to verify your email:
{verification_url}

This link will expire in 1 hour.

If you didn't create an account with Reform, you can safely ignore this email.

Best regards,
The Reform Team
"""
        
        # HTML version
        html_body = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px;">
    <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); padding: 30px; text-align: center; border-radius: 8px 8px 0 0;">
        <h1 style="color: white; margin: 0; font-size: 28px;">Welcome to Reform!</h1>
    </div>
    
    <div style="background: #ffffff; padding: 30px; border: 1px solid #e5e7eb; border-top: none; border-radius: 0 0 8px 8px;">
        <p style="font-size: 16px; margin-bottom: 20px;">Hi {user_name},</p>
        
        <p style="font-size: 16px; margin-bottom: 20px;">
            Please verify your email address to complete your account setup and start using Reform's features.
        </p>
        
        <div style="text-align: center; margin: 30px 0;">
            <a href="{verification_url}" 
               style="display: inline-block; background: #10b981; color: white; padding: 14px 28px; text-decoration: none; border-radius: 6px; font-weight: 600; font-size: 16px;">
                Verify Email Address
            </a>
        </div>
        
        <p style="font-size: 14px; color: #6b7280; margin-top: 30px; margin-bottom: 10px;">
            Or copy and paste this link into your browser:
        </p>
        <p style="font-size: 12px; color: #9ca3af; word-break: break-all; background: #f9fafb; padding: 10px; border-radius: 4px; margin: 0;">
            {verification_url}
        </p>
        
        <p style="font-size: 14px; color: #6b7280; margin-top: 30px; margin-bottom: 10px;">
            <strong>This link will expire in 1 hour.</strong>
        </p>
        
        <hr style="border: none; border-top: 1px solid #e5e7eb; margin: 30px 0;">
        
        <p style="font-size: 12px; color: #9ca3af; margin: 0;">
            If you didn't create an account with Reform, you can safely ignore this email.
        </p>
    </div>
    
    <div style="text-align: center; margin-top: 20px; padding: 20px;">
        <p style="font-size: 12px; color: #9ca3af; margin: 0;">
            © {os.environ.get('CURRENT_YEAR', '2025')} Reform. All rights reserved.
        </p>
    </div>
</body>
</html>
"""
        
        # Attach both versions
        part1 = MIMEText(text_body, 'plain')
        part2 = MIMEText(html_body, 'html')
        msg.attach(part1)
        msg.attach(part2)
        
        # Send email
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()  # Enable encryption
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
        
        logging.info(f"Verification email sent successfully to {user_email}")
        return True
        
    except Exception as e:
        logging.error(f"Failed to send verification email to {user_email}: {str(e)}", exc_info=True)
        return False

