#!/usr/bin/env python3
"""
Migration script to create email_verification_tokens table for email verification feature.

This script:
1. Creates email_verification_tokens table
2. Creates indexes for efficient lookups

Tokens expire after 1 hour.
When a new token is generated, old tokens for that user are invalidated.
"""

import os
import sys
from sqlalchemy import create_engine, text
from sqlalchemy.exc import ProgrammingError

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DATABASE_URL = os.environ.get("DATABASE_URL")

if not DATABASE_URL:
    print("ERROR: DATABASE_URL environment variable is required.")
    sys.exit(1)

# Heroku uses postgres:// but SQLAlchemy 2.0+ requires postgresql://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL)


def check_table_exists(connection, table_name):
    """Check if a table exists."""
    query = text("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' AND table_name = :table_name
    """)
    result = connection.execute(query, {"table_name": table_name})
    return result.fetchone() is not None


def create_email_verification_tokens_table():
    """Create email_verification_tokens table."""
    with engine.begin() as connection:
        try:
            if not check_table_exists(connection, "email_verification_tokens"):
                print("Creating 'email_verification_tokens' table...")
                connection.execute(text("""
                    CREATE TABLE email_verification_tokens (
                        id VARCHAR PRIMARY KEY,
                        user_id VARCHAR NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                        token VARCHAR NOT NULL UNIQUE,
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        expires_at TIMESTAMP NOT NULL,
                        used_at TIMESTAMP NULL
                    )
                """))
                print("✓ Successfully created 'email_verification_tokens' table.")
                
                # Create indexes
                print("Creating indexes for 'email_verification_tokens' table...")
                connection.execute(text("""
                    CREATE INDEX idx_email_verification_tokens_user_id 
                    ON email_verification_tokens(user_id)
                """))
                connection.execute(text("""
                    CREATE INDEX idx_email_verification_tokens_token 
                    ON email_verification_tokens(token)
                """))
                connection.execute(text("""
                    CREATE INDEX idx_email_verification_tokens_expires_at 
                    ON email_verification_tokens(expires_at)
                """))
                print("✓ Successfully created indexes.")
            else:
                print("✓ Table 'email_verification_tokens' already exists.")
                
        except ProgrammingError as e:
            print(f"ERROR: Database error: {str(e)}")
            sys.exit(1)
        except Exception as e:
            print(f"ERROR: {str(e)}")
            sys.exit(1)


if __name__ == "__main__":
    print("Running migration to create email_verification_tokens table...")
    print(f"Database: {DATABASE_URL.split('@')[-1] if '@' in DATABASE_URL else 'local'}")
    print()
    
    print("=" * 60)
    print("Creating email_verification_tokens table")
    print("=" * 60)
    create_email_verification_tokens_table()
    print()
    
    print("=" * 60)
    print("Migration completed successfully!")
    print("=" * 60)
    print()
    print("Note: Tokens expire after 1 hour.")
    print("When a new token is generated, old tokens for that user are invalidated.")

