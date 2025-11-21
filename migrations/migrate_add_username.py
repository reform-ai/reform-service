#!/usr/bin/env python3
"""
Migration script to add username column to users table.
Run this script to add the username column to existing databases.
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

def check_column_exists(connection, table_name, column_name):
    """Check if a column exists in a table."""
    query = text("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = :table_name AND column_name = :column_name
    """)
    result = connection.execute(query, {"table_name": table_name, "column_name": column_name})
    return result.fetchone() is not None

def add_username_column():
    """Add username column to users table if it doesn't exist."""
    with engine.begin() as connection:
        try:
            # Check if column already exists
            if check_column_exists(connection, "users", "username"):
                print("✓ Column 'username' already exists in 'users' table.")
                return
            
            # Add username column
            print("Adding 'username' column to 'users' table...")
            connection.execute(text("""
                ALTER TABLE users 
                ADD COLUMN username VARCHAR UNIQUE
            """))
            
            # Create index on username for faster lookups
            print("Creating index on 'username' column...")
            connection.execute(text("""
                CREATE INDEX IF NOT EXISTS ix_users_username ON users(username)
            """))
            
            print("✓ Successfully added 'username' column to 'users' table.")
            print("✓ Index created on 'username' column.")
                
        except ProgrammingError as e:
            print(f"ERROR: Database error: {str(e)}")
            sys.exit(1)
        except Exception as e:
            print(f"ERROR: {str(e)}")
            sys.exit(1)

if __name__ == "__main__":
    print("Running migration to add username column...")
    print(f"Database: {DATABASE_URL.split('@')[-1] if '@' in DATABASE_URL else 'local'}")
    print()
    add_username_column()
    print()
    print("Migration completed successfully!")

