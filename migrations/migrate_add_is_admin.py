#!/usr/bin/env python3
"""Migration script to add is_admin column to users table for admin role management."""

import os
import sys
from sqlalchemy import create_engine, text, inspect
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

def column_exists(connection, table_name, column_name):
    """Check if a column exists in a table."""
    inspector = inspect(connection)
    columns = inspector.get_columns(table_name)
    return any(c['name'] == column_name for c in columns)

def run_migration():
    print("Running migration to add is_admin column to users table...")
    print(f"Database: {engine.url.host}:{engine.url.port}/{engine.url.database}")

    with engine.connect() as connection:
        # Check if column exists before adding
        if not column_exists(connection, 'users', 'is_admin'):
            print("Adding is_admin column to users table...")
            connection.execute(text("ALTER TABLE users ADD COLUMN is_admin BOOLEAN DEFAULT FALSE NOT NULL"))
            connection.commit()
            print("✓ Successfully added is_admin column to users table.")
        else:
            print("✓ Column 'is_admin' already exists in 'users' table.")
    
    print("\n✓ Migration completed successfully!")

if __name__ == "__main__":
    run_migration()

