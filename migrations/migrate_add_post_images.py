#!/usr/bin/env python3
"""
Migration script to add image_urls column to posts table.

This script:
1. Adds image_urls JSONB column to posts table (nullable, for photo posts)
"""

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

# Get database URL
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    print("ERROR: DATABASE_URL environment variable is required.")
    sys.exit(1)

# Fix Heroku postgres:// URL
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL)


def table_exists(connection, table_name):
    """Check if a table exists."""
    inspector = inspect(engine)
    return table_name in inspector.get_table_names()


def column_exists(connection, table_name, column_name):
    """Check if a column exists in a table."""
    inspector = inspect(engine)
    columns = [col['name'] for col in inspector.get_columns(table_name)]
    return column_name in columns


def run_migration():
    print("Running migration to add image_urls column to posts table...")
    
    with engine.connect() as connection:
        try:
            connection.begin()
            
            # Check if posts table exists
            if not table_exists(connection, 'posts'):
                print("   ⚠️  Table 'posts' does not exist. Please run migrate_add_social_feed.py first.")
                connection.rollback()
                return False
            
            # Check if image_urls column already exists
            if column_exists(connection, 'posts', 'image_urls'):
                print("   ✓ Column 'image_urls' already exists in 'posts' table.")
                connection.rollback()
                return True
            
            # Add image_urls column
            print("   Adding image_urls column to posts table...")
            connection.execute(text("""
                ALTER TABLE posts 
                ADD COLUMN image_urls JSONB
            """))
            connection.commit()
            print("   ✓ Successfully added image_urls column to posts table.")
            return True
            
        except ProgrammingError as e:
            connection.rollback()
            error_str = str(e)
            if "already exists" in error_str.lower() or "duplicate" in error_str.lower():
                print("   ✓ Column 'image_urls' already exists (safe to ignore).")
                return True
            else:
                print(f"   ✗ Error: {error_str}")
                return False
        except Exception as e:
            connection.rollback()
            print(f"   ✗ Unexpected error: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    success = run_migration()
    sys.exit(0 if success else 1)

