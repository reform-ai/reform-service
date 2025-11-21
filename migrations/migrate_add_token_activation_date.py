import os
import sys
from sqlalchemy import create_engine, text
from sqlalchemy.exc import ProgrammingError

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DATABASE_URL = os.environ.get("DATABASE_URL")

if not DATABASE_URL:
    print("ERROR: DATABASE_URL environment variable is required.")
    sys.exit(1)

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL)

def check_column_exists(connection, table_name, column_name):
    query = text("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = :table_name AND column_name = :column_name
    """)
    result = connection.execute(query, {"table_name": table_name, "column_name": column_name})
    return result.fetchone() is not None

def add_token_activation_date():
    with engine.begin() as connection:
        print("Adding 'token_activation_date' column to 'users' table...")
        if not check_column_exists(connection, "users", "token_activation_date"):
            connection.execute(text("ALTER TABLE users ADD COLUMN token_activation_date TIMESTAMP"))
            print("✓ Successfully added 'token_activation_date' column.")
        else:
            print("✓ Column 'token_activation_date' already exists.")

if __name__ == "__main__":
    print("Running migration to add token_activation_date column...")
    print(f"Database: {DATABASE_URL.split('@')[-1] if '@' in DATABASE_URL else 'local'}")
    print()
    add_token_activation_date()
    print()
    print("Migration completed successfully!")

