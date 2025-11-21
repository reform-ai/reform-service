#!/usr/bin/env python3
"""
Migration script to add social feed tables and update users table.

This script:
1. Adds is_public column to users table
2. Creates posts table
3. Creates likes table
4. Creates comments table
5. Creates follows table
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
    print("Running migration to add social feed tables...")
    print(f"Database: {engine.url.host}:{engine.url.port}/{engine.url.database}")
    print()

    with engine.connect() as connection:
        # 1. Add is_public column to users table
        print("1. Checking users table...")
        if not table_exists(connection, 'users'):
            print("   ERROR: users table does not exist. Please run initial migration first.")
            sys.exit(1)
        
        if not column_exists(connection, 'users', 'is_public'):
            print("   Adding is_public column to users table...")
            connection.execute(text("""
                ALTER TABLE users 
                ADD COLUMN is_public BOOLEAN DEFAULT TRUE NOT NULL
            """))
            print("   ✓ Added is_public column to users table.")
        else:
            print("   ✓ Column 'is_public' already exists in 'users' table.")

        # 2. Create posts table
        print("\n2. Checking posts table...")
        if not table_exists(connection, 'posts'):
            print("   Creating posts table...")
            connection.execute(text("""
                CREATE TABLE posts (
                    id UUID PRIMARY KEY,
                    user_id VARCHAR NOT NULL,
                    post_type VARCHAR NOT NULL,
                    content TEXT,
                    analysis_id VARCHAR,
                    score_data JSONB,
                    plot_config JSONB,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT fk_posts_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
            """))
            print("   Creating indexes on posts table...")
            connection.execute(text("CREATE INDEX idx_posts_user_id ON posts(user_id)"))
            connection.execute(text("CREATE INDEX idx_posts_created_at ON posts(created_at)"))
            connection.execute(text("CREATE INDEX idx_posts_user_created ON posts(user_id, created_at)"))
            print("   ✓ Created posts table with indexes.")
        else:
            print("   ✓ Table 'posts' already exists.")

        # 3. Create likes table
        print("\n3. Checking likes table...")
        if not table_exists(connection, 'likes'):
            print("   Creating likes table...")
            connection.execute(text("""
                CREATE TABLE likes (
                    id UUID PRIMARY KEY,
                    post_id UUID NOT NULL,
                    user_id VARCHAR NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT fk_likes_post FOREIGN KEY (post_id) REFERENCES posts(id) ON DELETE CASCADE,
                    CONSTRAINT fk_likes_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    CONSTRAINT uq_like_post_user UNIQUE (post_id, user_id)
                )
            """))
            print("   Creating indexes on likes table...")
            connection.execute(text("CREATE INDEX idx_likes_post_id ON likes(post_id)"))
            connection.execute(text("CREATE INDEX idx_likes_user_id ON likes(user_id)"))
            print("   ✓ Created likes table with indexes.")
        else:
            print("   ✓ Table 'likes' already exists.")

        # 4. Create comments table
        print("\n4. Checking comments table...")
        if not table_exists(connection, 'comments'):
            print("   Creating comments table...")
            connection.execute(text("""
                CREATE TABLE comments (
                    id UUID PRIMARY KEY,
                    post_id UUID NOT NULL,
                    user_id VARCHAR NOT NULL,
                    content TEXT NOT NULL,
                    parent_comment_id UUID,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT fk_comments_post FOREIGN KEY (post_id) REFERENCES posts(id) ON DELETE CASCADE,
                    CONSTRAINT fk_comments_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    CONSTRAINT fk_comments_parent FOREIGN KEY (parent_comment_id) REFERENCES comments(id) ON DELETE CASCADE
                )
            """))
            print("   Creating indexes on comments table...")
            connection.execute(text("CREATE INDEX idx_comments_post_id ON comments(post_id)"))
            connection.execute(text("CREATE INDEX idx_comments_user_id ON comments(user_id)"))
            connection.execute(text("CREATE INDEX idx_comments_created_at ON comments(created_at)"))
            connection.execute(text("CREATE INDEX idx_comments_post_created ON comments(post_id, created_at)"))
            print("   ✓ Created comments table with indexes.")
        else:
            print("   ✓ Table 'comments' already exists.")

        # 5. Create follows table
        print("\n5. Checking follows table...")
        if not table_exists(connection, 'follows'):
            print("   Creating follows table...")
            connection.execute(text("""
                CREATE TABLE follows (
                    id UUID PRIMARY KEY,
                    follower_id VARCHAR NOT NULL,
                    following_id VARCHAR NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT fk_follows_follower FOREIGN KEY (follower_id) REFERENCES users(id) ON DELETE CASCADE,
                    CONSTRAINT fk_follows_following FOREIGN KEY (following_id) REFERENCES users(id) ON DELETE CASCADE,
                    CONSTRAINT uq_follow_follower_following UNIQUE (follower_id, following_id)
                )
            """))
            print("   Creating indexes on follows table...")
            connection.execute(text("CREATE INDEX idx_follows_follower_id ON follows(follower_id)"))
            connection.execute(text("CREATE INDEX idx_follows_following_id ON follows(following_id)"))
            print("   ✓ Created follows table with indexes.")
        else:
            print("   ✓ Table 'follows' already exists.")

        # Commit all changes
        connection.commit()
        print("\n✓ Migration completed successfully!")
        print("\nSummary:")
        print("  - Added is_public column to users table")
        print("  - Created posts table")
        print("  - Created likes table")
        print("  - Created comments table")
        print("  - Created follows table")


if __name__ == "__main__":
    try:
        run_migration()
    except Exception as e:
        print(f"\n✗ Migration failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

