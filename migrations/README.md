# Database Migrations

This directory contains one-time migration scripts for updating existing databases.

## Migration Order

Run migrations in this order if setting up a new database or updating an existing one:

1. **migrate_add_username.py** - Adds `username` column to `users` table
2. **migrate_add_social_feed.py** - Creates social feed tables (posts, likes, comments, follows) and adds `is_public` column
3. **migrate_add_comment_is_deleted.py** - Adds `is_deleted` column to `comments` table for soft delete
4. **migrate_add_is_pt.py** - Adds `is_pt` column to `users` table for Personal Trainer attribute

## Usage

All migration scripts are idempotent (safe to run multiple times). They check if changes already exist before applying them.

```bash
# Set DATABASE_URL environment variable
export DATABASE_URL="postgresql://user:password@localhost:5432/dbname"

# Run a migration
python migrations/migrate_add_username.py
```

## Note

For new databases, tables are automatically created via SQLAlchemy's `create_all()` on app startup. These migrations are only needed for:
- Updating existing production databases
- Adding columns/tables to databases that were created before the feature was added
- Documentation of schema changes

