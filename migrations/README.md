# Database Migrations

This directory contains one-time migration scripts for updating existing databases.

## Migration Order

Run migrations in this order if setting up a new database or updating an existing one:

### Core User & Auth
1. **migrate_add_username.py** - Adds `username` column to `users` table
2. **migrate_add_email_verification.py** - Creates `email_verification_tokens` table for email verification

### Social Features
3. **migrate_add_social_feed.py** - Creates social feed tables (posts, likes, comments, follows) and adds `is_public` column to `users`
4. **migrate_add_comment_is_deleted.py** - Adds `is_deleted` column to `comments` table for soft delete

### User Profile & Roles
5. **migrate_add_is_pt.py** - Adds `is_pt` column to `users` table for Personal Trainer attribute
6. **migrate_add_is_admin.py** - Adds `is_admin` column to `users` table for admin role
7. **migrate_add_user_profile_attributes.py** - Adds `technical_level`, `favorite_exercise`, `community_preference` to `users` table

### Payment & Tokens
8. **migrate_add_payment_tables.py** - Creates `token_transactions`, `subscriptions`, `payments` tables and adds payment-related columns to `users` table
9. **migrate_add_token_activation_date.py** - Adds `token_activation_date` column to `users` table

### Analysis History
10. **migrate_add_analyses_table.py** - Creates `analyses` table for analysis history (creates `fps` as INTEGER)
11. **migrate_fps_to_float.py** - Changes `fps` column from INTEGER to FLOAT (must run after analyses table is created)

## Usage

All migration scripts are idempotent (safe to run multiple times). They check if changes already exist before applying them.

```bash
# Set DATABASE_URL environment variable
export DATABASE_URL="postgresql://user:password@localhost:5432/dbname"

# Run a migration
python migrations/migrate_add_username.py

# Or run all migrations in order (example script)
python migrations/run_all_migrations.py  # (if created)
```

## Migration Utilities

The `migration_utils.py` module provides common helper functions:
- `get_database_url()` - Get and validate DATABASE_URL
- `get_engine()` - Create SQLAlchemy engine
- `table_exists()` - Check if table exists
- `column_exists()` - Check if column exists
- `index_exists()` - Check if index exists
- `run_migration()` - Run migration with error handling

## Important Notes

1. **For new databases**: Tables are automatically created via SQLAlchemy's `create_all()` on app startup. These migrations are only needed for:
   - Updating existing production databases
   - Adding columns/tables to databases that were created before the feature was added
   - Documentation of schema changes

2. **Migration dependencies**: Some migrations depend on others:
   - `migrate_fps_to_float.py` must run after `migrate_add_analyses_table.py`
   - `migrate_add_social_feed.py` must run after `migrate_add_username.py` (if using username in social features)

3. **Idempotency**: All migrations check for existing changes before applying them, so they can be safely re-run.

4. **Production**: Always test migrations on a staging database before running on production.
