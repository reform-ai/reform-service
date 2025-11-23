"""Database models for social feed feature."""

from sqlalchemy import Column, String, Boolean, DateTime, Integer, Text, ForeignKey, UniqueConstraint, Index
from sqlalchemy.dialects.postgresql import UUID, JSON, JSONB
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

# Import Base from auth database to use the same declarative base
from src.shared.auth.database import Base


class Post(Base):
    """Post model for social feed."""
    __tablename__ = "posts"

    id = Column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    post_type = Column(String, nullable=False)  # 'score', 'text', 'plot'
    content = Column(Text, nullable=True)  # Caption or text post content
    analysis_id = Column(String, nullable=True)  # Link to analysis if sharing from analysis
    score_data = Column(JSON, nullable=True)  # Snapshot of score details
    plot_config = Column(JSON, nullable=True)  # Config to recreate plots
    image_urls = Column(JSONB, nullable=True)  # List of image URLs (for photo posts, future: analysis-to-photo conversion)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    likes = relationship("Like", back_populates="post", cascade="all, delete-orphan")
    comments = relationship("Comment", back_populates="post", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_posts_user_created', 'user_id', 'created_at'),
    )


class Like(Base):
    """Like model for posts."""
    __tablename__ = "likes"

    id = Column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4()))
    post_id = Column(UUID(as_uuid=False), ForeignKey("posts.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    post = relationship("Post", back_populates="likes")

    __table_args__ = (
        UniqueConstraint('post_id', 'user_id', name='uq_like_post_user'),
    )


class Comment(Base):
    """Comment model for posts."""
    __tablename__ = "comments"

    id = Column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4()))
    post_id = Column(UUID(as_uuid=False), ForeignKey("posts.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    content = Column(Text, nullable=False)
    parent_comment_id = Column(UUID(as_uuid=False), ForeignKey("comments.id", ondelete="CASCADE"), nullable=True)  # For nested replies
    is_deleted = Column(Boolean, default=False, nullable=False)  # Soft delete flag (Rule 7)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    post = relationship("Post", back_populates="comments")
    parent_comment = relationship("Comment", remote_side=[id], backref="replies")

    __table_args__ = (
        Index('idx_comments_post_created', 'post_id', 'created_at'),
    )


class Follow(Base):
    """Follow model for user relationships."""
    __tablename__ = "follows"

    id = Column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4()))
    follower_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    following_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint('follower_id', 'following_id', name='uq_follow_follower_following'),
        Index('idx_follows_follower', 'follower_id'),
        Index('idx_follows_following', 'following_id'),
    )

