"""Pydantic schemas for social feed API."""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class PostType(str, Enum):
    """Post type enumeration."""
    SCORE = "score"
    TEXT = "text"
    PLOT = "plot"


class PostCreate(BaseModel):
    """Schema for creating a post."""
    post_type: PostType
    content: Optional[str] = None  # Caption or text post content
    analysis_id: Optional[str] = None  # Link to analysis if sharing from analysis
    score_data: Optional[Dict[str, Any]] = None  # Snapshot of score details
    plot_config: Optional[Dict[str, Any]] = None  # Config to recreate plots


class PostResponse(BaseModel):
    """Schema for post response."""
    id: str
    user_id: str
    username: Optional[str] = None  # Will be populated from user
    is_pt: bool = False  # Whether the post author is a verified Personal Trainer
    post_type: str
    content: Optional[str] = None
    analysis_id: Optional[str] = None
    score_data: Optional[Dict[str, Any]] = None
    plot_config: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime
    like_count: int = 0
    comment_count: int = 0
    is_liked: bool = False  # Whether current user has liked this post
    is_following: Optional[bool] = None  # Whether current user is following the post author (optional, not always included)

    class Config:
        from_attributes = True


class CommentCreate(BaseModel):
    """Schema for creating a comment."""
    content: str = Field(..., min_length=1, max_length=1000)
    parent_comment_id: Optional[str] = None  # For nested replies


class CommentResponse(BaseModel):
    """Schema for comment response."""
    id: str
    post_id: str
    user_id: str
    username: Optional[str] = None  # Will be populated from user
    is_pt: bool = False  # Whether the comment author is a verified Personal Trainer
    content: str
    parent_comment_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    reply_count: int = 0  # Number of replies (for future nested comments)

    class Config:
        from_attributes = True


class LikeResponse(BaseModel):
    """Schema for like response."""
    post_id: str
    user_id: str
    username: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class FollowResponse(BaseModel):
    """Schema for follow response."""
    follower_id: str
    follower_username: Optional[str] = None
    following_id: str
    following_username: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class PrivacyUpdate(BaseModel):
    """Schema for updating privacy setting."""
    is_public: bool


class PrivacyResponse(BaseModel):
    """Schema for privacy setting response."""
    is_public: bool


class FeedResponse(BaseModel):
    """Schema for feed response with pagination."""
    posts: List[PostResponse]
    total: int
    limit: int
    offset: int
    has_more: bool


class FollowerInfo(BaseModel):
    """Schema for follower/following user information."""
    id: str
    user_id: Optional[str] = None  # For backward compatibility
    username: Optional[str] = None
    full_name: str
    is_following_back: bool = False  # Whether current user is following this follower back

    class Config:
        from_attributes = True


class FollowersResponse(BaseModel):
    """Schema for followers list response."""
    followers: List[FollowerInfo]
    total: int


class PublicUserProfileResponse(BaseModel):
    """Schema for public user profile response.
    
    Privacy rules:
    - full_name and email are NEVER exposed to other users
    - username and preferences (technical_level, favorite_exercise, community_preference) are always visible
    - posts are only visible if user is public OR if viewer follows the user
    """
    username: Optional[str] = None
    is_pt: bool = False  # Whether the user is a verified Personal Trainer
    technical_level: Optional[str] = None  # answer1
    favorite_exercise: Optional[str] = None  # answer2
    community_preference: Optional[str] = None  # answer3
    is_public: bool
    posts: Optional[List[PostResponse]] = None  # Only included if visible (public or followed)
    can_see_posts: bool  # Whether the viewer can see posts

