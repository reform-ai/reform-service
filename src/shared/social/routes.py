"""Social feed routes: posts, likes, comments, follows."""

from fastapi import APIRouter, Depends, HTTPException, status, Query, UploadFile, File, Request
from sqlalchemy.orm import Session
from sqlalchemy import func, or_, and_
from datetime import datetime
from typing import List, Optional
from uuid import uuid4
from pathlib import Path
import os

from src.shared.auth.database import get_db, User
from src.shared.auth.dependencies import get_current_user, require_username_and_verified
from src.shared.social.database import Post, Like, Comment, Follow
from src.shared.social.image_utils import process_image
from src.shared.social.schemas import (
    PostCreate,
    PostResponse,
    CommentCreate,
    CommentResponse,
    LikeResponse,
    FollowResponse,
    PrivacyUpdate,
    PrivacyResponse,
    FeedResponse,
    FollowerInfo,
    FollowersResponse,
    PublicUserProfileResponse
)

router = APIRouter(prefix="/api/social", tags=["social"])

# Image upload directory (similar to OUTPUTS_DIR pattern)
if os.environ.get("DYNO"):  # Heroku
    POST_IMAGES_DIR = Path("/tmp/uploads/posts")
else:
    POST_IMAGES_DIR = Path("uploads/posts")
POST_IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def get_user_by_username(db: Session, username: str) -> Optional[User]:
    """
    Get user by username (case-insensitive lookup).
    
    Usernames are stored in lowercase in the database, so we normalize
    the input to lowercase for consistent case-insensitive matching.
    """
    if not username:
        return None
    normalized_username = username.strip().lower()
    return db.query(User).filter(User.username == normalized_username).first()


def check_user_can_see_posts(viewer_id: str, post_owner: User, db: Session) -> bool:
    """Check if viewer can see posts from post_owner."""
    # Owner can always see their own posts
    if viewer_id == post_owner.id:
        return True
    
    # Public users' posts are visible to everyone
    if post_owner.is_public:
        return True
    
    # Private users' posts only visible to followers
    follow = db.query(Follow).filter(
        and_(
            Follow.follower_id == viewer_id,
            Follow.following_id == post_owner.id
        )
    ).first()
    return follow is not None


def check_user_can_comment(viewer_id: str, post_owner: User, db: Session) -> bool:
    """Check if viewer can comment on posts from post_owner.
    
    Rules:
    - Anyone can comment on public posts (Rule 1)
    - Only followers can comment on private posts (Rule 3)
    """
    # Owner can always comment on their own posts
    if viewer_id == post_owner.id:
        return True
    
    # Public posts: anyone can comment (Rule 1)
    if post_owner.is_public:
        return True
    
    # Private posts: only followers can comment (Rule 3)
    follow = db.query(Follow).filter(
        and_(
            Follow.follower_id == viewer_id,
            Follow.following_id == post_owner.id
        )
    ).first()
    return follow is not None


@router.get("/feed", response_model=FeedResponse)
async def get_feed(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get chronological feed of posts."""
    # Get all users the current user follows
    following_ids_query = db.query(Follow.following_id).filter(
        Follow.follower_id == current_user.id
    )
    following_ids_list = [row[0] for row in following_ids_query.all()]
    
    # Build filter conditions
    # Posts are visible if:
    # 1. User is currently public (regardless of when post was created)
    # 2. User is private BUT we follow them
    # 3. It's our own post
    conditions = [
        User.is_public == True,  # Public users' posts (checks current privacy setting)
        Post.user_id == current_user.id  # Our own posts
    ]
    
    # Add condition for users we follow (only if we follow anyone)
    if following_ids_list:
        conditions.append(Post.user_id.in_(following_ids_list))
    
    query = db.query(Post).join(User, Post.user_id == User.id).filter(
        or_(*conditions)
    ).order_by(Post.created_at.desc())
    
    total = query.count()
    posts = query.offset(offset).limit(limit).all()
    
    # Get like counts, comment counts, and whether current user liked each post
    post_ids = [str(post.id) for post in posts]
    
    like_counts = db.query(
        Like.post_id,
        func.count(Like.id).label('count')
    ).filter(Like.post_id.in_(post_ids)).group_by(Like.post_id).all()
    like_count_map = {str(post_id): count for post_id, count in like_counts}
    
    comment_counts = db.query(
        Comment.post_id,
        func.count(Comment.id).label('count')
    ).filter(Comment.post_id.in_(post_ids)).group_by(Comment.post_id).all()
    comment_count_map = {str(post_id): count for post_id, count in comment_counts}
    
    user_likes = db.query(Like.post_id).filter(
        and_(
            Like.post_id.in_(post_ids),
            Like.user_id == current_user.id
        )
    ).all()
    liked_post_ids = {str(post_id[0]) for post_id in user_likes}
    
    # Get usernames and emails for all post owners
    user_ids = list(set(post.user_id for post in posts))
    users = db.query(User).filter(User.id.in_(user_ids)).all()
    username_map = {user.id: user.username for user in users}
    is_pt_map = {user.id: user.is_pt for user in users}  # Map user IDs to is_pt status
    
    # Build response
    post_responses = []
    for post in posts:
        post_responses.append(PostResponse(
            id=str(post.id),
            user_id=post.user_id,
            username=username_map.get(post.user_id),
            is_pt=is_pt_map.get(post.user_id, False),  # Use database field instead of email
            post_type=post.post_type,
            content=post.content,
            analysis_id=post.analysis_id,
            score_data=post.score_data,
            plot_config=post.plot_config,
            image_urls=post.image_urls,  # Include image URLs
            thumbnail_urls=post.thumbnail_urls,  # Include thumbnail URLs
            created_at=post.created_at,
            updated_at=post.updated_at,
            like_count=like_count_map.get(str(post.id), 0),
            comment_count=comment_count_map.get(str(post.id), 0),
            is_liked=str(post.id) in liked_post_ids
        ))
    
    return FeedResponse(
        posts=post_responses,
        total=total,
        limit=limit,
        offset=offset,
        has_more=(offset + limit) < total
    )


@router.post("/posts/images/upload")
async def upload_post_image(
    request: Request,
    image: UploadFile = File(...),
    current_user: User = Depends(require_username_and_verified),
):
    """
    Upload an image for a post.
    Returns image URL and optional thumbnail URL.
    """
    try:
        image_url, thumbnail_url = process_image(image, POST_IMAGES_DIR)
        
        # Construct full URLs
        if os.environ.get("DYNO"):  # Heroku
            scheme = request.headers.get("X-Forwarded-Proto", "https")
            host = request.headers.get("Host", request.url.hostname)
            base_url = f"{scheme}://{host}"
        else:  # Localhost
            base_url = str(request.base_url).rstrip('/')
        
        full_image_url = f"{base_url}{image_url}"
        full_thumbnail_url = f"{base_url}{thumbnail_url}" if thumbnail_url else None
        
        return {
            "image_url": full_image_url,
            "thumbnail_url": full_thumbnail_url
        }
    except HTTPException:
        raise
    except Exception as e:
        import logging
        logging.error(f"Error uploading image: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload image"
        )


@router.post("/posts", response_model=PostResponse, status_code=status.HTTP_201_CREATED)
async def create_post(
    post_data: PostCreate,
    current_user: User = Depends(require_username_and_verified),
    db: Session = Depends(get_db)
):
    """Create a new post."""
    
    # Create post
    post = Post(
        id=str(uuid4()),
        user_id=current_user.id,
        post_type=post_data.post_type.value,
        content=post_data.content,
        analysis_id=post_data.analysis_id,
        score_data=post_data.score_data,
        plot_config=post_data.plot_config,
        image_urls=post_data.image_urls,  # Store image URLs
        thumbnail_urls=post_data.thumbnail_urls,  # Store thumbnail URLs
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    
    db.add(post)
    db.commit()
    db.refresh(post)
    
    return PostResponse(
        id=str(post.id),
        user_id=post.user_id,
        username=current_user.username,
        is_pt=current_user.is_pt,  # Use database field instead of email
        post_type=post.post_type,
        content=post.content,
        analysis_id=post.analysis_id,
        score_data=post.score_data,
        plot_config=post.plot_config,
        image_urls=post.image_urls,  # Include image URLs in response
        thumbnail_urls=post.thumbnail_urls,  # Include thumbnail URLs in response
        created_at=post.created_at,
        updated_at=post.updated_at,
        like_count=0,
        comment_count=0,
        is_liked=False
    )


@router.get("/posts/{post_id}", response_model=PostResponse)
async def get_post(
    post_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get a single post by ID."""
    post = db.query(Post).filter(Post.id == post_id).first()
    if not post:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Post not found"
        )
    
    # Check if user can see this post
    post_owner = db.query(User).filter(User.id == post.user_id).first()
    if not check_user_can_see_posts(current_user.id, post_owner, db):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to view this post"
        )
    
    # Get counts and like status
    like_count = db.query(func.count(Like.id)).filter(Like.post_id == post_id).scalar() or 0
    comment_count = db.query(func.count(Comment.id)).filter(Comment.post_id == post_id).scalar() or 0
    is_liked = db.query(Like).filter(
        and_(Like.post_id == post_id, Like.user_id == current_user.id)
    ).first() is not None
    
    return PostResponse(
        id=str(post.id),
        user_id=post.user_id,
        username=post_owner.username,
        is_pt=post_owner.is_pt,  # Use database field instead of email
        post_type=post.post_type,
        content=post.content,
        analysis_id=post.analysis_id,
        score_data=post.score_data,
        plot_config=post.plot_config,
        image_urls=post.image_urls,  # Include image URLs
        created_at=post.created_at,
        updated_at=post.updated_at,
        like_count=like_count,
        comment_count=comment_count,
        is_liked=is_liked
    )


@router.delete("/posts/{post_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_post(
    post_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete own post."""
    post = db.query(Post).filter(Post.id == post_id).first()
    if not post:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Post not found"
        )
    
    if post.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only delete your own posts"
        )
    
    db.delete(post)
    db.commit()
    return None


@router.post("/posts/{post_id}/like", response_model=dict)
async def toggle_like(
    post_id: str,
    current_user: User = Depends(require_username_and_verified),
    db: Session = Depends(get_db)
):
    """Toggle like on a post."""
    
    # Check if post exists and user can see it
    post = db.query(Post).filter(Post.id == post_id).first()
    if not post:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Post not found"
        )
    
    post_owner = db.query(User).filter(User.id == post.user_id).first()
    if not check_user_can_see_posts(current_user.id, post_owner, db):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to interact with this post"
        )
    
    # Check if already liked
    existing_like = db.query(Like).filter(
        and_(Like.post_id == post_id, Like.user_id == current_user.id)
    ).first()
    
    if existing_like:
        # Unlike
        db.delete(existing_like)
        db.commit()
        return {"liked": False, "message": "Post unliked"}
    else:
        # Like
        like = Like(
            id=str(uuid4()),
            post_id=post_id,
            user_id=current_user.id,
            created_at=datetime.utcnow()
        )
        db.add(like)
        db.commit()
        return {"liked": True, "message": "Post liked"}


@router.get("/posts/{post_id}/comments", response_model=List[CommentResponse])
async def get_comments(
    post_id: str,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get comments for a post."""
    # Check if post exists and user can see it
    post = db.query(Post).filter(Post.id == post_id).first()
    if not post:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Post not found"
        )
    
    post_owner = db.query(User).filter(User.id == post.user_id).first()
    if not check_user_can_see_posts(current_user.id, post_owner, db):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to view this post"
        )
    
    # Get all comments (including replies) for this post, ordered by newest first
    # Include deleted comments (Rule 7: show "User deleted this comment" instead of removing)
    all_comments = db.query(Comment).filter(
        Comment.post_id == post_id
    ).order_by(Comment.created_at.desc()).all()
    
    # Get usernames and is_pt status for all commenters
    user_ids = list(set(comment.user_id for comment in all_comments))
    users = db.query(User).filter(User.id.in_(user_ids)).all()
    username_map = {user.id: user.username for user in users}
    is_pt_map = {user.id: user.is_pt for user in users}  # Map user IDs to is_pt status
    
    # Separate top-level comments and replies
    top_level_comments = [c for c in all_comments if c.parent_comment_id is None]
    replies = {str(c.parent_comment_id): [] for c in all_comments if c.parent_comment_id is not None}
    for comment in all_comments:
        if comment.parent_comment_id:
            parent_id = str(comment.parent_comment_id)
            if parent_id not in replies:
                replies[parent_id] = []
            replies[parent_id].append(comment)
    
    # Sort replies by newest first (they're already in desc order from query, but ensure consistency)
    for parent_id in replies:
        replies[parent_id].sort(key=lambda x: x.created_at, reverse=True)
    
    # Build response with nested structure (newest top-level comments first)
    result = []
    for comment in top_level_comments[:limit]:  # Apply limit to top-level only
        comment_replies = replies.get(str(comment.id), [])
        # Rule 7: Show "User deleted this comment" for deleted comments
        display_content = "User deleted this comment" if comment.is_deleted else comment.content
        result.append(CommentResponse(
            id=str(comment.id),
            post_id=str(comment.post_id),
            user_id=comment.user_id,
            username=username_map.get(comment.user_id),
            is_pt=is_pt_map.get(comment.user_id, False),  # Use database field instead of email
            content=display_content,
            parent_comment_id=None,
            created_at=comment.created_at,
            updated_at=comment.updated_at,
            reply_count=len(comment_replies)
        ))
        # Add replies for this comment (newest first)
        for reply in comment_replies:
            # Rule 7: Show "User deleted this comment" for deleted replies
            reply_display_content = "User deleted this comment" if reply.is_deleted else reply.content
            result.append(CommentResponse(
                id=str(reply.id),
                post_id=str(reply.post_id),
                user_id=reply.user_id,
                username=username_map.get(reply.user_id),
                is_pt=is_pt_map.get(reply.user_id, False),  # Use database field instead of email
                content=reply_display_content,
                parent_comment_id=str(reply.parent_comment_id),
                created_at=reply.created_at,
                updated_at=reply.updated_at,
                reply_count=0  # Replies can't have replies (max 2 levels)
            ))
    
    return result


@router.post("/posts/{post_id}/comments", response_model=CommentResponse, status_code=status.HTTP_201_CREATED)
async def create_comment(
    post_id: str,
    comment_data: CommentCreate,
    current_user: User = Depends(require_username_and_verified),
    db: Session = Depends(get_db)
):
    """Create a comment on a post."""
    
    # Check if post exists and user can see it
    post = db.query(Post).filter(Post.id == post_id).first()
    if not post:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Post not found"
        )
    
    post_owner = db.query(User).filter(User.id == post.user_id).first()
    # Check if user can comment (Rules 1 & 3: public posts anyone can comment, private posts only followers)
    if not check_user_can_comment(current_user.id, post_owner, db):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to comment on this post. Only followers can comment on private posts."
        )
    
    # Validate parent comment if provided
    if comment_data.parent_comment_id:
        parent = db.query(Comment).filter(Comment.id == comment_data.parent_comment_id).first()
        if not parent or parent.post_id != post_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid parent comment"
            )
    
    comment = Comment(
        id=str(uuid4()),
        post_id=post_id,
        user_id=current_user.id,
        content=comment_data.content,
        parent_comment_id=comment_data.parent_comment_id,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    
    db.add(comment)
    db.commit()
    db.refresh(comment)
    
    return CommentResponse(
        id=str(comment.id),
        post_id=str(comment.post_id),
        user_id=comment.user_id,
        username=current_user.username,
        is_pt=current_user.is_pt,  # Use database field instead of email
        content=comment.content,
        parent_comment_id=str(comment.parent_comment_id) if comment.parent_comment_id else None,
        created_at=comment.created_at,
        updated_at=comment.updated_at,
        reply_count=0
    )


@router.delete("/comments/{comment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_comment(
    comment_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Soft delete own comment (Rule 7: shows 'User deleted this comment' instead of removing)."""
    comment = db.query(Comment).filter(Comment.id == comment_id).first()
    if not comment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Comment not found"
        )
    
    if comment.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only delete your own comments"
        )
    
    # Rule 7: Soft delete - mark as deleted instead of removing
    comment.is_deleted = True
    comment.updated_at = datetime.utcnow()
    db.commit()
    return None


@router.get("/users/{username}/follow", response_model=dict)
async def get_follow_status(
    username: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get follow status for a user."""
    target_user = get_user_by_username(db, username)
    if not target_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Check if already following
    existing_follow = db.query(Follow).filter(
        and_(
            Follow.follower_id == current_user.id,
            Follow.following_id == target_user.id
        )
    ).first()
    
    return {"following": existing_follow is not None}


@router.post("/users/{username}/follow", response_model=dict)
async def toggle_follow(
    username: str,
    current_user: User = Depends(require_username_and_verified),
    db: Session = Depends(get_db)
):
    """Follow or unfollow a user."""
    
    target_user = get_user_by_username(db, username)
    if not target_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    if target_user.id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You cannot follow yourself"
        )
    
    # Check if already following
    existing_follow = db.query(Follow).filter(
        and_(
            Follow.follower_id == current_user.id,
            Follow.following_id == target_user.id
        )
    ).first()
    
    if existing_follow:
        # Unfollow
        db.delete(existing_follow)
        db.commit()
        return {"following": False, "message": f"Unfollowed {username}"}
    else:
        # Follow
        follow = Follow(
            id=str(uuid4()),
            follower_id=current_user.id,
            following_id=target_user.id,
            created_at=datetime.utcnow()
        )
        db.add(follow)
        db.commit()
        return {"following": True, "message": f"Following {username}"}


@router.get("/users/{username}/posts", response_model=List[PostResponse])
async def get_user_posts(
    username: str,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get posts by a specific user."""
    target_user = get_user_by_username(db, username)
    if not target_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Check if user can see posts
    if not check_user_can_see_posts(current_user.id, target_user, db):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This user's posts are private"
        )
    
    posts = db.query(Post).filter(
        Post.user_id == target_user.id
    ).order_by(Post.created_at.desc()).offset(offset).limit(limit).all()
    
    # Get counts and like status
    post_ids = [str(post.id) for post in posts]
    
    like_counts = db.query(
        Like.post_id,
        func.count(Like.id).label('count')
    ).filter(Like.post_id.in_(post_ids)).group_by(Like.post_id).all()
    like_count_map = {str(post_id): count for post_id, count in like_counts}
    
    comment_counts = db.query(
        Comment.post_id,
        func.count(Comment.id).label('count')
    ).filter(Comment.post_id.in_(post_ids)).group_by(Comment.post_id).all()
    comment_count_map = {str(post_id): count for post_id, count in comment_counts}
    
    user_likes = db.query(Like.post_id).filter(
        and_(
            Like.post_id.in_(post_ids),
            Like.user_id == current_user.id
        )
    ).all()
    liked_post_ids = {str(post_id[0]) for post_id in user_likes}
    
    return [
        PostResponse(
            id=str(post.id),
            user_id=post.user_id,
            username=target_user.username,
            is_pt=target_user.is_pt,  # Use database field instead of email
            post_type=post.post_type,
            content=post.content,
            analysis_id=post.analysis_id,
            score_data=post.score_data,
            plot_config=post.plot_config,
            image_urls=post.image_urls,  # Include image URLs
            created_at=post.created_at,
            updated_at=post.updated_at,
            like_count=like_count_map.get(str(post.id), 0),
            comment_count=comment_count_map.get(str(post.id), 0),
            is_liked=str(post.id) in liked_post_ids
        )
        for post in posts
    ]


@router.get("/users/{username}/profile", response_model=PublicUserProfileResponse)
async def get_user_profile(
    username: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get public profile for a user by username.
    
    Privacy rules (hierarchy):
    1. full_name and email are NEVER exposed to other users
    2. username and preferences (technical_level, favorite_exercise, community_preference) are always visible
    3. posts are only visible if:
       - User is public (is_public=True), OR
       - User is private (is_public=False) but current_user follows them
    """
    # Get target user
    target_user = get_user_by_username(db, username)
    if not target_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Check if viewer can see posts (using existing helper function)
    can_see_posts = check_user_can_see_posts(current_user.id, target_user, db)
    
    # Build base response - always include username and preferences
    # Rule 1: NEVER include full_name or email
    response_data = {
        "username": target_user.username,
        "is_pt": target_user.is_pt,
        "technical_level": target_user.technical_level,
        "favorite_exercise": target_user.favorite_exercise,
        "community_preference": target_user.community_preference,
        "is_public": target_user.is_public,
        "can_see_posts": can_see_posts,
        "posts": None
    }
    
    # Rule 3: Only include posts if viewer can see them
    if can_see_posts:
        # Get user's posts (same logic as get_user_posts endpoint)
        posts = db.query(Post).filter(
            Post.user_id == target_user.id
        ).order_by(Post.created_at.desc()).limit(20).all()  # Limit to recent 20 posts
        
        # Get counts and like status for posts
        post_ids = [str(post.id) for post in posts]
        
        if post_ids:
            like_counts = db.query(
                Like.post_id,
                func.count(Like.id).label('count')
            ).filter(Like.post_id.in_(post_ids)).group_by(Like.post_id).all()
            like_count_map = {str(post_id): count for post_id, count in like_counts}
            
            comment_counts = db.query(
                Comment.post_id,
                func.count(Comment.id).label('count')
            ).filter(Comment.post_id.in_(post_ids)).group_by(Comment.post_id).all()
            comment_count_map = {str(post_id): count for post_id, count in comment_counts}
            
            user_likes = db.query(Like.post_id).filter(
                and_(
                    Like.post_id.in_(post_ids),
                    Like.user_id == current_user.id
                )
            ).all()
            liked_post_ids = {str(post_id[0]) for post_id in user_likes}
            
            # Build post responses (without exposing email/full_name)
            response_data["posts"] = [
                PostResponse(
                    id=str(post.id),
                    user_id=post.user_id,
                    username=target_user.username,
                    is_pt=target_user.is_pt,
                    post_type=post.post_type,
                    content=post.content,
                    analysis_id=post.analysis_id,
                    score_data=post.score_data,
                    plot_config=post.plot_config,
                    image_urls=post.image_urls,  # Include image URLs
                    created_at=post.created_at,
                    updated_at=post.updated_at,
                    like_count=like_count_map.get(str(post.id), 0),
                    comment_count=comment_count_map.get(str(post.id), 0),
                    is_liked=str(post.id) in liked_post_ids
                )
                for post in posts
            ]
        else:
            response_data["posts"] = []
    
    return PublicUserProfileResponse(**response_data)


@router.patch("/users/me/privacy", response_model=PrivacyResponse)
async def update_privacy(
    privacy_data: PrivacyUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update user privacy setting."""
    current_user.is_public = privacy_data.is_public
    db.commit()
    db.refresh(current_user)
    
    return PrivacyResponse(is_public=current_user.is_public)


@router.get("/users/me/privacy", response_model=PrivacyResponse)
async def get_privacy(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get current user privacy setting."""
    return PrivacyResponse(is_public=current_user.is_public)


@router.get("/users/me/followers", response_model=FollowersResponse)
async def get_my_followers(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get list of users who follow the current user."""
    # Get all follows where current_user is being followed
    follows = db.query(Follow).filter(
        Follow.following_id == current_user.id
    ).all()
    
    # Get user IDs of followers
    follower_ids = [follow.follower_id for follow in follows]
    
    if not follower_ids:
        return FollowersResponse(followers=[], total=0)
    
    # Get user details for all followers
    followers = db.query(User).filter(User.id.in_(follower_ids)).all()
    
    # Get list of users that current_user follows (to check is_following_back)
    following_ids = db.query(Follow.following_id).filter(
        Follow.follower_id == current_user.id
    ).all()
    following_ids_set = {row[0] for row in following_ids}
    
    # Build response
    # Never expose email - use username or "Unknown User" as fallback
    follower_list = []
    for user in followers:
        follower_list.append(FollowerInfo(
            id=user.id,
            user_id=user.id,
            username=user.username,
            full_name=user.full_name or user.username or "Unknown User",  # Never expose email
            is_following_back=user.id in following_ids_set
        ))
    
    return FollowersResponse(followers=follower_list, total=len(follower_list))


@router.get("/users/me/following", response_model=FollowersResponse)
async def get_my_following(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get list of users the current user is following."""
    # Get all follows where current_user is the follower
    follows = db.query(Follow).filter(
        Follow.follower_id == current_user.id
    ).all()
    
    # Get user IDs of users being followed
    following_ids = [follow.following_id for follow in follows]
    
    if not following_ids:
        return FollowersResponse(followers=[], total=0)
    
    # Get user details for all users being followed
    following_users = db.query(User).filter(User.id.in_(following_ids)).all()
    
    # Build response (is_following_back is always True since we're following them)
    # Never expose email - use username or "Unknown User" as fallback
    following_list = []
    for user in following_users:
        following_list.append(FollowerInfo(
            id=user.id,
            user_id=user.id,
            username=user.username,
            full_name=user.full_name or user.username or "Unknown User",  # Never expose email
            is_following_back=True  # Always true since we're following them
        ))
    
    return FollowersResponse(followers=following_list, total=len(following_list))

