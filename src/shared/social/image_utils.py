"""Image upload and processing utilities for social posts."""

import os
import uuid
import re
from pathlib import Path
from typing import Tuple, Optional
from fastapi import UploadFile, HTTPException, status
from PIL import Image
import io

# Allowed image MIME types
ALLOWED_IMAGE_TYPES = {
    'image/jpeg': '.jpg',
    'image/jpg': '.jpg',
    'image/png': '.png',
    'image/webp': '.webp',
    'image/gif': '.gif'
}

# Max file size: 10MB per image
MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024

# Thumbnail dimensions
THUMBNAIL_SIZE = (800, 800)  # Max width/height, maintains aspect ratio


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal attacks."""
    if not filename:
        return None
    
    # Remove path components
    filename = os.path.basename(filename)
    filename = filename.replace('/', '').replace('\\', '')
    
    # Extract extension
    name, ext = os.path.splitext(filename)
    
    # Sanitize name: only alphanumeric, dots, hyphens, underscores
    name = re.sub(r'[^a-zA-Z0-9._-]', '_', name)
    ext = re.sub(r'[^a-zA-Z0-9.]', '', ext)
    
    # Limit length
    name = name[:100]
    ext = ext[:10]
    
    # If name is empty, use UUID
    if not name or name == '_':
        name = str(uuid.uuid4())[:8]
    
    return name + ext if ext else name


def validate_image_file(file: UploadFile) -> None:
    """Validate image file type and size."""
    # Check content type
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Allowed types: {', '.join(ALLOWED_IMAGE_TYPES.keys())}"
        )
    
    # Check file size (read first chunk to check)
    file.file.seek(0, os.SEEK_END)
    file_size = file.file.tell()
    file.file.seek(0)
    
    if file_size > MAX_IMAGE_SIZE_BYTES:
        size_mb = file_size / (1024 * 1024)
        max_mb = MAX_IMAGE_SIZE_BYTES / (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image too large ({size_mb:.1f}MB). Maximum size: {max_mb}MB"
        )
    
    if file_size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image file is empty"
        )


def process_image(file: UploadFile, upload_dir: Path) -> Tuple[str, Optional[str]]:
    """
    Process and save image, generating thumbnail if needed.
    
    Returns:
        Tuple of (image_url, thumbnail_url)
        thumbnail_url is None if image is already small enough
    """
    # Validate file
    validate_image_file(file)
    
    # Read image data
    file.file.seek(0)
    image_data = file.file.read()
    
    # Verify it's a valid image using PIL
    try:
        image = Image.open(io.BytesIO(image_data))
        # Convert to RGB if necessary (handles RGBA, P, etc.)
        if image.mode in ('RGBA', 'LA', 'P'):
            # Create white background for transparency
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'P':
                image = image.convert('RGBA')
            rgb_image.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
            image = rgb_image
        elif image.mode != 'RGB':
            image = image.convert('RGB')
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image file: {str(e)}"
        )
    
    # Generate unique filename
    file_ext = ALLOWED_IMAGE_TYPES.get(file.content_type, '.jpg')
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    
    # Ensure upload directory exists
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Save original image
    original_path = upload_dir / unique_filename
    image.save(original_path, 'JPEG', quality=85, optimize=True)
    
    # Generate thumbnail if image is large
    thumbnail_url = None
    if image.width > THUMBNAIL_SIZE[0] or image.height > THUMBNAIL_SIZE[1]:
        thumbnail_filename = f"{uuid.uuid4()}_thumb.jpg"
        thumbnail_path = upload_dir / thumbnail_filename
        
        # Create thumbnail maintaining aspect ratio (use copy to preserve original)
        thumbnail_image = image.copy()
        thumbnail_image.thumbnail(THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
        thumbnail_image.save(thumbnail_path, 'JPEG', quality=85, optimize=True)
        thumbnail_url = f"/uploads/posts/{thumbnail_filename}"
    
    image_url = f"/uploads/posts/{unique_filename}"
    
    return image_url, thumbnail_url

