"""Analysis history routes: list, get, and progress tracking.

This module provides API endpoints for:
- Listing analyses with filters (exercise, score range, date range)
- Getting single analysis details
- Getting progress metrics and trends
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import and_, func, desc
from datetime import datetime, timedelta
from typing import Optional

from src.shared.auth.database import get_db, User
from src.shared.auth.dependencies import get_current_user
from src.shared.analyses.database import Analysis
from src.shared.analyses.schemas import (
    AnalysisResponse,
    AnalysisListResponse,
    AnalysisListItem,
    ProgressMetrics,
    UpdateNotesRequest
)

router = APIRouter(prefix="/api/analyses", tags=["analyses"])


def _remove_microseconds(dt: Optional[datetime]) -> Optional[datetime]:
    """Remove microseconds from datetime for consistent display."""
    return dt.replace(microsecond=0) if dt else dt


def _build_analysis_response(analysis: Analysis) -> AnalysisResponse:
    """Build AnalysisResponse from Analysis model."""
    return AnalysisResponse(
        id=str(analysis.id),
        user_id=analysis.user_id,
        exercise=analysis.exercise,
        exercise_name=analysis.exercise_name,
        score=analysis.score,
        frame_count=analysis.frame_count,
        fps=analysis.fps,
        calculation_results=analysis.calculation_results,
        form_analysis=analysis.form_analysis,
        camera_angle_info=analysis.camera_angle_info,
        phases=analysis.phases,
        visualization_url=analysis.visualization_url,
        visualization_filename=analysis.visualization_filename,
        file_size=analysis.file_size,
        notes=analysis.notes,
        created_at=_remove_microseconds(analysis.created_at),
        updated_at=_remove_microseconds(analysis.updated_at)
    )


def _parse_date_filter(date_str: str, is_end_date: bool = False) -> datetime:
    """
    Parse date string (YYYY-MM-DD) to datetime.
    
    Args:
        date_str: Date string in YYYY-MM-DD format
        is_end_date: If True, includes entire day (up to 23:59:59)
    
    Returns:
        datetime object
    
    Raises:
        HTTPException: If date format is invalid
    """
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        if is_end_date:
            # Include the entire end date (up to 23:59:59)
            date_obj = date_obj + timedelta(days=1) - timedelta(seconds=1)
        return date_obj
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid date format. Use YYYY-MM-DD"
        )


@router.get("", response_model=AnalysisListResponse)
async def list_analyses(
    limit: int = Query(20, ge=1, le=100, description="Number of analyses to return"),
    offset: int = Query(0, ge=0, description="Number of analyses to skip"),
    exercise: Optional[int] = Query(None, description="Filter by exercise type (1=Squat, 2=Bench, 3=Deadlift)"),
    min_score: Optional[int] = Query(None, ge=0, le=100, description="Minimum score filter"),
    max_score: Optional[int] = Query(None, ge=0, le=100, description="Maximum score filter"),
    start_date: Optional[str] = Query(None, description="Start date filter (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date filter (YYYY-MM-DD)"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get list of analyses for the current user with optional filters.
    
    Filters:
    - exercise: Filter by exercise type
    - min_score/max_score: Filter by score range
    - start_date/end_date: Filter by date range (YYYY-MM-DD format)
    """
    # Build query filters
    filters = [Analysis.user_id == current_user.id]
    
    if exercise is not None:
        filters.append(Analysis.exercise == exercise)
    
    if min_score is not None:
        filters.append(Analysis.score >= min_score)
    
    if max_score is not None:
        filters.append(Analysis.score <= max_score)
    
    if start_date:
        start_datetime = _parse_date_filter(start_date, is_end_date=False)
        filters.append(Analysis.created_at >= start_datetime)
    
    if end_date:
        end_datetime = _parse_date_filter(end_date, is_end_date=True)
        filters.append(Analysis.created_at <= end_datetime)
    
    # Get total count
    total = db.query(func.count(Analysis.id)).filter(and_(*filters)).scalar()
    
    # Get analyses (summary only for list view)
    analyses = db.query(Analysis).filter(
        and_(*filters)
    ).order_by(desc(Analysis.created_at)).offset(offset).limit(limit).all()
    
    # Convert to response format
    analysis_items = [
        AnalysisListItem(
            id=str(analysis.id),
            exercise=analysis.exercise,
            exercise_name=analysis.exercise_name,
            score=analysis.score,
            frame_count=analysis.frame_count,
            created_at=_remove_microseconds(analysis.created_at)
        )
        for analysis in analyses
    ]
    
    return AnalysisListResponse(
        analyses=analysis_items,
        total=total,
        limit=limit,
        offset=offset
    )


@router.get("/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis(
    analysis_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get single analysis details by ID."""
    analysis = db.query(Analysis).filter(
        and_(
            Analysis.id == analysis_id,
            Analysis.user_id == current_user.id
        )
    ).first()
    
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found"
        )
    
    return _build_analysis_response(analysis)


@router.get("/progress/metrics", response_model=ProgressMetrics)
async def get_progress_metrics(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get progress metrics and trends for the current user.
    Returns statistics, trends, and recent analyses.
    """
    # Get all analyses for the user
    all_analyses = db.query(Analysis).filter(
        Analysis.user_id == current_user.id
    ).order_by(desc(Analysis.created_at)).all()
    
    total_analyses = len(all_analyses)
    
    if total_analyses == 0:
        return ProgressMetrics(
            total_analyses=0,
            average_score=None,
            best_score=None,
            worst_score=None,
            score_trend=[],
            analyses_by_exercise={},
            recent_analyses=[]
        )
    
    # Calculate statistics
    scores = [a.score for a in all_analyses]
    average_score = sum(scores) / len(scores) if scores else None
    best_score = max(scores) if scores else None
    worst_score = min(scores) if scores else None
    
    # Build score trend: one point per day per exercise, averaged if multiple analyses on same day
    # Group analyses by date (YYYY-MM-DD) and exercise, average scores per day per exercise
    from collections import defaultdict
    
    # Get all analyses to properly group by day and exercise
    # Structure: {(date_str, exercise_name): {"scores": [], "date": None}}
    daily_exercise_scores = defaultdict(lambda: {"scores": [], "date": None})
    
    for analysis in all_analyses:
        if analysis.created_at:
            # Get date string (YYYY-MM-DD) for grouping
            date_str = analysis.created_at.date().isoformat()
            exercise_name = analysis.exercise_name
            key = (date_str, exercise_name)
            
            daily_exercise_scores[key]["scores"].append(analysis.score)
            # Store the datetime for the date field (use first analysis of the day)
            if daily_exercise_scores[key]["date"] is None:
                daily_exercise_scores[key]["date"] = _remove_microseconds(analysis.created_at)
    
    # Convert to list with exercise information
    score_trend = []
    for (date_str, exercise_name), day_data in sorted(daily_exercise_scores.items()):  # Sort chronologically
        avg_score = sum(day_data["scores"]) / len(day_data["scores"])
        score_trend.append({
            "date": day_data["date"].isoformat() if day_data["date"] else None,
            "score": round(avg_score, 1),
            "exercise": exercise_name
        })
    
    # Count analyses by exercise
    analyses_by_exercise = {}
    for analysis in all_analyses:
        exercise_name = analysis.exercise_name
        analyses_by_exercise[exercise_name] = analyses_by_exercise.get(exercise_name, 0) + 1
    
    # Get most recent analysis (last 1)
    recent_analyses = [
        AnalysisListItem(
            id=str(analysis.id),
            exercise=analysis.exercise,
            exercise_name=analysis.exercise_name,
            score=analysis.score,
            frame_count=analysis.frame_count,
            created_at=_remove_microseconds(analysis.created_at)
        )
        for analysis in all_analyses[:1]
    ] if all_analyses else []
    
    return ProgressMetrics(
        total_analyses=total_analyses,
        average_score=round(average_score, 1) if average_score is not None else None,
        best_score=best_score,
        worst_score=worst_score,
        score_trend=score_trend,
        analyses_by_exercise=analyses_by_exercise,
        recent_analyses=recent_analyses
    )


@router.put("/{analysis_id}/notes", response_model=AnalysisResponse)
async def update_analysis_notes(
    analysis_id: str,
    request: UpdateNotesRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update notes for a specific analysis."""
    analysis = db.query(Analysis).filter(
        and_(
            Analysis.id == analysis_id,
            Analysis.user_id == current_user.id
        )
    ).first()
    
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found"
        )
    
    # Validate and sanitize notes
    from src.shared.auth.input_validation import validate_notes
    analysis.notes = validate_notes(request.notes)
    db.commit()
    db.refresh(analysis)
    
    return _build_analysis_response(analysis)

