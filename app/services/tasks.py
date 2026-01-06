"""Celery tasks for asynchronous experiment execution."""

import logging
from typing import Optional

from celery import Celery

from app.api.schemas.experiments import ExperimentResults
from app.core.config import get_settings
from app.db.experiments import get_experiment
from app.services.experiment_runner import ExperimentRunner

logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()

# Create Celery app
celery_app = Celery(
    "scientific_api",
    broker=settings.get_celery_broker_url(),
    backend=settings.get_celery_result_backend(),
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=7200,  # 2 hours max per task
    task_soft_time_limit=6600,  # 1h 50min soft limit
    worker_prefetch_multiplier=1,  # One task at a time per worker
    worker_max_tasks_per_child=10,  # Restart worker after 10 tasks (memory cleanup)
)


@celery_app.task(
    name="run_experiment_pipeline",
    bind=True,
    max_retries=2,
    default_retry_delay=300,  # 5 minutes
)
def run_experiment_pipeline_task(self, experiment_id: str) -> dict:
    """
    Celery task to run experiment pipeline asynchronously.
    
    Args:
        self: Celery task instance (injected by bind=True)
        experiment_id: Experiment ID
        
    Returns:
        dict: Serialized ExperimentResults
        
    Raises:
        Exception: If pipeline execution fails after retries
    """
    logger.info(f"[CELERY TASK] Starting experiment {experiment_id}")
    
    try:
        # Get experiment document to extract config
        import asyncio
        document = asyncio.run(get_experiment(experiment_id))
        
        if not document:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        # Parse config
        from app.api.schemas.experiments import ExperimentConfig
        config = ExperimentConfig(**document["config"])
        
        # Run pipeline
        runner = ExperimentRunner(experiment_id, config)
        results = asyncio.run(runner.run_full_pipeline())
        
        logger.info(f"[CELERY TASK] Experiment {experiment_id} completed successfully")
        
        # Return serialized results
        return results.model_dump()
        
    except Exception as e:
        logger.error(f"[CELERY TASK] Experiment {experiment_id} failed: {e}", exc_info=True)
        
        # Retry logic
        if self.request.retries < self.max_retries:
            logger.warning(f"Retrying experiment {experiment_id} (attempt {self.request.retries + 1})")
            raise self.retry(exc=e)
        
        # Max retries exhausted
        logger.error(f"Experiment {experiment_id} failed after {self.max_retries} retries")
        raise


@celery_app.task(name="cleanup_experiment_data", bind=True)
def cleanup_experiment_data_task(self, experiment_id: str, keep_models: bool = True) -> dict:
    """
    Celery task to cleanup experiment data files.
    
    Args:
        self: Celery task instance
        experiment_id: Experiment ID
        keep_models: Whether to keep trained models (default: True)
        
    Returns:
        dict: Cleanup statistics
    """
    logger.info(f"[CELERY TASK] Cleaning up experiment {experiment_id} data")
    
    import shutil
    from pathlib import Path
    
    settings = get_settings()
    experiment_dir = settings.DATA_ROOT / "experiments" / experiment_id
    
    if not experiment_dir.exists():
        logger.warning(f"Experiment directory {experiment_dir} does not exist")
        return {"deleted": False, "reason": "directory_not_found"}
    
    stats = {
        "deleted": True,
        "graphs_deleted": False,
        "features_deleted": False,
        "models_deleted": False,
        "distances_deleted": False,
    }
    
    try:
        # Delete graphs
        graphs_dir = experiment_dir / "graphs"
        if graphs_dir.exists():
            shutil.rmtree(graphs_dir)
            stats["graphs_deleted"] = True
        
        # Delete features
        features_dir = experiment_dir / "features"
        if features_dir.exists():
            shutil.rmtree(features_dir)
            stats["features_deleted"] = True
        
        # Delete distances
        distances_dir = experiment_dir / "distances"
        if distances_dir.exists():
            shutil.rmtree(distances_dir)
            stats["distances_deleted"] = True
        
        # Delete models (optional)
        if not keep_models:
            models_dir = experiment_dir / "models"
            if models_dir.exists():
                shutil.rmtree(models_dir)
                stats["models_deleted"] = True
        
        logger.info(f"Cleanup completed for experiment {experiment_id}: {stats}")
        return stats
        
    except Exception as e:
        logger.error(f"Failed to cleanup experiment {experiment_id}: {e}", exc_info=True)
        return {"deleted": False, "error": str(e)}


@celery_app.task(name="batch_run_experiments", bind=True)
def batch_run_experiments_task(self, experiment_ids: list[str]) -> dict:
    """
    Celery task to run multiple experiments in sequence.
    
    Args:
        self: Celery task instance
        experiment_ids: List of experiment IDs
        
    Returns:
        dict: Results summary
    """
    logger.info(f"[CELERY TASK] Running batch of {len(experiment_ids)} experiments")
    
    results = {
        "total": len(experiment_ids),
        "completed": 0,
        "failed": 0,
        "experiment_results": {},
    }
    
    for exp_id in experiment_ids:
        try:
            # Run experiment pipeline
            result = run_experiment_pipeline_task(exp_id)
            results["completed"] += 1
            results["experiment_results"][exp_id] = {"status": "completed"}
            logger.info(f"Batch experiment {exp_id} completed")
            
        except Exception as e:
            results["failed"] += 1
            results["experiment_results"][exp_id] = {
                "status": "failed",
                "error": str(e)
            }
            logger.error(f"Batch experiment {exp_id} failed: {e}")
    
    logger.info(f"Batch run completed: {results['completed']} succeeded, {results['failed']} failed")
    return results


# Periodic tasks (if using Celery Beat)

@celery_app.task(name="cleanup_old_experiments")
def cleanup_old_experiments_task(days: int = 30) -> dict:
    """
    Periodic task to cleanup experiments older than specified days.
    
    Args:
        days: Age threshold in days
        
    Returns:
        dict: Cleanup statistics
    """
    logger.info(f"[CELERY TASK] Cleaning up experiments older than {days} days")
    
    import asyncio
    from datetime import datetime, timedelta
    from app.db.experiments import list_experiments, delete_experiment
    
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    
    # Get old experiments
    experiments = asyncio.run(list_experiments(limit=1000))
    
    deleted_count = 0
    failed_count = 0
    
    for exp in experiments:
        created_at = exp.get("created_at")
        if created_at and created_at < cutoff_date:
            exp_id = exp["id"]
            
            # Delete from DB
            success = asyncio.run(delete_experiment(exp_id))
            
            if success:
                # Delete files
                cleanup_experiment_data_task(exp_id, keep_models=False)
                deleted_count += 1
                logger.info(f"Deleted old experiment {exp_id}")
            else:
                failed_count += 1
                logger.warning(f"Failed to delete experiment {exp_id}")
    
    results = {
        "cutoff_date": cutoff_date.isoformat(),
        "deleted": deleted_count,
        "failed": failed_count,
    }
    
    logger.info(f"Cleanup completed: {results}")
    return results


# Example Celery Beat schedule (add to settings if using periodic tasks)
"""
celery_app.conf.beat_schedule = {
    "cleanup-old-experiments-weekly": {
        "task": "cleanup_old_experiments",
        "schedule": crontab(day_of_week=0, hour=2, minute=0),  # Every Sunday at 2 AM
        "args": (30,),  # 30 days threshold
    },
}
"""
