from typing import Optional
from apscheduler.schedulers.blocking import BlockingScheduler

from sreejita.automation.batch_runner import run_batch
from sreejita.utils.logger import get_logger

log = get_logger("scheduler")


def start_scheduler(
    schedule_config: dict,
    input_dir: str,
    config_path: Optional[str] = None,
    output_root: str = "runs"
) -> None:
    """
    Start time-based automation using APScheduler.

    schedule_config example:
    {
        "hour": 9,
        "minute": 0
    }
    """

    if not isinstance(schedule_config, dict):
        raise ValueError("schedule_config must be a dictionary")

    scheduler = BlockingScheduler()

    log.info("Starting scheduler with config: %s", schedule_config)

    scheduler.add_job(
        run_batch,
        trigger="cron",
        id="sreejita-batch-job",
        replace_existing=True,
        kwargs={
            "input_folder": input_dir,
            "config_path": config_path,
            "output_root": output_root
        },
        **schedule_config
    )

    log.info("Scheduler started. Press CTRL+C to stop.")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown(wait=False)
        log.info("Scheduler stopped.")
