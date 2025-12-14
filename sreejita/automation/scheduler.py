from apscheduler.schedulers.blocking import BlockingScheduler

from sreejita.automation.batch_runner import run_batch
from sreejita.utils.logger import get_logger

log = get_logger("scheduler")


def start_scheduler(
    schedule_config: dict,
    input_dir: str,
    config_path: str,
    output_root: str = "runs"
):
    """
    Start time-based automation using APScheduler.

    schedule_config example:
    {
        "hour": 9,
        "minute": 0
    }
    """

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
        log.info("Scheduler stopped.")
