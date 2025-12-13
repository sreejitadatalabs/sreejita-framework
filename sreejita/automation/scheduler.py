from apscheduler.schedulers.blocking import BlockingScheduler
from sreejita.automation.batch_runner import run_batch

def start_scheduler(schedule_config, input_dir, config_path):
    scheduler = BlockingScheduler()

    scheduler.add_job(
        run_batch,
        trigger="cron",
        **schedule_config,
        args=[input_dir, config_path]
    )

    scheduler.start()
