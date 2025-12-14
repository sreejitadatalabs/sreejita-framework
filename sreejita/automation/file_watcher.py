import time
from pathlib import Path
from datetime import datetime
from typing import Optional

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from sreejita.automation.batch_runner import run_single_file
from sreejita.config.loader import load_config
from sreejita.utils.logger import get_logger

log = get_logger("file-watcher")

SUPPORTED_EXT = (".csv", ".xlsx")
COOLDOWN_SECONDS = 10


class NewFileHandler(FileSystemEventHandler):
    def __init__(
        self,
        watch_dir: Path,
        config: dict,
        output_root: str = "runs"
    ) -> None:
        self.watch_dir = watch_dir
        self.config = config
        self.output_root = Path(output_root)
        self._cooldown = {}

    def on_created(self, event) -> None:
        if event.is_directory:
            return

        path = Path(event.src_path)

        if path.suffix.lower() not in SUPPORTED_EXT:
            return

        now = time.time()
        last_seen = self._cooldown.get(path.name)

        if last_seen and (now - last_seen) < COOLDOWN_SECONDS:
            return

        self._cooldown[path.name] = now

        log.info("New file detected: %s", path.name)

        # Allow OS to finish writing file
        time.sleep(2)

        timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = self.output_root / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)

        try:
            run_single_file(path, self.config, run_dir)
        except Exception as e:
            failed_dir = run_dir / "failed"
            failed_dir.mkdir(parents=True, exist_ok=True)
            failed_path = failed_dir / path.name
            failed_path.write_bytes(path.read_bytes())

            log.error(
                "File failed after retries: %s | Reason: %s",
                path.name,
                str(e)
            )


def start_watcher(
    watch_dir: str,
    config_path: Optional[str] = None,
    output_root: str = "runs"
) -> None:
    watch_dir = Path(watch_dir)

    if not watch_dir.exists():
        raise FileNotFoundError(f"Watch directory not found: {watch_dir}")

    config = load_config(config_path)

    event_handler = NewFileHandler(
        watch_dir=watch_dir,
        config=config,
        output_root=output_root
    )

    observer = Observer()
    observer.schedule(event_handler, str(watch_dir), recursive=False)

    log.info("Watching folder: %s", watch_dir)
    log.info("Press CTRL+C to stop")

    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        log.info("File watcher stopped")

    observer.join()
