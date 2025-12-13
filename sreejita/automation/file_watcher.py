import time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from sreejita.automation.batch_runner import run_batch
from sreejita.automation.batch_runner import run_single_file
from sreejita.utils.logger import get_logger

log = get_logger("file-watcher")

SUPPORTED_EXT = (".csv", ".xlsx")


class NewFileHandler(FileSystemEventHandler):
    def __init__(self, watch_dir, config_path):
        self.watch_dir = watch_dir
        self.config_path = config_path
        self._cooldown = set()

    def on_created(self, event):
        if event.is_directory:
            return

        path = Path(event.src_path)

        if path.suffix.lower() not in SUPPORTED_EXT:
            return

        # Prevent duplicate triggers
        if path.name in self._cooldown:
            return

        log.info("New file detected: %s", path.name)
        self._cooldown.add(path.name)

        # Give OS time to finish writing file
        time.sleep(2)

        run_single_file(path, self.config_path)


def start_watcher(watch_dir: str, config_path: str):
    watch_dir = Path(watch_dir)

    if not watch_dir.exists():
        raise FileNotFoundError(f"Watch directory not found: {watch_dir}")

    event_handler = NewFileHandler(str(watch_dir), config_path)
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

