# modules/trigger/trigger.py (updated with time.sleep)
"""
Folder watcher for new .wav + .ready files using Watchdog.
Emits job_id and enqueues the pipeline.
"""

import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import portalocker
import uuid
import time

class ReadyHandler(FileSystemEventHandler):
    def __init__(self, config, enqueue_func):
        self.config = config
        self.enqueue = enqueue_func

    def on_created(self, event):
        if event.src_path.endswith('.ready'):
            wav_path = event.src_path.replace('.ready', '.wav')
            if os.path.exists(wav_path):
                with open(event.src_path, 'r+') as lock_file:
                    portalocker.lock(lock_file, portalocker.LOCK_EX)
                    job_id = str(uuid.uuid4())
                    self.enqueue(job_id, wav_path)
                    lock_file.truncate(0)
                    portalocker.unlock(lock_file)
                    os.remove(event.src_path)

def run_trigger_watcher(config, enqueue_func):
    event_handler = ReadyHandler(config, enqueue_func)
    observer = Observer()
    observer.schedule(event_handler, config['raw_audio_dir'], recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    finally:
        observer.stop()
        observer.join()