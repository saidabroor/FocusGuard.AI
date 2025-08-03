import time
import csv
from datetime import datetime
import os

class FocusEventTracker:
    def __init__(self, threshold_seconds=2, csv_path="logs/focus_log.csv"):
        self.threshold = threshold_seconds
        self.csv_path = csv_path
        self.current_status = "focused"
        self.status_start_time = None
        self.is_tracking = False

        # Write CSV header if file doesn't exist
        try:
            with open(self.csv_path, "x", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Start Time", "End Time", "Status"])
        except FileExistsError:
            pass  # File already exists

    def update_status(self, new_status):
        now = time.time()

        if new_status == "focused":
            # If previously tracking unfocused state and threshold is passed, log it
            if self.is_tracking:
                duration = now - self.status_start_time
                if duration >= self.threshold:
                    self._log_event(self.status_start_time, now, self.current_status)
                self.is_tracking = False
                self.status_start_time = None
            self.current_status = "focused"
            return

        # If new status is unfocused and different from current
        if new_status != self.current_status:
            self.current_status = new_status
            self.status_start_time = now
            self.is_tracking = True

        elif self.is_tracking:
            duration = now - self.status_start_time
            if duration >= self.threshold:
                self._log_event(self.status_start_time, now, new_status)
                self.is_tracking = False  # Reset after logging
                self.status_start_time = None

    def _log_event(self, start_time, end_time, status):
        start_str = datetime.fromtimestamp(start_time).strftime("%H:%M:%S")
        end_str = datetime.fromtimestamp(end_time).strftime("%H:%M:%S")
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([start_str, end_str, status])
        print(f"Logged: {status} from {start_str} to {end_str}")
