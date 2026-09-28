import os
import tempfile
import threading
import time
import unittest

import cv2
import numpy as np

import dashboard.app as app_module
from modules.alert_logger import AlertLogger


class AlertLoggerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.snapshots_dir = os.path.join(self.temp_dir.name, "snapshots")
        self.logs_dir = os.path.join(self.temp_dir.name, "logs")
        self.logger = AlertLogger(
            snapshot_dir=self.snapshots_dir,
            log_dir=self.logs_dir,
            log_queue_maxsize=10,
        )
        self.frame = np.zeros((64, 96, 3), dtype=np.uint8)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_same_second_alerts_save_distinct_gallery_images(self) -> None:
        timestamp = time.time()
        for _ in range(2):
            self.logger.submit(
                track_id=-1,
                violation="Crowd Count Mismatch",
                timestamp=timestamp,
                raw_frame=self.frame,
                all_tracks=[],
            )

        stop_event = threading.Event()
        stop_event.set()
        self.logger.run_worker(stop_event)

        events = self.logger.get_recent_events()
        self.assertEqual(len(events), 2)
        paths = [event["snapshot_path"] for event in events]
        self.assertEqual(len(set(paths)), 2)
        for path in paths:
            self.assertTrue(os.path.isfile(path), path)
            self.assertIsNotNone(cv2.imread(path), path)

    def test_worker_drains_pending_evidence_after_stop_signal(self) -> None:
        self.logger.submit(
            track_id=7,
            violation="Unsafe Activity",
            timestamp=time.time(),
            raw_frame=self.frame,
            all_tracks=[],
        )
        stop_event = threading.Event()
        stop_event.set()

        self.logger.run_worker(stop_event)

        self.assertEqual(len(self.logger.get_recent_events()), 1)
        self.assertTrue(os.path.isfile(
            self.logger.get_recent_events()[0]["snapshot_path"]
        ))

    def test_saved_snapshot_is_listed_and_served_by_dashboard(self) -> None:
        self.logger.submit(
            track_id=9,
            violation="Unsafe Activity",
            timestamp=time.time(),
            raw_frame=self.frame,
            all_tracks=[],
        )
        stop_event = threading.Event()
        stop_event.set()
        self.logger.run_worker(stop_event)

        previous_logger = app_module._alert_logger
        previous_preproc_logger = app_module._preproc_alert_logger
        try:
            app_module._alert_logger = self.logger
            app_module._preproc_alert_logger = None
            with app_module.app.test_client() as client, \
                    unittest.mock.patch.object(
                        app_module.config, "SNAPSHOT_DIR", self.snapshots_dir
                    ):
                gallery = client.get("/snapshots").get_json()
                self.assertEqual(len(gallery), 1)
                response = client.get(gallery[0]["url"])
                self.assertEqual(response.status_code, 200)
                self.assertTrue(response.data.startswith(b"\xff\xd8"))
        finally:
            app_module._alert_logger = previous_logger
            app_module._preproc_alert_logger = previous_preproc_logger


if __name__ == "__main__":
    unittest.main()
