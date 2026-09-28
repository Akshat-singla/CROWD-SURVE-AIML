import os
import io
import tempfile
import unittest
from unittest.mock import patch

import dashboard.app as app_module


class CrowdRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app_module.app.test_client()

    def test_crowd_rejects_duplicate_cameras(self) -> None:
        response = self.client.post(
            "/start",
            json={"source": "crowd", "entry_index": 1, "exit_index": 1},
        )

        self.assertEqual(response.status_code, 400)

    def test_people_mismatch_alert_waits_for_both_feeds(self) -> None:
        with (
            patch.dict(
                app_module._crowd_camera_status,
                {"entry": "online", "exit": "offline"},
            ),
            patch.dict(app_module._crowd_people_counts, {"entry": 3, "exit": 0}),
            patch.object(app_module, "_crowd_monitor", None),
            patch.object(app_module.config, "CROWD_DEFAULT_MISMATCH_THRESHOLD", 2),
        ):
            response = self.client.get("/stats")
            data = response.get_json()
            self.assertFalse(data["crowd_feeds_ready"])
            self.assertFalse(data["crowd_people_alert_active"])

            app_module._crowd_camera_status["exit"] = "online"
            response = self.client.get("/stats")
            data = response.get_json()

        self.assertTrue(data["crowd_feeds_ready"])
        self.assertTrue(data["crowd_people_alert_active"])

    def test_crowd_starts_with_two_uploaded_videos(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = os.path.join(temp_dir, "data", "input")
            os.makedirs(input_dir)
            entry_video = os.path.join(input_dir, "entry.mp4")
            exit_video = os.path.join(input_dir, "exit.mp4")
            for path in (entry_video, exit_video):
                with open(path, "wb") as video_file:
                    video_file.write(b"test video")

            capture = unittest.mock.MagicMock()
            capture.isOpened.return_value = True
            with patch.object(app_module.config, "BASE_DIR", temp_dir), \
                    patch.object(app_module.cv2, "VideoCapture", return_value=capture), \
                    patch.object(app_module, "_launch_crowd_pipeline") as launch:
                response = self.client.post(
                    "/start",
                    json={
                        "source": "crowd",
                        "crowd_source_type": "videos",
                        "entry_path": entry_video,
                        "exit_path": exit_video,
                        "window_seconds": 60,
                        "alert_threshold": 3,
                    },
                )

        self.assertEqual(response.status_code, 200, response.get_json())
        launch.assert_called_once_with(
            os.path.realpath(entry_video),
            os.path.realpath(exit_video),
            "videos",
            60,
            3,
            "left_to_right",
            "left_to_right",
        )

    def test_video_uploads_get_unique_names_and_reject_non_video_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(app_module.config, "BASE_DIR", temp_dir):
                first = self.client.post(
                    "/upload",
                    data={"file": (io.BytesIO(b"one"), "same.mp4")},
                    content_type="multipart/form-data",
                )
                second = self.client.post(
                    "/upload",
                    data={"file": (io.BytesIO(b"two"), "same.mp4")},
                    content_type="multipart/form-data",
                )
                invalid = self.client.post(
                    "/upload",
                    data={"file": (io.BytesIO(b"not a video"), "notes.txt")},
                    content_type="multipart/form-data",
                )
                first_path = first.get_json()["path"]
                second_path = second.get_json()["path"]
                self.assertTrue(os.path.isfile(first_path))
                self.assertTrue(os.path.isfile(second_path))

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertNotEqual(first_path, second_path)
        self.assertEqual(invalid.status_code, 400)


if __name__ == "__main__":
    unittest.main()
