import unittest
from types import SimpleNamespace

from modules.crowd_flow import CrowdFlowMonitor


def track(track_id: int, x: float) -> SimpleNamespace:
    return SimpleNamespace(track_id=track_id, cx=x)


class CrowdFlowMonitorTests(unittest.TestCase):
    def test_counts_only_configured_crossing_direction(self) -> None:
        monitor = CrowdFlowMonitor(window_seconds=10, alert_threshold=2)

        self.assertEqual(
            monitor.update_tracks("entry", [track(1, 20)], 100, "left_to_right", 1),
            0,
        )
        self.assertEqual(
            monitor.update_tracks("entry", [track(1, 80)], 100, "left_to_right", 2),
            1,
        )
        self.assertEqual(
            monitor.update_tracks("entry", [track(1, 20)], 100, "left_to_right", 3),
            0,
        )
        self.assertEqual(monitor.snapshot(3)["entry_count"], 1)

    def test_deadband_does_not_reset_the_last_stable_side(self) -> None:
        monitor = CrowdFlowMonitor()

        monitor.update_tracks("entry", [track(1, 20)], 100, "left_to_right", 1)
        monitor.update_tracks("entry", [track(1, 50)], 100, "left_to_right", 2)

        self.assertEqual(
            monitor.update_tracks("entry", [track(1, 80)], 100, "left_to_right", 3),
            1,
        )

    def test_rolling_window_expires_old_crossings_and_clears_alert(self) -> None:
        monitor = CrowdFlowMonitor(window_seconds=10, alert_threshold=1)
        monitor.update_tracks("entry", [track(1, 20)], 100, "left_to_right", 1)
        monitor.update_tracks("entry", [track(1, 80)], 100, "left_to_right", 2)

        self.assertTrue(monitor.snapshot(2)["alert_active"])
        expired = monitor.snapshot(13)
        self.assertEqual(expired["entry_count"], 0)
        self.assertFalse(expired["alert_active"])


if __name__ == "__main__":
    unittest.main()
