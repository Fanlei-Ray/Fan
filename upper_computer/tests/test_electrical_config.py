from __future__ import annotations

import json
from pathlib import Path
import unittest


APP_ROOT = Path(__file__).resolve().parents[1]


class ElectricalConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(
            (APP_ROOT / "config" / "electrical_roboarm.json").read_text(
                encoding="utf-8"
            )
        )

    def test_confirmed_serial_and_can_settings(self):
        self.assertEqual(self.config["usart10"]["baudrate"], 115200)
        self.assertEqual(self.config["usart10"]["logic_voltage_v"], 3.3)
        self.assertEqual(self.config["can_network"]["nominal_bitrate_bps"], 1_000_000)
        self.assertFalse(self.config["safety"]["write_control_allowed"])

    def test_joint_ids_buses_and_limits_are_consistent(self):
        motors = self.config["motors"]
        self.assertEqual(len(motors), 7)
        self.assertEqual(
            [int(item["can_id"], 16) for item in motors],
            list(range(1, 8)),
        )
        expected_buses = ["FDCAN3", "FDCAN3", "FDCAN1", "FDCAN1", "FDCAN2", "FDCAN2", "FDCAN2"]
        self.assertEqual([item["can_bus"] for item in motors], expected_buses)
        for item in motors:
            endpoints = [
                item["clockwise_limit_rad"],
                item["counterclockwise_limit_rad"],
            ]
            self.assertEqual(item["relative_limit_rad"], [min(endpoints), max(endpoints)])

    def test_current_com_ports_are_not_selected(self):
        probe = self.config["usart10"]["host_port_probe"]
        self.assertIsNone(probe["selected_port"])
        self.assertFalse(probe["usb_uart_detected"])
        self.assertTrue(all(not item["safe_for_robot"] for item in probe["detected_ports"]))


if __name__ == "__main__":
    unittest.main()
