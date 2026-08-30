from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT / "src"))

from openarm_upper.messages import Detection
from openarm_upper.session_logger import SessionLogger


class SessionLoggerTests(unittest.TestCase):
    def test_serializes_nested_dataclass_and_omits_bytes(self):
        work_dir = APP_ROOT / "work"
        work_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=work_dir) as directory:
            logger = SessionLogger(Path(directory))
            detection = Detection(
                "d1", "orange_cube", 0.99, (1, 2, 3, 4),
                (0.5, 0.0, 1.0), (0, 0, 0, 1),
            )
            logger.log("nested", {"detection": detection, "image": b"12345"})
            record = json.loads(logger.path.read_text(encoding="utf-8"))
            self.assertEqual(record["payload"]["detection"]["id"], "d1")
            self.assertEqual(record["payload"]["image"], {"bytes_omitted": 5})


if __name__ == "__main__":
    unittest.main()
