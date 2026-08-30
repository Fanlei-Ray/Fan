from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT / "src"))

from openarm_upper.protocols.usart10 import (
    FRAME_SIZE,
    Usart10Frame,
    Usart10StreamParser,
    decode_command,
    decode_telemetry,
    encode_command,
    encode_telemetry,
)


class Usart10CodecTests(unittest.TestCase):
    def setUp(self):
        self.frame = Usart10Frame(
            state=-2,
            p_int=0x1234,
            v_int=0x5678,
            t_int=0x9ABC,
            position=1.0,
            velocity=-2.0,
            torque=0.5,
        )

    def test_command_golden_bytes(self):
        raw = encode_command(self.frame)
        self.assertEqual(len(raw), FRAME_SIZE)
        self.assertEqual(
            raw.hex(),
            "5647feff34127856bc9a0000803f000000c00000003f4e45",
        )
        self.assertEqual(decode_command(raw), self.frame)

    def test_telemetry_header_and_tail(self):
        raw = encode_telemetry(self.frame)
        self.assertEqual(raw[:2], b"GV")
        self.assertEqual(raw[-2:], b"EN")
        self.assertEqual(decode_telemetry(raw), self.frame)

    def test_rejects_non_finite_float(self):
        with self.assertRaises(ValueError):
            Usart10Frame(0, 0, 0, 0, math.nan, 0.0, 0.0)

    def test_rejects_wrong_tail(self):
        raw = bytearray(encode_telemetry(self.frame))
        raw[-1] = 0
        with self.assertRaises(ValueError):
            decode_telemetry(bytes(raw))


class Usart10StreamParserTests(unittest.TestCase):
    def test_fragmentation_noise_and_two_frames(self):
        first = Usart10Frame(1, 2, 3, 4, 5.0, 6.0, 7.0)
        second = Usart10Frame(8, 9, 10, 11, 12.0, 13.0, 14.0)
        raw1 = encode_telemetry(first)
        raw2 = encode_telemetry(second)
        parser = Usart10StreamParser()
        self.assertEqual(parser.feed(b"noiseG"), [])
        frames = parser.feed(raw1[1:10])
        self.assertEqual(frames, [])
        frames = parser.feed(raw1[10:] + raw2)
        self.assertEqual(frames, [first, second])
        self.assertEqual(parser.discarded_bytes, 5)

    def test_invalid_candidate_resynchronizes(self):
        frame = Usart10Frame(1, 2, 3, 4, 5.0, 6.0, 7.0)
        bad = bytearray(encode_telemetry(frame))
        bad[-2:] = b"XX"
        parser = Usart10StreamParser()
        parsed = parser.feed(bytes(bad) + encode_telemetry(frame))
        self.assertEqual(parsed, [frame])
        self.assertGreaterEqual(parser.invalid_frames, 1)


if __name__ == "__main__":
    unittest.main()
