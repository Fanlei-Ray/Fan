from __future__ import annotations

"""Read-only serial-port inventory for selecting the USART10 adapter."""

import argparse
import json
from pathlib import Path
import sys


APP_ROOT = Path(__file__).resolve().parents[1]
SRC = APP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def scan_ports() -> list[dict]:
    try:
        from serial.tools import list_ports
    except ImportError as exc:
        raise RuntimeError(
            "pyserial 未安装；请使用 upper_computer/.venv/Scripts/python.exe"
        ) from exc

    result = []
    for port in sorted(list_ports.comports(), key=lambda item: item.device):
        hwid = str(port.hwid or "")
        description = str(port.description or "")
        bluetooth = "BTHENUM" in hwid.upper() or "BLUETOOTH" in description.upper()
        usb = port.vid is not None or "USB" in hwid.upper() or "USB" in description.upper()
        result.append({
            "port": port.device,
            "description": description,
            "hwid": hwid,
            "vid": f"0x{port.vid:04X}" if port.vid is not None else None,
            "pid": f"0x{port.pid:04X}" if port.pid is not None else None,
            "bluetooth_virtual": bluetooth,
            "usb_uart_candidate": bool(usb and not bluetooth),
        })
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="只读列出 COM 端口；不会打开端口或发送数据"
    )
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    args = parser.parse_args()
    ports = scan_ports()
    if args.json:
        print(json.dumps(ports, ensure_ascii=False, indent=2))
        return 0
    if not ports:
        print("未检测到串口。请连接 3.3V TTL USB-UART 后重新运行。")
        return 0
    print("PORT   TYPE                  VID:PID       DESCRIPTION")
    for item in ports:
        if item["bluetooth_virtual"]:
            kind = "Bluetooth (不可选)"
        elif item["usb_uart_candidate"]:
            kind = "USB-UART 候选"
        else:
            kind = "未知（需核对）"
        identity = f"{item['vid'] or '----'}:{item['pid'] or '----'}"
        print(f"{item['port']:<6} {kind:<21} {identity:<13} {item['description']}")
    candidates = [item["port"] for item in ports if item["usb_uart_candidate"]]
    print("\nUSB-UART 候选：" + (", ".join(candidates) if candidates else "无"))
    print("本工具只枚举端口；选择前仍须核对适配器型号、3.3V TTL 和 TX/RX/GND。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
