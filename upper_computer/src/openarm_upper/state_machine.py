from __future__ import annotations


class InvalidTransition(RuntimeError):
    pass


class TaskStateMachine:
    TRANSITIONS = {
        "IDLE": {"TARGET_READY", "PLANNING", "ESTOP"},
        "TARGET_READY": {"IDLE", "CONFIRMED", "ESTOP"},
        "CONFIRMED": {"PLANNING", "CANCELLED", "FAILED", "ESTOP"},
        "PLANNING": {"EXECUTING", "CANCELLED", "FAILED", "ESTOP"},
        "EXECUTING": {"SUCCEEDED", "CANCELLED", "FAILED", "ESTOP"},
        "SUCCEEDED": {"IDLE", "TARGET_READY", "ESTOP"},
        "FAILED": {"IDLE", "TARGET_READY", "ESTOP"},
        "CANCELLED": {"IDLE", "TARGET_READY", "ESTOP"},
        "ESTOP": {"IDLE"},
    }

    def __init__(self) -> None:
        self.state = "IDLE"
        self.estop_latched = False

    def transition(self, target: str) -> None:
        target = str(target).upper()
        if self.estop_latched and target != "ESTOP":
            raise InvalidTransition("急停已锁存，必须先人工复位")
        if target == self.state:
            return
        if target not in self.TRANSITIONS[self.state]:
            raise InvalidTransition(f"非法状态转换：{self.state} -> {target}")
        self.state = target

    def emergency_stop(self) -> None:
        self.estop_latched = True
        self.state = "ESTOP"

    def reset_estop(self) -> None:
        if not self.estop_latched:
            return
        self.estop_latched = False
        self.state = "IDLE"
