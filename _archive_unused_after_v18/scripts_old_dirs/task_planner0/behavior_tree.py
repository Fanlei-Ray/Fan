from __future__ import annotations

"""Small behavior-tree executor used by V12.

This module is intentionally lightweight and dependency-free.  It provides
Sequence/Fallback/Action/Condition nodes plus a tiny trace system so the demo can
show that the task is executed by a behavior tree, not only by a printed tree.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, List, Optional
import time


class BTStatus(Enum):
    SUCCESS = auto()
    FAILURE = auto()
    RUNNING = auto()


@dataclass
class BTTraceRow:
    timestamp: str
    task_id: str
    node_name: str
    node_type: str
    status: str
    message: str = ""


@dataclass
class BTContext:
    task: object
    planner: object
    obs: Optional[dict] = None
    selected_arm: str = ""
    safety_snapshot: object = None
    waypoints: list = field(default_factory=list)
    path_safe: bool = False
    result: object = None
    trace: List[BTTraceRow] = field(default_factory=list)

    def add_trace(self, node_name: str, node_type: str, status: BTStatus, message: str = "") -> None:
        self.trace.append(
            BTTraceRow(
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                task_id=getattr(self.task, "task_id", ""),
                node_name=node_name,
                node_type=node_type,
                status=status.name,
                message=message,
            )
        )


class BTNode:
    def __init__(self, name: str):
        self.name = name

    @property
    def node_type(self) -> str:
        return self.__class__.__name__

    def tick(self, ctx: BTContext) -> BTStatus:
        raise NotImplementedError


class SequenceNode(BTNode):
    def __init__(self, name: str, children: List[BTNode]):
        super().__init__(name)
        self.children = children

    def tick(self, ctx: BTContext) -> BTStatus:
        print(f"[BT] Sequence start: {self.name}")
        for child in self.children:
            status = child.tick(ctx)
            if status != BTStatus.SUCCESS:
                ctx.add_trace(self.name, self.node_type, status, f"child {child.name} returned {status.name}")
                print(f"[BT] Sequence {self.name}: {status.name} at child={child.name}")
                return status
        ctx.add_trace(self.name, self.node_type, BTStatus.SUCCESS, "all children succeeded")
        print(f"[BT] Sequence success: {self.name}")
        return BTStatus.SUCCESS


class FallbackNode(BTNode):
    def __init__(self, name: str, children: List[BTNode]):
        super().__init__(name)
        self.children = children

    def tick(self, ctx: BTContext) -> BTStatus:
        print(f"[BT] Fallback start: {self.name}")
        last_status = BTStatus.FAILURE
        for child in self.children:
            status = child.tick(ctx)
            if status == BTStatus.SUCCESS:
                ctx.add_trace(self.name, self.node_type, BTStatus.SUCCESS, f"child {child.name} succeeded")
                print(f"[BT] Fallback success: {self.name}, child={child.name}")
                return BTStatus.SUCCESS
            last_status = status
        ctx.add_trace(self.name, self.node_type, last_status, "all children failed")
        print(f"[BT] Fallback failed: {self.name}")
        return last_status


class ActionNode(BTNode):
    def __init__(self, name: str, fn: Callable[[BTContext], bool], message_fn: Optional[Callable[[BTContext], str]] = None):
        super().__init__(name)
        self.fn = fn
        self.message_fn = message_fn

    def tick(self, ctx: BTContext) -> BTStatus:
        print(f"[BT] Action: {self.name}")
        try:
            ok = bool(self.fn(ctx))
            status = BTStatus.SUCCESS if ok else BTStatus.FAILURE
            message = self.message_fn(ctx) if self.message_fn is not None else ""
        except Exception as exc:
            status = BTStatus.FAILURE
            message = f"exception: {exc}"
            print(f"[BT] Action exception in {self.name}: {exc}")
        ctx.add_trace(self.name, self.node_type, status, message)
        print(f"[BT] Action result: {self.name} -> {status.name} {message}")
        return status


class ConditionNode(BTNode):
    def __init__(self, name: str, predicate: Callable[[BTContext], bool], message_fn: Optional[Callable[[BTContext], str]] = None):
        super().__init__(name)
        self.predicate = predicate
        self.message_fn = message_fn

    def tick(self, ctx: BTContext) -> BTStatus:
        print(f"[BT] Condition: {self.name}")
        try:
            ok = bool(self.predicate(ctx))
            status = BTStatus.SUCCESS if ok else BTStatus.FAILURE
            message = self.message_fn(ctx) if self.message_fn is not None else ""
        except Exception as exc:
            status = BTStatus.FAILURE
            message = f"exception: {exc}"
            print(f"[BT] Condition exception in {self.name}: {exc}")
        ctx.add_trace(self.name, self.node_type, status, message)
        print(f"[BT] Condition result: {self.name} -> {status.name} {message}")
        return status
