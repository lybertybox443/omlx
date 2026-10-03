# SPDX-License-Identifier: Apache-2.0
"""Which pipeline stage edges ride a verified RDMA link: the contract entries ranks read."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import PurePosixPath
from typing import Any

from . import layout

_MAX_STAGE_LINKS = 64
# sun_path holds 104 bytes on macOS and 108 on Linux, including the terminating NUL.
_MAX_SOCKET_BYTES = 103


@dataclass(frozen=True)
class StageLink:
    """Rank `sender_rank` sends its stage output to `receiver_rank` through mailbox `link`."""

    sender_rank: int
    receiver_rank: int
    link: str
    service_socket: str
    rank_stride: int = 1

    def __post_init__(self) -> None:
        for rank in (self.sender_rank, self.receiver_rank):
            if not isinstance(rank, int) or isinstance(rank, bool) or rank < 0:
                raise ValueError("stage link ranks must be non-negative integers")
        stride = self.rank_stride
        if not isinstance(stride, int) or isinstance(stride, bool) or stride < 1:
            raise ValueError("stage link rank_stride must be a positive integer")
        if self.sender_rank != self.receiver_rank + stride:
            raise ValueError("a stage link must join a rank to the rank before it")
        layout.valid_link_name(self.link)
        socket_path = self.service_socket
        if (
            not isinstance(socket_path, str)
            or "\x00" in socket_path
            or len(socket_path.encode()) > _MAX_SOCKET_BYTES
            or not PurePosixPath(socket_path).is_absolute()
        ):
            raise ValueError(
                "stage link service socket must be an absolute Unix socket path"
            )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "sender_rank": self.sender_rank,
            "receiver_rank": self.receiver_rank,
            "link": self.link,
            "service_socket": self.service_socket,
        }
        if self.rank_stride != 1:
            payload["rank_stride"] = self.rank_stride
        return payload

    @classmethod
    def from_dict(cls, payload: Any) -> StageLink:
        if not isinstance(payload, dict):
            raise ValueError("stage link must be an object")
        return cls(
            sender_rank=payload.get("sender_rank"),
            receiver_rank=payload.get("receiver_rank"),
            link=payload.get("link"),
            service_socket=payload.get("service_socket"),
            rank_stride=payload.get("rank_stride", 1),
        )


def validate_stage_links(
    raw: Any, world_size: int | None = None
) -> tuple[StageLink, ...]:
    """Parse contract stage links; each rank may appear at most once per direction."""
    if raw is None:
        return ()
    if not isinstance(raw, (list, tuple)) or len(raw) > _MAX_STAGE_LINKS:
        raise ValueError("stage_links must be a list of at most 64 entries")
    links = tuple(
        item if isinstance(item, StageLink) else StageLink.from_dict(item)
        for item in raw
    )
    senders = [item.sender_rank for item in links]
    if len(set(senders)) != len(senders) or len({item.link for item in links}) != len(
        links
    ):
        raise ValueError("stage_links repeats a rank or a link")
    receivers = [item.receiver_rank for item in links]
    if len(set(receivers)) != len(receivers):
        raise ValueError("stage_links repeats an incoming rank")
    if world_size is not None and any(
        item.sender_rank >= world_size or item.receiver_rank >= world_size
        for item in links
    ):
        raise ValueError("stage_links names a rank outside the deployment")
    return links


def pipeline_stage_links(
    links: Any, tp_size: int, tp_rank: int, world_size: int
) -> tuple[StageLink, ...]:
    """Links of this rank's TP column, translated to pipeline-stage ranks."""
    for value in (tp_size, tp_rank, world_size):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError("tp_size, tp_rank and world_size must be integers")
    if tp_size < 1 or world_size < 1 or not 0 <= tp_rank < tp_size:
        raise ValueError("invalid tensor-parallel topology")
    if world_size % tp_size:
        raise ValueError("world_size must be a multiple of tp_size")
    known = validate_stage_links(links, world_size)
    if tp_size == 1:
        return known
    if known and world_size == tp_size:
        raise ValueError("pure tensor-parallel deployments have no stage links")
    out = []
    for item in known:
        if (
            item.rank_stride != tp_size
            or item.sender_rank // tp_size != item.receiver_rank // tp_size + 1
            or item.sender_rank % tp_size != item.receiver_rank % tp_size
        ):
            raise ValueError("stage link must join the same tensor-parallel column")
        if item.sender_rank % tp_size == tp_rank:
            out.append(
                replace(
                    item,
                    sender_rank=item.sender_rank // tp_size,
                    receiver_rank=item.receiver_rank // tp_size,
                    rank_stride=1,
                )
            )
    return tuple(out)


def links_for_rank(
    links: Iterable[StageLink], rank: int
) -> tuple[StageLink | None, StageLink | None]:
    """This rank's (incoming, outgoing) stage links."""
    known = tuple(links)
    incoming = next((item for item in known if item.receiver_rank == rank), None)
    outgoing = next((item for item in known if item.sender_rank == rank), None)
    return incoming, outgoing
