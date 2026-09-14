from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def structure_aware(
    converted: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    heading_stack: list[dict[str, Any]] = []
    content: list[dict[str, Any]] = []

    for index, raw_block in enumerate(converted):
        if not isinstance(raw_block, Mapping):
            raise TypeError(f"converted block {index} must be a mapping")

        block = dict(raw_block)
        heading = _heading_info(block)
        if heading is not None:
            had_content = bool(content)
            if had_content:
                sections.append(_make_section(heading_stack, content))
                content = []

            level, title = heading
            _update_heading_stack(
                heading_stack,
                level,
                title,
                block,
                merge_same_level=not had_content,
            )
            continue

        content.append(block)

    if content or heading_stack:
        sections.append(_make_section(heading_stack, content))

    return sections


def _heading_info(block: Mapping[str, Any]) -> tuple[int, str] | None:
    title = block.get("text")
    if not isinstance(title, str) or not title.strip():
        return None

    level = block.get("text_level")
    if not isinstance(level, int) or isinstance(level, bool) or level <= 0:
        return None

    try:
        normalized_level = int(level)
    except (TypeError, ValueError):
        return None

    return normalized_level, title.strip()


def _update_heading_stack(
    stack: list[dict[str, Any]],
    level: int,
    title: str,
    block: Mapping[str, Any],
    *,
    merge_same_level: bool,
) -> None:
    if merge_same_level and stack and stack[-1]["level"] == level:
        stack[-1]["title"] = f"{stack[-1]['title']}\n{title}"
        stack[-1]["blocks"].append(dict(block))
        return

    while stack and stack[-1]["level"] >= level:
        stack.pop()
    stack.append(
        {
            "level": level,
            "title": title,
            "blocks": [dict(block)],
        }
    )


def _make_section(
    heading_stack: list[dict[str, Any]],
    content: list[dict[str, Any]],
) -> dict[str, Any]:
    headings = [
        dict(heading_block)
        for entry in heading_stack
        for heading_block in entry["blocks"]
    ]
    section: dict[str, Any] = {
        "type": "section",
        "heading_path": [entry["title"] for entry in heading_stack],
        "headings": headings,
        "content": [dict(block) for block in content],
    }
    if heading_stack:
        section["title"] = heading_stack[-1]["title"]
        section["text_level"] = heading_stack[-1]["level"]
    else:
        section["title"] = None
        section["text_level"] = None
    return section
