"""Tile layer descriptors.

OSM's tile policy allows caching tiles the user has viewed but prohibits bulk
prefetching, so nothing in this package downloads areas; the disk cache in
tile_cache.py only ever holds tiles that were actually displayed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TileLayer:
    id: str
    url_template: str
    attribution: str
    max_zoom: int


OSM_LAYER = TileLayer(
    id="osm",
    url_template="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    attribution="© OpenStreetMap contributors",
    max_zoom=19,
)
