"""Tile pixmap cache: memory LRU over a persistent disk cache, network last."""

from __future__ import annotations

import logging
from collections import OrderedDict
from functools import lru_cache

from PySide6.QtCore import QObject, QUrl, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest

from deepreefmap import __version__ as deepreefmap_version
from deepreefmap.gui.map.layers import TileLayer
from deepreefmap.paths import tile_cache_dir

logger = logging.getLogger(__name__)

_MEMORY_TILES = 256


class TileCache(QObject):
    """Serves tiles from memory, then disk; missing tiles are fetched once.

    Tiles reach the disk cache only by being displayed, which keeps previously
    visited areas available offline without violating the OSM tile policy.
    """

    tile_ready = Signal(int, int, int)

    def __init__(self, layer: TileLayer, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._layer = layer
        self._memory: OrderedDict[tuple[int, int, int], QPixmap] = OrderedDict()
        self._in_flight: set[tuple[int, int, int]] = set()
        self._failed: set[tuple[int, int, int]] = set()
        self._network: QNetworkAccessManager | None = None
        self.network_enabled = True

    @property
    def layer(self) -> TileLayer:
        return self._layer

    def pixmap(self, zoom: int, x: int, y: int) -> QPixmap | None:
        """Cached tile pixmap, scheduling a network fetch when absent."""
        if not 0 <= y < 2**zoom:
            return None
        x %= 2**zoom
        key = (zoom, x, y)
        cached = self._memory.get(key)
        if cached is not None:
            self._memory.move_to_end(key)
            return cached
        disk_path = tile_cache_dir() / self._layer.id / str(zoom) / str(x) / f"{y}.png"
        if disk_path.is_file():
            pixmap = QPixmap(str(disk_path))
            if not pixmap.isNull():
                self._remember(key, pixmap)
                return pixmap
        self._fetch(key)
        return None

    def _remember(self, key: tuple[int, int, int], pixmap: QPixmap) -> None:
        self._memory[key] = pixmap
        while len(self._memory) > _MEMORY_TILES:
            self._memory.popitem(last=False)

    def _fetch(self, key: tuple[int, int, int]) -> None:
        if not self.network_enabled or key in self._in_flight or key in self._failed:
            return
        if self._network is None:
            self._network = QNetworkAccessManager(self)
        zoom, x, y = key
        url = self._layer.url_template.format(z=zoom, x=x, y=y)
        request = QNetworkRequest(QUrl(url))
        request.setHeader(
            QNetworkRequest.KnownHeaders.UserAgentHeader,
            f"deepreefmap/{deepreefmap_version} (+https://github.com/EPFL-ECEO/deepreefmap)",
        )
        self._in_flight.add(key)
        reply = self._network.get(request)
        reply.finished.connect(lambda: self._on_reply(key, reply))

    def _on_reply(self, key: tuple[int, int, int], reply: QNetworkReply) -> None:
        self._in_flight.discard(key)
        reply.deleteLater()
        if reply.error() != QNetworkReply.NetworkError.NoError:
            logger.debug("Tile %s failed: %s", key, reply.errorString())
            self._failed.add(key)
            return
        data = bytes(reply.readAll().data())
        pixmap = QPixmap()
        if not pixmap.loadFromData(data):
            self._failed.add(key)
            return
        zoom, x, y = key
        disk_path = tile_cache_dir() / self._layer.id / str(zoom) / str(x) / f"{y}.png"
        try:
            disk_path.parent.mkdir(parents=True, exist_ok=True)
            disk_path.write_bytes(data)
        except OSError:
            logger.warning("Could not write tile cache at %s", disk_path, exc_info=True)
        self._remember(key, pixmap)
        self.tile_ready.emit(zoom, x, y)


@lru_cache(maxsize=None)
def shared_tile_cache() -> TileCache:
    """One cache (and network manager) shared by every map widget."""
    from deepreefmap.gui.map.layers import OSM_LAYER

    return TileCache(OSM_LAYER)
