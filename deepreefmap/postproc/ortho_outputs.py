from __future__ import annotations

from dataclasses import dataclass

from deepreefmap.config.classes import ClassConfig
from deepreefmap.pipeline.artifacts import SemanticPointCloud
from deepreefmap.pointcloud.grid_ortho import OrthoGrid, aggregate_cloud_to_ortho_grid
from deepreefmap.pointcloud.transect_crop import (
    TransectCropGeometry,
    TransectCropSelection,
    crop_grid_around_transect,
    crop_grid_with_transect_geometry,
    crop_grid_with_transect_selection,
)
from deepreefmap.postproc.benthic_cover import compute_benthic_cover


@dataclass(frozen=True)
class OrthoOutputs:
    grid: OrthoGrid
    cover: dict[str, object]
    cropped: bool


@dataclass(frozen=True)
class TransectCropParams:
    transect_length_m: float
    crop_width_m: float


def build_ortho_outputs(
    cloud: SemanticPointCloud,
    classes_config: ClassConfig,
    *,
    bins: int = 2000,
    crop: TransectCropParams | None = None,
) -> OrthoOutputs:
    base_grid = aggregate_cloud_to_ortho_grid(cloud, bins=bins)
    return apply_ortho_crop(base_grid, classes_config, crop=crop)


def apply_ortho_crop(
    base_grid: OrthoGrid,
    classes_config: ClassConfig,
    *,
    crop: TransectCropParams | None = None,
    transect_geometry: TransectCropGeometry | None = None,
    transect_selection: TransectCropSelection | None = None,
) -> OrthoOutputs:
    grid = base_grid
    if crop is not None:
        if transect_selection is not None:
            grid = crop_grid_with_transect_selection(base_grid, transect_selection)
        elif transect_geometry is not None:
            grid = crop_grid_with_transect_geometry(
                grid=base_grid,
                geometry=transect_geometry,
                transect_length_m=crop.transect_length_m,
                crop_width_m=crop.crop_width_m,
            )
        else:
            grid = crop_grid_around_transect(
                grid=base_grid,
                transect_label=classes_config.single_id_for_role("transect_line"),
                transect_tools_label=classes_config.single_id_for_role("transect_tools"),
                transect_length_m=crop.transect_length_m,
                crop_width_m=crop.crop_width_m,
            )
    cover = compute_benthic_cover(grid.labels, classes_config=classes_config, counts=grid.counts)
    return OrthoOutputs(grid=grid, cover=cover, cropped=grid is not base_grid)
