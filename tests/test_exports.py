import numpy as np

from deepreefmap.io.exports import save_geometry_cloud, save_ortho_grid, save_semantic_cloud
from deepreefmap.pipeline.artifacts import SemanticPointCloud
from deepreefmap.pointcloud.grid_ortho import OrthoGrid


def _read_ply(path):
    with open(path, "rb") as fh:
        header_bytes = b""
        while not header_bytes.endswith(b"end_header\n"):
            chunk = fh.readline()
            if not chunk:
                raise AssertionError("Truncated PLY header")
            header_bytes += chunk
        header_text = header_bytes.decode("ascii")
        body = fh.read()

    ply_to_np = {"float": np.float32, "double": np.float64, "uchar": np.uint8, "int": np.int32, "uint": np.uint32}
    fields = []
    n = None
    for line in header_text.splitlines():
        if line.startswith("element vertex"):
            n = int(line.split()[-1])
        elif line.startswith("property"):
            _, ply_type, name = line.split()
            fields.append((name, ply_to_np[ply_type]))
    assert n is not None
    dtype = np.dtype([(name, np.dtype(np_t).newbyteorder("<")) for name, np_t in fields])
    record = np.frombuffer(body, dtype=dtype, count=n)
    return {name: record[name] for name, _ in fields}


def test_save_semantic_cloud_round_trip(tmp_path):
    path = tmp_path / "cloud.ply"
    cloud = SemanticPointCloud(
        xyz=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        rgb=np.array([[10, 20, 30], [40, 50, 60]], dtype=np.uint8),
        labels=np.array([1, 2], dtype=np.int32),
        confidence=np.array([0.5, 0.9], dtype=np.float32),
        frame_indices=np.array([7, 8], dtype=np.int32),
    )

    save_semantic_cloud(path, cloud)
    data = _read_ply(path)

    assert data["x"].tolist() == [1.0, 4.0]
    assert data["red"].tolist() == [10, 40]
    assert data["label"].tolist() == [1, 2]
    assert data["confidence"].dtype == np.float32
    assert data["frame_index"].tolist() == [7, 8]


def test_save_semantic_cloud_omits_optional_fields(tmp_path):
    path = tmp_path / "cloud.ply"
    cloud = SemanticPointCloud(
        xyz=np.zeros((1, 3), dtype=np.float32),
        rgb=np.zeros((1, 3), dtype=np.uint8),
        labels=np.array([5], dtype=np.int32),
    )
    save_semantic_cloud(path, cloud)
    data = _read_ply(path)
    assert set(data.keys()) == {"x", "y", "z", "red", "green", "blue", "label"}


def test_save_geometry_cloud_round_trip(tmp_path):
    path = tmp_path / "geom.ply"
    save_geometry_cloud(
        path,
        xyz=np.array([[0.0, 1.0, 2.0]], dtype=np.float32),
        rgb=np.array([[255, 128, 0]], dtype=np.uint8),
    )
    data = _read_ply(path)
    assert data["y"].tolist() == [1.0]
    assert data["green"].tolist() == [128]
    assert "label" not in data


def test_save_ortho_grid_contains_expected_keys(tmp_path):
    path = tmp_path / "ortho.npz"
    grid = OrthoGrid(
        rgb=np.zeros((1, 1, 3), dtype=np.uint8),
        labels=np.array([[1]], dtype=np.int32),
        height=np.array([[0.5]], dtype=np.float32),
        counts=np.array([[3]], dtype=np.int32),
        frame_index=np.array([[4]], dtype=np.int32),
        cell_size=0.1,
    )

    save_ortho_grid(path, grid)

    data = np.load(path)
    assert {"rgb", "labels", "height", "counts", "frame_index", "cell_size"} <= set(data.files)
