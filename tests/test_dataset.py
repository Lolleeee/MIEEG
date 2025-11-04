import os
import numpy as np
import torch
import pytest

from packages.data_objects.dataset import (
    BasicDataset,
    FileDataset,
    TorchDataset,
    TestTorchDataset,
)


def write_npy(path, arr):
    np.save(path, arr)


def write_pt(path, obj):
    obj = torch.from_numpy(obj) 
    torch.save(obj, path)


def test_basicdataset_finds_files_recursively(tmp_path):
    # create nested structure
    d1 = tmp_path / "a"
    d2 = d1 / "b"
    d2.mkdir(parents=True)
    arr = np.arange(6)
    f1 = d2 / "x.npy"
    f2 = tmp_path / "y.npy"
    write_npy(f1, arr)
    write_npy(f2, arr + 1)

    ds = BasicDataset(str(tmp_path), unpack_func=lambda x: x)
    # order may vary, but length must match files written
    assert len(ds) == 2
    items = [ds[i] for i in range(len(ds))]
    # loaded numpy arrays
    assert any(np.array_equal(item, arr) for item in items)
    assert any(np.array_equal(item, arr + 1) for item in items)


def test_basicdataset_loads_pt_and_npy(tmp_path):
    arr = np.random.randn(4, 3).astype(np.float32)
    np_path = tmp_path / "a.npy"
    pt_path = tmp_path / "b.pt"
    write_npy(np_path, arr)
    write_pt(pt_path, arr)

    ds = BasicDataset(str(tmp_path), unpack_func=lambda x: x)
    loaded = [ds[i] for i in range(len(ds))]
    # both should load to numpy-like or tensors depending on loader; unpack_func returns raw object
    assert any(isinstance(x, np.ndarray) for x in loaded) or any(torch.is_tensor(x) for x in loaded)


def test_filedataset_yield_identifiers_parsing(tmp_path):
    # filename contains patient and trial patterns
    fname = tmp_path / "patient12_trial3.npy"
    write_npy(fname, np.ones((2, 2)))
    fd = FileDataset(str(tmp_path), unpack_func=lambda x: x, yield_identifiers=True)
    patient, trial, data = fd[0]
    assert patient == 12
    assert trial == 3
    assert isinstance(data, np.ndarray)


def test_torchdataset_converts_numpy_and_applies_normalization(tmp_path):
    arr = np.ones((1, 10), dtype=np.float32) * 2.0
    np_path = tmp_path / "d.npy"
    write_npy(np_path, arr)

    td = TorchDataset(str(tmp_path), unpack_func=lambda x: x, chunk_size=None)
    # set normalization params (mean=2, std=1)
    td._norm_params = (torch.tensor(2.0), torch.tensor(1.0))
    out = td[0]
    assert torch.is_tensor(out)
    # (2 - 2)/1 => 0
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_torchdataset_get_chunks_behavior(tmp_path):
    # data shape (1, 10), chunk_size 5 -> expect shape (2,5)
    arr = np.arange(10, dtype=np.float32).reshape(1, 10)
    np_path = tmp_path / "c.npy"
    write_npy(np_path, arr)

    td = TorchDataset(str(tmp_path), unpack_func=lambda x: x, chunk_size=5)
    out = td[0]  # returned is float tensor from .__getitem__()
    # after chunking expected shape (n_chunks, chunk_size) -> (2,5)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 5)


def test_torchdataset_chunk_too_small_raises(tmp_path):
    # data length 4, chunk_size 5 -> should raise when __getitem__ called
    arr = np.arange(4, dtype=np.float32).reshape(1, 4)
    np_path = tmp_path / "small.npy"
    write_npy(np_path, arr)

    td = TorchDataset(str(tmp_path), unpack_func=lambda x: x, chunk_size=5)
    with pytest.raises(ValueError):
        _ = td[0]


def test_testtorchdataset_with_root_folder_works(tmp_path):
    # avoid the code path that constructs random list (which is buggy)
    # create some files and pass root_folder so TestTorchDataset uses BasicDataset logic
    for i in range(3):
        write_npy(tmp_path / f"s{i}.npy", np.ones((1, 5), dtype=np.float32) * i)

    ttd = TestTorchDataset(root_folder=str(tmp_path), unpack_func=lambda x: x, nsamples=2, shape=(1, 5))
    # should sample nsamples items
    assert len(ttd.item_list) == 2
    for i in range(len(ttd)):
        _ = ttd[i]  # ensure indexing works

def test_testtorchdataset_random_list_generation():
    # deterministic generation - two instances should produce identical sampled lists
    ds1 = TestTorchDataset(root_folder=None, nsamples=5, shape=(2, 3))
    ds2 = TestTorchDataset(root_folder=None, nsamples=5, shape=(2, 3))

    assert isinstance(ds1.item_list, list)
    assert len(ds1.item_list) == 5
    assert len(ds2.item_list) == 5

    # each item should be a torch.Tensor with the requested shape
    for t in ds1.item_list:
        assert isinstance(t, torch.Tensor)
        assert t.shape == (2, 3)

    # because generation and sampling are seeded, the two datasets should match elementwise
    for a, b in zip(ds1.item_list, ds2.item_list):
        assert torch.allclose(a, b)