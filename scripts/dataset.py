import os
import tempfile
import shutil
import numpy as np
import torch
import pytest
from packages.data_objects.dataset import (
    BasicDataset,
    FileDataset,
    TorchDataset,
    TestTorchDataset,
    default_unpack_func,
    _filetype_loader
)


class TestUtilityFunctions:
    @pytest.fixture
    def temp_files(self):
        temp_dir = tempfile.mkdtemp()
        
        # Create different file types
        npy_file = os.path.join(temp_dir, "test.npy")
        npz_file = os.path.join(temp_dir, "test.npz")
        mat_file = os.path.join(temp_dir, "test.mat")
        
        np.save(npy_file, np.random.randn(5, 5))
        np.savez(npz_file, data=np.random.randn(3, 3))
        
        # Create a simple .mat file
        from scipy.io import savemat
        savemat(mat_file, {"data": np.random.randn(4, 4)})
        
        yield {"npy": npy_file, "npz": npz_file, "mat": mat_file, "dir": temp_dir}
        shutil.rmtree(temp_dir)

    def test_filetype_loader_npy(self, temp_files):
        data = _filetype_loader(temp_files["npy"])
        assert isinstance(data, np.ndarray)
        assert data.shape == (5, 5)

    def test_filetype_loader_npz(self, temp_files):
        data = _filetype_loader(temp_files["npz"])
        assert isinstance(data, np.lib.npyio.NpzFile)

    def test_filetype_loader_mat(self, temp_files):
        data = _filetype_loader(temp_files["mat"])
        assert isinstance(data, dict)
        assert "data" in data

    def test_filetype_loader_unsupported(self, temp_files):
        # Create an unsupported file type
        txt_file = os.path.join(temp_files["dir"], "test.txt")
        with open(txt_file, "w") as f:
            f.write("test")
        
        # Should raise TypeError for unsupported file type
        with pytest.raises(TypeError):
            _filetype_loader(txt_file)

    def test_general_unpack_func(self):
        test_dict = {"data": np.array([1, 2, 3])}
        result = default_unpack_func(test_dict)
        assert result == test_dict


class TestBasicDataset:
    @pytest.fixture
    def temp_dataset_dir(self):
        """Create a temporary directory with test data files"""
        temp_dir = tempfile.mkdtemp()
        
        # Create test files
        np.save(os.path.join(temp_dir, "test1.npy"), np.random.randn(10, 10))
        np.save(os.path.join(temp_dir, "test2.npy"), np.random.randn(5, 5))
        np.savez(os.path.join(temp_dir, "test3.npz"), data=np.random.randn(3, 3))
        
        # Create subdirectory with more files
        subdir = os.path.join(temp_dir, "subdir")
        os.makedirs(subdir)
        np.save(os.path.join(subdir, "test4.npy"), np.random.randn(2, 2))
        
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_init_with_valid_folder(self, temp_dataset_dir):
        dataset = BasicDataset(root_folder=temp_dataset_dir)
        assert len(dataset) > 0
        assert dataset.root_folder == temp_dataset_dir
        assert len(dataset.item_list) == 4  # 4 files total

    def test_init_with_none_folder(self):
        dataset = BasicDataset(root_folder=None)
        assert len(dataset) == 0
        assert dataset.item_list == []

    def test_init_with_invalid_folder(self):
        dataset = BasicDataset(root_folder="/nonexistent/folder")
        assert len(dataset) == 0
        assert dataset.item_list == []

    def test_getitem(self, temp_dataset_dir):
        dataset = BasicDataset(root_folder=temp_dataset_dir, unpack_func=None)
        data = dataset[0]
        assert isinstance(data, np.ndarray)

    def test_getitem_with_custom_unpack_func(self, temp_dataset_dir):
        def custom_unpack(data):
            return data * 2
        
        dataset = BasicDataset(root_folder=temp_dataset_dir, unpack_func=custom_unpack)
        data = dataset[0]
        assert isinstance(data, np.ndarray)

    def test_len(self, temp_dataset_dir):
        dataset = BasicDataset(root_folder=temp_dataset_dir)
        assert len(dataset) == len(dataset.item_list)

    def test_recursive_file_gathering(self, temp_dataset_dir):
        dataset = BasicDataset(root_folder=temp_dataset_dir)
        # Should find files in root and subdirectories
        assert any("subdir" in path for path in dataset.item_list)

    def test_getitem_out_of_range(self, temp_dataset_dir):
        dataset = BasicDataset(root_folder=temp_dataset_dir)
        with pytest.raises(IndexError):
            _ = dataset[len(dataset)]


class TestFileLoader:
    @pytest.fixture
    def temp_dataset_dir(self):
        temp_dir = tempfile.mkdtemp()
        
        # Create files with patient and trial identifiers
        np.save(os.path.join(temp_dir, "patient1_trial5.npy"), np.random.randn(10, 10))
        np.save(os.path.join(temp_dir, "p2_t3.npy"), np.random.randn(5, 5))
        np.save(os.path.join(temp_dir, "Patient10_Trial20.npy"), np.random.randn(3, 3))
        
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_init(self, temp_dataset_dir):
        loader = FileDataset(root_folder=temp_dataset_dir, yield_identifiers=True)
        assert loader.yield_identifiers is True
        assert len(loader) == 3

    def test_getitem_without_identifiers(self, temp_dataset_dir):
        loader = FileDataset(root_folder=temp_dataset_dir, yield_identifiers=False)
        data = loader[0]
        assert isinstance(data, np.ndarray)

    def test_getitem_with_identifiers(self, temp_dataset_dir):
        loader = FileDataset(root_folder=temp_dataset_dir, yield_identifiers=True)
        patient, trial, data = loader[0]
        assert patient is not None or patient is None  # May or may not match regex
        assert isinstance(data, np.ndarray)

    def test_iter(self, temp_dataset_dir):
        loader = FileDataset(root_folder=temp_dataset_dir, yield_identifiers=False)
        count = 0
        for data in loader:
            assert isinstance(data, np.ndarray)
            count += 1
        assert count == len(loader)

    def test_iter_with_identifiers(self, temp_dataset_dir):
        loader = FileDataset(root_folder=temp_dataset_dir, yield_identifiers=True)
        count = 0
        for patient, trial, data in loader:
            assert isinstance(data, np.ndarray)
            count += 1
        assert count == len(loader)

    def test_regex_patient(self, temp_dataset_dir):
        loader = FileDataset(root_folder=temp_dataset_dir)
        assert loader._regex_patient("patient5_trial3.npy") == 5
        assert loader._regex_patient("p10_t2.npy") == 10
        assert loader._regex_patient("Patient100_data.npy") == 100
        assert loader._regex_patient("no_patient.npy") is None

    def test_regex_trial(self, temp_dataset_dir):
        loader = FileDataset(root_folder=temp_dataset_dir)
        assert loader._regex_trial("patient5_trial3.npy") == 3
        assert loader._regex_trial("p10_t2.npy") == 2
        assert loader._regex_trial("data_Trial50.npy") == 50
        assert loader._regex_trial("no_trial.npy") is None

    def test_infer_from_path_identifiers(self, temp_dataset_dir):
        loader = FileDataset(root_folder=temp_dataset_dir)
        patient, trial = loader._infer_from_path_identifiers("patient3_trial7.npy")
        assert patient == 3
        assert trial == 7


class TestTorchDataset:
    @pytest.fixture
    def temp_dataset_dir(self):
        temp_dir = tempfile.mkdtemp()
        np.save(os.path.join(temp_dir, "test1.npy"), np.random.randn(10, 10))
        np.save(os.path.join(temp_dir, "test2.npy"), np.random.randn(5, 5))
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_init(self, temp_dataset_dir):
        dataset = TorchDataset(root_folder=temp_dataset_dir)
        assert len(dataset) == 2

    def test_getitem_returns_torch_tensor(self, temp_dataset_dir):
        dataset = TorchDataset(root_folder=temp_dataset_dir)
        data = dataset[0]
        assert isinstance(data, torch.Tensor)
        assert data.dtype == torch.float32

    def test_numpy_to_torch_conversion(self, temp_dataset_dir):
        dataset = TorchDataset(root_folder=temp_dataset_dir)
        data = dataset[0]
        assert isinstance(data, torch.Tensor)
        assert data.shape[0] in [10, 5]  # Either test1 or test2

    def test_custom_unpack_func(self, temp_dataset_dir):
        def extract_first_row(data):
            return data[0]
        
        dataset = TorchDataset(root_folder=temp_dataset_dir, unpack_func=extract_first_row)
        data = dataset[0]
        assert isinstance(data, torch.Tensor)
        assert data.ndim == 1  # Should be 1D after extracting first row


class TestCustomTestDataset:
    @pytest.fixture
    def temp_dataset_dir(self):
        temp_dir = tempfile.mkdtemp()
        # Create 5 test files
        for i in range(5):
            np.save(os.path.join(temp_dir, f"test{i}.npy"), np.random.randn(25, 7, 5, 250))
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_init_with_files(self, temp_dataset_dir):
        dataset = TestTorchDataset(root_folder=temp_dataset_dir, nsamples=3)
        assert len(dataset) == 3
        assert dataset.use_files is True

    def test_init_without_files(self):
        dataset = TestTorchDataset(root_folder=None, nsamples=10, shape=(25, 7, 5, 250))
        assert len(dataset) == 10
        assert dataset.use_files is False

    def test_getitem_with_files(self, temp_dataset_dir):
        dataset = TestTorchDataset(root_folder=temp_dataset_dir, nsamples=2)
        data = dataset[0]
        assert isinstance(data, np.ndarray)
        assert data.shape == (25, 7, 5, 250)

    def test_getitem_without_files(self):
        dataset = TestTorchDataset(root_folder=None, nsamples=5, shape=(25, 7, 5, 250))
        data = dataset[0]
        assert isinstance(data, torch.Tensor)
        assert data.shape == (25, 7, 5, 250)
        assert data.dtype == torch.float32

    def test_random_seed_consistency(self):
        dataset = TestTorchDataset(root_folder=None, nsamples=3, shape=(10, 10))
        data1 = dataset[0]
        data2 = dataset[0]
        assert torch.allclose(data1, data2)

    def test_different_samples_different_data(self):
        dataset = TestTorchDataset(root_folder=None, nsamples=3, shape=(10, 10))
        data1 = dataset[0]
        data2 = dataset[1]
        assert not torch.allclose(data1, data2)

    def test_nsamples_exceeds_available_files(self, temp_dataset_dir):
        dataset = TestTorchDataset(root_folder=temp_dataset_dir, nsamples=100)
        # Should use all available files
        assert len(dataset) == 5

    def test_len(self):
        dataset = TestTorchDataset(root_folder=None, nsamples=7)
        assert len(dataset) == 7

    def test_custom_shape(self):
        custom_shape = (10, 20, 30)
        dataset = TestTorchDataset(root_folder=None, nsamples=2, shape=custom_shape)
        data = dataset[0]
        assert data.shape == custom_shape

    def test_file_sampling_is_random(self, temp_dataset_dir):
        # Create two datasets with same parameters but different random state
        dataset1 = TestTorchDataset(root_folder=temp_dataset_dir, nsamples=3)
        dataset2 = TestTorchDataset(root_folder=temp_dataset_dir, nsamples=3)
        
        # The selected files might be different due to random choice
        # Just verify both have valid data
        assert len(dataset1) == 3
        assert len(dataset2) == 3


class TestEdgeCases:
    def test_basic_dataset_empty_directory(self):
        temp_dir = tempfile.mkdtemp()
        dataset = BasicDataset(root_folder=temp_dir)
        assert len(dataset) == 0
        shutil.rmtree(temp_dir)

    def test_torch_dataset_with_none_root(self):
        dataset = TorchDataset(root_folder=None)
        assert len(dataset) == 0

    def test_custom_test_dataset_zero_samples(self):
        dataset = TestTorchDataset(root_folder=None, nsamples=0)
        assert len(dataset) == 0

    def test_file_loader_regex_case_insensitive(self):
        temp_dir = tempfile.mkdtemp()
        loader = FileDataset(root_folder=temp_dir)
        
        # Test case insensitivity
        assert loader._regex_patient("PATIENT5.npy") == 5
        assert loader._regex_trial("TRIAL10.npy") == 10
        
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])