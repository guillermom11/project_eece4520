from HuggingFaceDatasetAdapter import HuggingFaceDatasetAdapter
from CustomDataLoader import CustomDataLoader

def test_prepare_bpe_data_truncates():
    dummy_texts = ["Hello world"] * 20000
    bpe_data = CustomDataLoader.prepare_bpe_data(dummy_texts, limit=10000)
    
    assert len(bpe_data) <= 20000

def test_dataset_adapter_load(mock_dataset):
    adapter = HuggingFaceDatasetAdapter()
    train, valid, test = adapter.load_and_prepare_data()
    assert all(isinstance(t, str) for t in train)
    assert all(isinstance(t, str) for t in valid)
    assert all(isinstance(t, str) for t in test)
