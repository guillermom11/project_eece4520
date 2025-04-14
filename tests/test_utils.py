from HuggingFaceDatasetAdapter import HuggingFaceDatasetAdapter
from CustomDataLoader import CustomDataLoader
from HuggingFaceDataset import HuggingFaceDataset
def test_prepare_bpe_data_truncates():
    dummy_texts = ["Hello world"] * 20000
    bpe_data = CustomDataLoader.prepare_bpe_data(dummy_texts, limit=10000)
    assert len(bpe_data) == 20000  # 2 tokens per line

def test_prepare_bpe_data_removes_empty():
    dummy_texts = ["", "  ", "not empty"]
    bpe_data = CustomDataLoader.prepare_bpe_data(dummy_texts, limit=10)
    assert all(len(word) > 0 for word in bpe_data)
    assert any("n" in token for token in bpe_data)

def test_adapter_returns_split_data(monkeypatch):
    mock_data = ["This is text.", "More data."]
    
    
    def mock_loader(self, name, split="train"):
        return mock_data

    monkeypatch.setattr(HuggingFaceDataset, "load_hf_dataset", mock_loader)
    adapter = HuggingFaceDatasetAdapter()
    train, valid, test = adapter.load_and_prepare_data()

    assert train == mock_data
    assert valid == mock_data
    assert test == mock_data
