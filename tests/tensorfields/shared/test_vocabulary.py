from json2vec.tensorfields.shared.vocabulary import OnlineVocabularyModel, shared_manager


def test_growth_assigns_sequential_ids_and_dedupes():
    model = OnlineVocabularyModel(max_vocab_size=10_000)
    vocab = model.state

    first = [vocab(f"w{i}", update=True) for i in range(2_000)]
    assert first == list(range(2_000))

    # Repeats return the same id; nothing new is appended.
    assert [vocab(f"w{i}", update=True) for i in range(2_000)] == list(range(2_000))
    assert len(model.master) == 2_000


def test_full_vocabulary_falls_back_to_unavailable_bucket():
    model = OnlineVocabularyModel(max_vocab_size=4)
    vocab = model.state

    assert [vocab(f"w{i}", update=True) for i in range(4)] == [0, 1, 2, 3]
    assert vocab("overflow", update=True) == vocab.unavailable_index


def test_all_vocabularies_share_one_manager():
    a = OnlineVocabularyModel(max_vocab_size=8)
    b = OnlineVocabularyModel(max_vocab_size=8)
    assert shared_manager() is shared_manager()

    # Independent state on a single shared manager process.
    a.state("alpha", update=True)
    assert list(a.master) == ["alpha"]
    assert list(b.master) == []
