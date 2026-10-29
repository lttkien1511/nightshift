import pytest

from nightshift.storage import DraftTooLarge, QueueFull, StorageError, Store, is_slug, slugify


@pytest.mark.parametrize(
    ("name", "slug"),
    [
        ("Gỡ Rancher khỏi k3s", "go-rancher-khoi-k3s"),
        ("Đường dẫn ĐẸP.md", "duong-dan-dep"),
        ("  hello   world  ", "hello-world"),
        ("a/b\\c..d", "a-b-c-d"),
        ("../../etc/passwd", "etc-passwd"),
        ("x" * 200, "x" * 80),
    ],
)
def test_slugify(name, slug):
    assert slugify(name) == slug
    assert is_slug(slug)


@pytest.mark.parametrize("name", ["", "   ", "---", ".md", "???"])
def test_slugify_rejects_names_without_letters_or_digits(name):
    with pytest.raises(StorageError):
        slugify(name)


@pytest.mark.parametrize("value", ["", "-a", "a-", "a--b", "A", "a b", "a/b", "..", "a" * 81])
def test_is_slug_rejects_unsafe_values(value):
    assert not is_slug(value)


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path, max_bytes=100, max_queue=2)
    s.ensure()
    return s


def test_save_and_read_round_trip(store):
    assert store.save("Bài viết đầu", "Xin chào") == "bai-viet-dau"
    assert store.read("bai-viet-dau") == "Xin chào"
    assert store.read("missing") is None
    assert store.read("../etc/passwd") is None


def test_saving_the_same_name_replaces_the_draft_and_does_not_use_queue_space(store):
    store.save("a", "one")
    store.save("b", "two")
    store.save("a", "three")
    assert store.read("a") == "three"
    assert [d.slug for d in store.drafts()] == ["a", "b"]


def test_queue_limit(store):
    store.save("a", "1")
    store.save("b", "2")
    with pytest.raises(QueueFull):
        store.save("c", "3")


def test_size_limit_counts_bytes_not_characters(store):
    store.save("ok", "a" * 100)
    with pytest.raises(DraftTooLarge):
        store.save("big", "ế" * 40)


def test_empty_draft_is_refused(store):
    with pytest.raises(StorageError):
        store.save("a", " \n ")


def test_delete_removes_the_draft_and_its_results(store):
    store.save("a", "one")
    (store.outbox / "a").mkdir()
    (store.outbox / "a" / "model.json").write_text("{}")
    assert store.drafts()[0].has_results
    assert store.delete("a")
    assert store.drafts() == []
    assert not (store.outbox / "a").exists()
    assert not store.delete("a")
    assert not store.delete("../a")


def test_no_temporary_files_are_left_behind(store):
    store.save("a", "one")
    assert sorted(p.name for p in store.inbox.iterdir()) == ["a.md"]
