"""GREEN: FAISS dot-product/cosine semantics with fixed local vectors.

RED adapter tests separately enforce that our own adapter reproduces this.
"""
import pytest
faiss = pytest.importorskip("faiss")
np = pytest.importorskip("numpy")
pytestmark = pytest.mark.integration


def test_flat_inner_product_equals_cosine_after_normalization():
    vectors = np.array([[3., 0., 0.], [1., 1., 0.], [-1., 0., 0.]], dtype="float32")
    query = np.array([[9., 0., 0.]], dtype="float32")
    faiss.normalize_L2(vectors)
    faiss.normalize_L2(query)
    db = faiss.IndexFlatIP(3)
    db.add(vectors)
    scores, positions = db.search(query, 3)
    assert positions[0].tolist() == [0, 1, 2]
    assert scores[0].tolist() == pytest.approx([1.0, 2**-.5, -1.0], abs=1e-6)


def test_faiss_returns_top_k_without_api_requests():
    vectors = np.eye(3, dtype="float32")
    db = faiss.IndexFlatIP(3)
    db.add(vectors)
    scores, positions = db.search(np.array([[0., 1., 0.]], dtype="float32"), 1)
    assert positions.tolist() == [[1]]
    assert float(scores[0, 0]) == pytest.approx(1.0, abs=1e-6)
