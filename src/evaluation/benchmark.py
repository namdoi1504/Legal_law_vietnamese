# src/evaluation/benchmark.py
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.retriever.bm25_retriever import BM25LegalRetriever
from src.retriever.vector_retriever import VectorRetriever
from src.reranker.bge_reranker import BGEReranker

BENCHMARK_QUESTIONS = [
    # Hợp đồng lao động
    {"question": "Thời gian nghỉ thai sản là bao nhiêu tháng?", "expected_article": "Điều 139"},
    {"question": "Lương tối thiểu vùng được quy định như thế nào?", "expected_article": "Điều 91"},
    {"question": "Thời gian làm thêm giờ tối đa trong năm là bao nhiêu?", "expected_article": "Điều 107"},
    {"question": "Hợp đồng lao động không xác định thời hạn là gì?", "expected_article": "Điều 20"},
    {"question": "Thời gian thử việc tối đa là bao nhiêu ngày?", "expected_article": "Điều 25"},
    {"question": "Điều kiện ký hợp đồng lao động xác định thời hạn?", "expected_article": "Điều 20"},
    {"question": "Người lao động có thể đơn phương chấm dứt hợp đồng không?", "expected_article": "Điều 35"},
    {"question": "Trường hợp nào người sử dụng lao động được sa thải?", "expected_article": "Điều 125"},

    # Lương và phụ cấp
    {"question": "Nguyên tắc trả lương cho người lao động?", "expected_article": "Điều 94"},
    {"question": "Tiền lương làm thêm giờ được tính như thế nào?", "expected_article": "Điều 98"},
    {"question": "Người lao động có được tạm ứng lương không?", "expected_article": "Điều 101"},
    {"question": "Mức trợ cấp thôi việc được tính như thế nào?", "expected_article": "Điều 46"},
    {"question": "Trợ cấp mất việc làm là bao nhiêu?", "expected_article": "Điều 47"},

    # Thời giờ làm việc
    {"question": "Thời giờ làm việc bình thường tối đa là bao nhiêu giờ?", "expected_article": "Điều 105"},
    {"question": "Người lao động được nghỉ giữa ca bao nhiêu phút?", "expected_article": "Điều 109"},
    {"question": "Số ngày nghỉ lễ tết trong năm là bao nhiêu?", "expected_article": "Điều 112"},
    {"question": "Người lao động được nghỉ phép năm bao nhiêu ngày?", "expected_article": "Điều 113"},
    {"question": "Điều kiện để được nghỉ việc riêng có lương?", "expected_article": "Điều 115"},

    # Bảo hiểm và an toàn
    {"question": "Trách nhiệm của người sử dụng lao động về an toàn lao động?", "expected_article": "Điều 163"},
    {"question": "Tuổi nghỉ hưu của lao động nam và nữ là bao nhiêu?", "expected_article": "Điều 169"},
    {"question": "Lao động nữ có quyền lợi gì khi mang thai?", "expected_article": "Điều 137"},
    {"question": "Quy định về lao động chưa thành niên?", "expected_article": "Điều 143"},

    # Kỷ luật
    {"question": "Các hình thức kỷ luật lao động là gì?", "expected_article": "Điều 124"},
    {"question": "Thời hiệu xử lý kỷ luật lao động là bao lâu?", "expected_article": "Điều 123"},
    {"question": "Trường hợp nào không được xử lý kỷ luật lao động?", "expected_article": "Điều 122"},
]


def run_bm25_benchmark():
    print("=" * 55)
    print("BM25 BENCHMARK")
    print("=" * 55)

    bm25 = BM25LegalRetriever()
    bm25.load_index("data/processed/bm25_index.pkl")

    hit = 0
    for item in BENCHMARK_QUESTIONS:
        results = bm25.search(item["question"], top_k=5)
        retrieved = [r["metadata"].get("article", "") for r in results]
        ok = item["expected_article"] in retrieved
        if ok:
            hit += 1
        status = "✓" if ok else "✗"
        print(f"{status} {item['question'][:50]}")
        if not ok:
            print(f"    Expected : {item['expected_article']}")
            print(f"    Got      : {retrieved[:3]}")

    rate = hit / len(BENCHMARK_QUESTIONS)
    print(f"\nBM25 Hit Rate@5: {hit}/{len(BENCHMARK_QUESTIONS)} = {rate:.1%}")
    return rate


def run_vector_benchmark():
    print("\n" + "=" * 55)
    print("VECTOR BENCHMARK")
    print("=" * 55)

    retriever = VectorRetriever()

    hit = 0
    for item in BENCHMARK_QUESTIONS:
        results = retriever.retrieve(item["question"], top_k=5)
        retrieved = [r["metadata"].get("article", "") for r in results]
        ok = item["expected_article"] in retrieved
        if ok:
            hit += 1
        status = "✓" if ok else "✗"
        print(f"{status} {item['question'][:50]}")
        if not ok:
            print(f"    Expected : {item['expected_article']}")
            print(f"    Got      : {retrieved[:3]}")

    rate = hit / len(BENCHMARK_QUESTIONS)
    print(f"\nVector Hit Rate@5: {hit}/{len(BENCHMARK_QUESTIONS)} = {rate:.1%}")
    return rate


def run_hybrid_benchmark():
    print("\n" + "=" * 55)
    print("HYBRID (BM25 + Vector + Rerank) BENCHMARK")
    print("=" * 55)

    from src.retriever.hybrid_retriever import HybridRetriever
    retriever = HybridRetriever()
    reranker = BGEReranker()

    hit = 0
    for item in BENCHMARK_QUESTIONS:
        candidates = retriever.retrieve(item["question"])
        reranked = reranker.rerank(item["question"], candidates)
        retrieved = [r["metadata"].get("article", "") for r in reranked]
        ok = item["expected_article"] in retrieved
        if ok:
            hit += 1
        status = "✓" if ok else "✗"
        print(f"{status} {item['question'][:50]}")
        if not ok:
            print(f"    Expected : {item['expected_article']}")
            print(f"    Got      : {retrieved[:3]}")

    rate = hit / len(BENCHMARK_QUESTIONS)
    print(f"\nHybrid Hit Rate@5: {hit}/{len(BENCHMARK_QUESTIONS)} = {rate:.1%}")
    return rate


if __name__ == "__main__":
    bm25_rate = run_bm25_benchmark()
    vector_rate = run_vector_benchmark()
    hybrid_rate = run_hybrid_benchmark()

    print("\n" + "=" * 55)
    print("SUMMARY")
    print("=" * 55)
    print(f"BM25   : {bm25_rate:.1%}")
    print(f"Vector : {vector_rate:.1%}")
    print(f"Hybrid : {hybrid_rate:.1%}")