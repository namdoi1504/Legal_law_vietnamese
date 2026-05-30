# src/evaluation/benchmark.py
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.retriever.bm25_retriever import BM25LegalRetriever
from src.retriever.vector_retriever import VectorRetriever
from src.reranker.bge_reranker import BGEReranker

BENCHMARK_QUESTIONS = [
    # Hợp đồng lao động
    {"question": "Hợp đồng lao động không xác định thời hạn là gì?", "expected_article": "Điều 20"},
    {"question": "Nội dung hợp đồng lao động gồm những gì?", "expected_article": "Điều 21"},
    {"question": "Thời gian thử việc tối đa là bao nhiêu ngày?", "expected_article": "Điều 25"},
    {"question": "Tiền lương thử việc được quy định như thế nào?", "expected_article": "Điều 26"},
    {"question": "Các trường hợp chấm dứt hợp đồng lao động?", "expected_article": "Điều 34"},
    {"question": "Quyền đơn phương chấm dứt hợp đồng lao động của người lao động?", "expected_article": "Điều 35"},
    {"question": "Quyền đơn phương chấm dứt hợp đồng của người sử dụng lao động?", "expected_article": "Điều 36"},
    {"question": "Trường hợp nào người sử dụng lao động không được đơn phương chấm dứt?", "expected_article": "Điều 37"},
    {"question": "Trợ cấp thôi việc được tính như thế nào?", "expected_article": "Điều 46"},
    {"question": "Trợ cấp mất việc làm là bao nhiêu?", "expected_article": "Điều 47"},

    # Tiền lương
    {"question": "Tiền lương là gì theo quy định pháp luật?", "expected_article": "Điều 90"},
    {"question": "Mức lương tối thiểu được quy định như thế nào?", "expected_article": "Điều 91"},
    {"question": "Nguyên tắc trả lương cho người lao động?", "expected_article": "Điều 94"},
    {"question": "Tiền lương làm thêm giờ làm việc vào ban đêm?", "expected_article": "Điều 98"},
    {"question": "Người lao động được tạm ứng tiền lương không?", "expected_article": "Điều 101"},

    # Thời giờ làm việc
    {"question": "Thời giờ làm việc bình thường không quá bao nhiêu giờ?", "expected_article": "Điều 105"},
    {"question": "Làm thêm giờ tối đa bao nhiêu giờ trong năm?", "expected_article": "Điều 107"},
    {"question": "Nghỉ trong giờ làm việc được quy định như thế nào?", "expected_article": "Điều 109"},
    {"question": "Nghỉ lễ tết gồm những ngày nào trong năm?", "expected_article": "Điều 112"},
    {"question": "Số ngày nghỉ hằng năm của người lao động?", "expected_article": "Điều 113"},
    {"question": "Nghỉ việc riêng kết hôn được nghỉ bao nhiêu ngày?", "expected_article": "Điều 115"},

    # Kỷ luật lao động
    {"question": "Nguyên tắc xử lý kỷ luật lao động?", "expected_article": "Điều 122"},
    {"question": "Thời hiệu xử lý kỷ luật lao động là bao lâu?", "expected_article": "Điều 123"},
    {"question": "Hình thức xử lý kỷ luật lao động gồm những gì?", "expected_article": "Điều 124"},
    {"question": "Áp dụng hình thức xử lý kỷ luật sa thải khi nào?", "expected_article": "Điều 125"},

    # An toàn lao động
    {"question": "Bảo đảm an toàn vệ sinh lao động tại nơi làm việc?", "expected_article": "Điều 134"},

    # Lao động nữ
    {"question": "Bảo vệ thai sản cho lao động nữ mang thai?", "expected_article": "Điều 137"},
    {"question": "Lao động nữ mang thai được nghỉ thai sản bao nhiêu tháng?", "expected_article": "Điều 139"},

    # Lao động đặc thù
    {"question": "Lao động chưa thành niên là người chưa đủ bao nhiêu tuổi?", "expected_article": "Điều 143"},
    {"question": "Tuổi nghỉ hưu của người lao động?", "expected_article": "Điều 169"},
]


def run_bm25_benchmark():
    print("=" * 60)
    print("BM25 BENCHMARK")
    print("=" * 60)

    bm25 = BM25LegalRetriever()
    bm25.load_index("data/processed/bm25_index.pkl")

    hit = 0
    for item in BENCHMARK_QUESTIONS:
        results = bm25.search(item["question"], top_k=8)
        retrieved = [r["metadata"].get("article", "") for r in results]
        ok = item["expected_article"] in retrieved
        if ok:
            hit += 1
        status = "[OK]" if ok else "[X] "
        print(f"{status} {item['question'][:55]}")
        if not ok:
            print(f"    Expected : {item['expected_article']}")
            print(f"    Got      : {retrieved[:3]}")

    rate = hit / len(BENCHMARK_QUESTIONS)
    print(f"\nBM25 Hit Rate@8: {hit}/{len(BENCHMARK_QUESTIONS)} = {rate:.1%}")
    return rate


def run_vector_benchmark():
    print("\n" + "=" * 60)
    print("VECTOR BENCHMARK")
    print("=" * 60)

    retriever = VectorRetriever()

    hit = 0
    for item in BENCHMARK_QUESTIONS:
        results = retriever.retrieve(item["question"], top_k=8)
        retrieved = [r["metadata"].get("article", "") for r in results]
        ok = item["expected_article"] in retrieved
        if ok:
            hit += 1
        status = "[OK]" if ok else "[X] "
        print(f"{status} {item['question'][:55]}")
        if not ok:
            print(f"    Expected : {item['expected_article']}")
            print(f"    Got      : {retrieved[:3]}")

    rate = hit / len(BENCHMARK_QUESTIONS)
    print(f"\nVector Hit Rate@8: {hit}/{len(BENCHMARK_QUESTIONS)} = {rate:.1%}")
    return rate


def run_hybrid_benchmark():
    print("\n" + "=" * 60)
    print("HYBRID (BM25 + Vector + Rerank) BENCHMARK")
    print("=" * 60)

    from src.retriever.hybrid_retriever import HybridRetriever
    retriever = HybridRetriever()
    reranker = BGEReranker()

    hit = 0
    fail_cases = []
    for item in BENCHMARK_QUESTIONS:
        candidates = retriever.retrieve(item["question"], top_k=20)
        reranked = reranker.rerank(item["question"], candidates, top_k=8)
        retrieved = [r["metadata"].get("article", "") for r in reranked]
        ok = item["expected_article"] in retrieved
        if ok:
            hit += 1
        else:
            fail_cases.append(item)
        status = "[OK]" if ok else "[X] "
        # Show top rerank score for debugging
        top_score = reranked[0].get("rerank_score", "N/A") if reranked else "N/A"
        print(f"{status} {item['question'][:55]} [score={top_score:.3f}]" if isinstance(top_score, float) else f"{status} {item['question'][:55]}")
        if not ok:
            print(f"    Expected : {item['expected_article']}")
            print(f"    Got      : {retrieved[:3]}")

    rate = hit / len(BENCHMARK_QUESTIONS)
    print(f"\nHybrid Hit Rate@8: {hit}/{len(BENCHMARK_QUESTIONS)} = {rate:.1%}")

    if fail_cases:
        print(f"\n--- {len(fail_cases)} cau hoi that bai ---")
        for f in fail_cases:
            print(f"  - {f['question']} → {f['expected_article']}")

    return rate


if __name__ == "__main__":
    bm25_rate = run_bm25_benchmark()
    vector_rate = run_vector_benchmark()
    hybrid_rate = run_hybrid_benchmark()

    print("\n" + "=" * 60)
    print("SUMMARY (Hit Rate@8)")
    print("=" * 60)
    print(f"BM25   : {bm25_rate:.1%}")
    print(f"Vector : {vector_rate:.1%}")
    print(f"Hybrid : {hybrid_rate:.1%}")