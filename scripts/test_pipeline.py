# scripts/test_pipeline.py
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.rag_pipeline import LegalRAGPipeline

pipeline = LegalRAGPipeline()

questions = [
    "Nghỉ thai sản bao nhiêu tháng?",
    "Làm thêm giờ tối đa bao nhiêu giờ một năm?",
    "Tuổi nghỉ hưu của lao động nam là bao nhiêu?",
]

for q in questions:
    print(f"\n{'='*55}")
    print(f"HỎI: {q}")
    print("="*55)
    result = pipeline.run(q)
    print(f"TRẢ LỜI:\n{result.answer}")
    print(f"\nCĂN CỨ: {result.citations[0] if result.citations else 'N/A'}")