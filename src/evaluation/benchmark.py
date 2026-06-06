# src/evaluation/benchmark.py
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.retriever.bm25_retriever import BM25LegalRetriever
from src.retriever.vector_retriever import VectorRetriever
from src.reranker.bge_reranker import BGEReranker

# BENCHMARK_QUESTIONS = [
#     # Hợp đồng lao động
#     {"question": "Hợp đồng lao động không xác định thời hạn là gì?", "expected_article": "Điều 20"},
#     {"question": "Nội dung hợp đồng lao động gồm những gì?", "expected_article": "Điều 21"},
#     {"question": "Thời gian thử việc tối đa là bao nhiêu ngày?", "expected_article": "Điều 25"},
#     {"question": "Tiền lương thử việc được quy định như thế nào?", "expected_article": "Điều 26"},
#     {"question": "Các trường hợp chấm dứt hợp đồng lao động?", "expected_article": "Điều 34"},
#     {"question": "Quyền đơn phương chấm dứt hợp đồng lao động của người lao động?", "expected_article": "Điều 35"},
#     {"question": "Quyền đơn phương chấm dứt hợp đồng của người sử dụng lao động?", "expected_article": "Điều 36"},
#     {"question": "Trường hợp nào người sử dụng lao động không được đơn phương chấm dứt?", "expected_article": "Điều 37"},
#     {"question": "Trợ cấp thôi việc được tính như thế nào?", "expected_article": "Điều 46"},
#     {"question": "Trợ cấp mất việc làm là bao nhiêu?", "expected_article": "Điều 47"},

#     # Tiền lương
#     {"question": "Tiền lương là gì theo quy định pháp luật?", "expected_article": "Điều 90"},
#     {"question": "Mức lương tối thiểu được quy định như thế nào?", "expected_article": "Điều 91"},
#     {"question": "Nguyên tắc trả lương cho người lao động?", "expected_article": "Điều 94"},
#     {"question": "Tiền lương làm thêm giờ làm việc vào ban đêm?", "expected_article": "Điều 98"},
#     {"question": "Người lao động được tạm ứng tiền lương không?", "expected_article": "Điều 101"},

#     # Thời giờ làm việc
#     {"question": "Thời giờ làm việc bình thường không quá bao nhiêu giờ?", "expected_article": "Điều 105"},
#     {"question": "Làm thêm giờ tối đa bao nhiêu giờ trong năm?", "expected_article": "Điều 107"},
#     {"question": "Nghỉ trong giờ làm việc được quy định như thế nào?", "expected_article": "Điều 109"},
#     {"question": "Nghỉ lễ tết gồm những ngày nào trong năm?", "expected_article": "Điều 112"},
#     {"question": "Số ngày nghỉ hằng năm của người lao động?", "expected_article": "Điều 113"},
#     {"question": "Nghỉ việc riêng kết hôn được nghỉ bao nhiêu ngày?", "expected_article": "Điều 115"},

#     # Kỷ luật lao động
#     {"question": "Nguyên tắc xử lý kỷ luật lao động?", "expected_article": "Điều 122"},
#     {"question": "Thời hiệu xử lý kỷ luật lao động là bao lâu?", "expected_article": "Điều 123"},
#     {"question": "Hình thức xử lý kỷ luật lao động gồm những gì?", "expected_article": "Điều 124"},
#     {"question": "Áp dụng hình thức xử lý kỷ luật sa thải khi nào?", "expected_article": "Điều 125"},

#     # An toàn lao động
#     {"question": "Bảo đảm an toàn vệ sinh lao động tại nơi làm việc?", "expected_article": "Điều 134"},

#     # Lao động nữ
#     {"question": "Bảo vệ thai sản cho lao động nữ mang thai?", "expected_article": "Điều 137"},
#     {"question": "Lao động nữ mang thai được nghỉ thai sản bao nhiêu tháng?", "expected_article": "Điều 139"},

#     # Lao động đặc thù
#     {"question": "Lao động chưa thành niên là người chưa đủ bao nhiêu tuổi?", "expected_article": "Điều 143"},
#     {"question": "Tuổi nghỉ hưu của người lao động?", "expected_article": "Điều 169"},
# ]


BENCHMARK_QUESTIONS = [
    {"question": "nghỉ đẻ được mấy tháng", "expected_article": "Điều 139"},
    {"question": "bị đuổi việc oan thì được đền bao nhiêu tiền", "expected_article": "Điều 41"},
    {"question": "lương thử việc có được thấp hơn lương chính không", "expected_article": "Điều 26"},
    {"question": "công ty có được giữ bằng cấp chứng chỉ của nhân viên không", "expected_article": "Điều 17"},
    {"question": "ký hợp đồng miệng có hợp lệ không", "expected_article": "Điều 14"},

    # Câu tình huống — không có từ khóa rõ ràng
    {"question": "làm 3 năm bị cho nghỉ việc thì nhận được tiền gì", "expected_article": "Điều 46"},
    {"question": "bao nhiêu ngày phải báo trước khi tự nghỉ việc", "expected_article": "Điều 35"},
    {"question": "đi làm muộn bị trừ lương có đúng luật không", "expected_article": "Điều 102"},
    {"question": "công ty bắt làm thêm giờ mà không trả thêm tiền thì sao", "expected_article": "Điều 98"},

    # Câu phủ định / ngoại lệ — dễ retrieve nhầm điều liền kề
    {"question": "đang mang thai có bị sa thải được không", "expected_article": "Điều 137"},
    {"question": "thử việc xong công ty không nhận thì có được báo trước không", "expected_article": "Điều 27"},
    {"question": "nghỉ việc không báo trước thì mất gì", "expected_article": "Điều 40"},

    # Câu thiếu từ khóa — mơ hồ
    {"question": "làm ca đêm được thêm bao nhiêu phần trăm lương", "expected_article": "Điều 98"},
    {"question": "mỗi năm được nghỉ phép mấy ngày", "expected_article": "Điều 113"},
    {"question": "công ty nhỏ dưới 10 người có cần nội quy không", "expected_article": "Điều 118"},
]


# BENCHMARK_QUESTIONS = [
 
#     # ── PHẦN 1: PHẠM VI & NGUYÊN TẮC CHUNG (Điều 1–12) ──────
#     {"question": "Bộ luật Lao động 2019 điều chỉnh những quan hệ nào?",
#      "expected_article": "Điều 1", "type": "factual"},
#     {"question": "Những đối tượng nào được áp dụng Bộ luật Lao động?",
#      "expected_article": "Điều 2", "type": "factual"},
#     {"question": "Người lao động được định nghĩa như thế nào trong luật?",
#      "expected_article": "Điều 3", "type": "factual"},
#     {"question": "Các hành vi nào bị nghiêm cấm trong lĩnh vực lao động?",
#      "expected_article": "Điều 8", "type": "factual"},
#     {"question": "Người lao động có quyền tự do lựa chọn việc làm không?",
#      "expected_article": "Điều 10", "type": "negation"},
#     {"question": "Người sử dụng lao động có trách nhiệm gì về quản lý lao động?",
#      "expected_article": "Điều 12", "type": "factual"},
 
#     # ── PHẦN 2: HỢP ĐỒNG LAO ĐỘNG (Điều 13–51) ─────────────
#     {"question": "Hợp đồng lao động là gì theo quy định pháp luật?",
#      "expected_article": "Điều 13", "type": "factual"},
#     {"question": "Hợp đồng lao động có bắt buộc phải lập bằng văn bản không?",
#      "expected_article": "Điều 14", "type": "negation"},
#     {"question": "Nguyên tắc giao kết hợp đồng lao động là gì?",
#      "expected_article": "Điều 15", "type": "factual"},
#     {"question": "Người sử dụng lao động không được làm gì khi giao kết hợp đồng?",
#      "expected_article": "Điều 17", "type": "negation"},
#     {"question": "Người lao động có thể ký nhiều hợp đồng lao động cùng lúc không?",
#      "expected_article": "Điều 19", "type": "negation"},
#     {"question": "Hợp đồng lao động xác định thời hạn và không xác định thời hạn khác nhau thế nào?",
#      "expected_article": "Điều 20", "type": "comparison"},
#     {"question": "Phụ lục hợp đồng lao động có giá trị pháp lý như thế nào?",
#      "expected_article": "Điều 22", "type": "factual"},
#     {"question": "Hợp đồng lao động có hiệu lực từ khi nào?",
#      "expected_article": "Điều 23", "type": "factual"},
#     {"question": "Thử việc có được ghi trong hợp đồng lao động không?",
#      "expected_article": "Điều 24", "type": "negation"},
#     {"question": "Khi kết thúc thử việc người sử dụng lao động cần làm gì?",
#      "expected_article": "Điều 27", "type": "factual"},
#     {"question": "Người lao động có được làm thay người khác theo hợp đồng không?",
#      "expected_article": "Điều 28", "type": "negation"},
#     {"question": "Trường hợp nào người sử dụng lao động được chuyển người lao động làm việc khác?",
#      "expected_article": "Điều 29", "type": "condition"},
#     {"question": "Tạm hoãn hợp đồng lao động xảy ra trong những trường hợp nào?",
#      "expected_article": "Điều 30", "type": "factual"},
#     {"question": "Sau khi hết thời hạn tạm hoãn người lao động được nhận lại trong bao nhiêu ngày?",
#      "expected_article": "Điều 31", "type": "numeric"},
#     {"question": "Người lao động làm việc không trọn thời gian có quyền gì?",
#      "expected_article": "Điều 32", "type": "factual"},
#     {"question": "Sửa đổi hợp đồng lao động phải thực hiện như thế nào?",
#      "expected_article": "Điều 33", "type": "factual"},
#     {"question": "Nghĩa vụ của người lao động khi đơn phương chấm dứt hợp đồng trái pháp luật?",
#      "expected_article": "Điều 40", "type": "factual"},
#     {"question": "Người sử dụng lao động phải làm gì khi đơn phương chấm dứt hợp đồng trái luật?",
#      "expected_article": "Điều 41", "type": "factual"},
#     {"question": "Nghĩa vụ của người sử dụng lao động khi thay đổi cơ cấu tổ chức?",
#      "expected_article": "Điều 42", "type": "factual"},
#     {"question": "Phương án sử dụng lao động phải có những nội dung gì?",
#      "expected_article": "Điều 44", "type": "factual"},
#     {"question": "Người sử dụng lao động phải thông báo chấm dứt hợp đồng bằng hình thức nào?",
#      "expected_article": "Điều 45", "type": "factual"},
#     {"question": "Hợp đồng lao động vô hiệu trong những trường hợp nào?",
#      "expected_article": "Điều 49", "type": "factual"},
#     {"question": "Ai có thẩm quyền tuyên bố hợp đồng lao động vô hiệu?",
#      "expected_article": "Điều 50", "type": "factual"},
#     {"question": "Xử lý hợp đồng lao động vô hiệu như thế nào?",
#      "expected_article": "Điều 51", "type": "factual"},
 
#     # ── PHẦN 3: CHO THUÊ LẠI LAO ĐỘNG (Điều 52–58) ──────────
#     {"question": "Cho thuê lại lao động là gì?",
#      "expected_article": "Điều 52", "type": "factual"},
#     {"question": "Thời hạn cho thuê lại lao động tối đa là bao nhiêu tháng?",
#      "expected_article": "Điều 53", "type": "numeric"},
#     {"question": "Doanh nghiệp cho thuê lại lao động cần điều kiện gì?",
#      "expected_article": "Điều 54", "type": "condition"},
#     {"question": "Quyền và nghĩa vụ của bên thuê lại lao động là gì?",
#      "expected_article": "Điều 57", "type": "factual"},
 
#     # ── PHẦN 4: ĐÀO TẠO NGHỀ (Điều 59–62) ───────────────────
#     {"question": "Trách nhiệm của người sử dụng lao động về đào tạo nghề?",
#      "expected_article": "Điều 60", "type": "factual"},
#     {"question": "Học nghề để làm việc cho người sử dụng lao động là gì?",
#      "expected_article": "Điều 61", "type": "factual"},
#     {"question": "Khi nào phải ký hợp đồng đào tạo nghề?",
#      "expected_article": "Điều 62", "type": "condition"},
 
#     # ── PHẦN 5: ĐỐI THOẠI & THƯƠNG LƯỢNG (Điều 63–89) ──────
#     {"question": "Đối thoại tại nơi làm việc là gì?",
#      "expected_article": "Điều 63", "type": "factual"},
#     {"question": "Nội dung thương lượng tập thể bao gồm những gì?",
#      "expected_article": "Điều 67", "type": "factual"},
#     {"question": "Thỏa ước lao động tập thể là gì?",
#      "expected_article": "Điều 75", "type": "factual"},
#     {"question": "Thỏa ước lao động tập thể phải gửi đến cơ quan nào trong bao nhiêu ngày?",
#      "expected_article": "Điều 77", "type": "numeric"},
#     {"question": "Hiệu lực của thỏa ước lao động tập thể bắt đầu từ khi nào?",
#      "expected_article": "Điều 78", "type": "factual"},
#     {"question": "Thỏa ước lao động tập thể vô hiệu trong trường hợp nào?",
#      "expected_article": "Điều 86", "type": "condition"},
#     {"question": "Chi phí thương lượng và ký kết thỏa ước do ai chịu?",
#      "expected_article": "Điều 89", "type": "factual"},
 
#     # ── PHẦN 6: TIỀN LƯƠNG (Điều 90–104) ────────────────────
#     {"question": "Mức lương tối thiểu nhằm mục đích gì?",
#      "expected_article": "Điều 91", "type": "factual"},
#     {"question": "Hội đồng tiền lương quốc gia có chức năng gì?",
#      "expected_article": "Điều 92", "type": "factual"},
#     {"question": "Người sử dụng lao động phải xây dựng thang lương bảng lương để làm gì?",
#      "expected_article": "Điều 93", "type": "factual"},
#     {"question": "Tiền lương trả qua người cai thầu được quy định như thế nào?",
#      "expected_article": "Điều 100", "type": "factual"},
#     {"question": "Người sử dụng lao động có được khấu trừ tiền lương tùy ý không?",
#      "expected_article": "Điều 102", "type": "negation"},
#     {"question": "Tiền lương ngừng việc do lỗi của người sử dụng lao động được trả như thế nào?",
#      "expected_article": "Điều 99", "type": "factual"},
#     {"question": "Thưởng cho người lao động được quy định như thế nào?",
#      "expected_article": "Điều 104", "type": "factual"},
 
#     # ── PHẦN 7: THỜI GIỜ LÀM VIỆC & NGHỈ NGƠI (Điều 105–116)
#     {"question": "Giờ làm việc ban đêm tính từ mấy giờ đến mấy giờ?",
#      "expected_article": "Điều 106", "type": "numeric"},
#     {"question": "Làm thêm giờ trong trường hợp đặc biệt được quy định thế nào?",
#      "expected_article": "Điều 108", "type": "factual"},
#     {"question": "Người lao động làm ca được nghỉ chuyển ca bao nhiêu giờ?",
#      "expected_article": "Điều 110", "type": "numeric"},
#     {"question": "Mỗi tuần người lao động được nghỉ tối thiểu bao nhiêu giờ liên tục?",
#      "expected_article": "Điều 111", "type": "numeric"},
#     {"question": "Ngày nghỉ hằng năm tăng thêm theo thâm niên như thế nào?",
#      "expected_article": "Điều 114", "type": "factual"},
#     {"question": "Nghỉ việc riêng khi có người thân mất được nghỉ bao nhiêu ngày?",
#      "expected_article": "Điều 115", "type": "numeric"},
#     {"question": "Thời giờ làm việc đối với công việc có tính chất đặc biệt được quy định thế nào?",
#      "expected_article": "Điều 116", "type": "factual"},
 
#     # ── PHẦN 8: KỶ LUẬT LAO ĐỘNG (Điều 117–131) ─────────────
#     {"question": "Kỷ luật lao động là gì?",
#      "expected_article": "Điều 117", "type": "factual"},
#     {"question": "Doanh nghiệp bao nhiêu người lao động phải ban hành nội quy lao động bằng văn bản?",
#      "expected_article": "Điều 118", "type": "numeric"},
#     {"question": "Nội quy lao động có hiệu lực sau bao nhiêu ngày?",
#      "expected_article": "Điều 121", "type": "numeric"},
#     {"question": "Người lao động bị cấm làm gì khi xử lý kỷ luật?",
#      "expected_article": "Điều 127", "type": "negation"},
#     {"question": "Tạm đình chỉ công việc được áp dụng trong trường hợp nào?",
#      "expected_article": "Điều 128", "type": "condition"},
#     {"question": "Người lao động gây thiệt hại tài sản phải bồi thường như thế nào?",
#      "expected_article": "Điều 129", "type": "factual"},
#     {"question": "Cách xử lý bồi thường thiệt hại căn cứ vào điều gì?",
#      "expected_article": "Điều 130", "type": "factual"},
#     {"question": "Người bị xử lý kỷ luật có quyền khiếu nại không?",
#      "expected_article": "Điều 131", "type": "negation"},
 
#     # ── PHẦN 9: AN TOÀN VỆ SINH LAO ĐỘNG (Điều 132–134) ─────
#     {"question": "Trách nhiệm tuân thủ an toàn vệ sinh lao động thuộc về ai?",
#      "expected_article": "Điều 132", "type": "factual"},
#     {"question": "Chương trình an toàn vệ sinh lao động do cơ quan nào quyết định?",
#      "expected_article": "Điều 133", "type": "factual"},
 
#     # ── PHẦN 10: LAO ĐỘNG NỮ & BÌNH ĐẲNG GIỚI (Điều 135–142)
#     {"question": "Nhà nước có chính sách gì về bình đẳng giới trong lao động?",
#      "expected_article": "Điều 135", "type": "factual"},
#     {"question": "Người sử dụng lao động có trách nhiệm gì đối với lao động nữ?",
#      "expected_article": "Điều 136", "type": "factual"},
#     {"question": "Lao động nữ mang thai tháng thứ 7 được bảo vệ thế nào?",
#      "expected_article": "Điều 137", "type": "factual"},
#     {"question": "Lao động nữ mang thai có quyền đơn phương chấm dứt hợp đồng không?",
#      "expected_article": "Điều 138", "type": "negation"},
#     {"question": "Lao động được bảo đảm việc làm cũ sau khi nghỉ thai sản không?",
#      "expected_article": "Điều 140", "type": "negation"},
#     {"question": "Bộ trưởng nào ban hành danh mục nghề ảnh hưởng xấu đến chức năng sinh sản?",
#      "expected_article": "Điều 142", "type": "factual"},
 
#     # ── PHẦN 11: LAO ĐỘNG CHƯA THÀNH NIÊN (Điều 143–147) ────
#     {"question": "Nguyên tắc sử dụng lao động chưa thành niên là gì?",
#      "expected_article": "Điều 144", "type": "factual"},
#     {"question": "Người chưa đủ 15 tuổi làm việc có cần ký hợp đồng với ai?",
#      "expected_article": "Điều 145", "type": "factual"},
#     {"question": "Thời giờ làm việc của người chưa đủ 15 tuổi tối đa bao nhiêu giờ ngày?",
#      "expected_article": "Điều 146", "type": "numeric"},
#     {"question": "Những công việc nào bị cấm sử dụng người chưa thành niên?",
#      "expected_article": "Điều 147", "type": "negation"},
 
#     # ── PHẦN 12: LAO ĐỘNG CAO TUỔI & ĐẶC THÙ (Điều 148–167)
#     {"question": "Người lao động cao tuổi là ai?",
#      "expected_article": "Điều 148", "type": "factual"},
#     {"question": "Sử dụng người lao động cao tuổi có được ký hợp đồng nhiều lần không?",
#      "expected_article": "Điều 149", "type": "negation"},
#     {"question": "Điều kiện để người lao động nước ngoài làm việc tại Việt Nam?",
#      "expected_article": "Điều 151", "type": "condition"},
#     {"question": "Thời hạn giấy phép lao động cho người nước ngoài tối đa bao nhiêu năm?",
#      "expected_article": "Điều 155", "type": "numeric"},
#     {"question": "Giấy phép lao động hết hiệu lực trong các trường hợp nào?",
#      "expected_article": "Điều 156", "type": "factual"},
#     {"question": "Người lao động khuyết tật bị sử dụng như thế nào là vi phạm?",
#      "expected_article": "Điều 160", "type": "negation"},
#     {"question": "Lao động giúp việc gia đình là gì?",
#      "expected_article": "Điều 161", "type": "factual"},
#     {"question": "Hợp đồng với lao động giúp việc gia đình có bắt buộc bằng văn bản không?",
#      "expected_article": "Điều 162", "type": "negation"},
#     {"question": "Hành vi nào bị nghiêm cấm với lao động giúp việc gia đình?",
#      "expected_article": "Điều 165", "type": "negation"},
 
#     # ── PHẦN 13: BẢO HIỂM & TUỔI NGHỈ HƯU (Điều 168–169) ───
#     {"question": "Người sử dụng lao động có bắt buộc tham gia bảo hiểm xã hội không?",
#      "expected_article": "Điều 168", "type": "negation"},
 
#     # ── PHẦN 14: TỔ CHỨC ĐẠI DIỆN NGƯỜI LAO ĐỘNG (Điều 170–178)
#     {"question": "Người lao động có quyền thành lập công đoàn không?",
#      "expected_article": "Điều 170", "type": "negation"},
#     {"question": "Tổ chức của người lao động tại doanh nghiệp cần bao nhiêu thành viên tối thiểu?",
#      "expected_article": "Điều 173", "type": "numeric"},
#     {"question": "Những hành vi nào người sử dụng lao động bị nghiêm cấm với tổ chức đại diện NLĐ?",
#      "expected_article": "Điều 175", "type": "negation"},
#     {"question": "Nghĩa vụ của người sử dụng lao động với tổ chức đại diện người lao động?",
#      "expected_article": "Điều 177", "type": "factual"},
 
#     # ── PHẦN 15: TRANH CHẤP LAO ĐỘNG (Điều 179–197) ─────────
#     {"question": "Tranh chấp lao động gồm những loại nào?",
#      "expected_article": "Điều 179", "type": "factual"},
#     {"question": "Nguyên tắc giải quyết tranh chấp lao động là gì?",
#      "expected_article": "Điều 180", "type": "factual"},
#     {"question": "Hòa giải viên lao động do ai bổ nhiệm?",
#      "expected_article": "Điều 184", "type": "factual"},
#     {"question": "Hội đồng trọng tài lao động do ai thành lập?",
#      "expected_article": "Điều 185", "type": "factual"},
#     {"question": "Trong khi tranh chấp đang giải quyết, các bên có được hành động đơn phương không?",
#      "expected_article": "Điều 186", "type": "negation"},
#     {"question": "Thời hiệu yêu cầu giải quyết tranh chấp lao động cá nhân là bao lâu?",
#      "expected_article": "Điều 190", "type": "numeric"},
 
#     # ── PHẦN 16: ĐÌNH CÔNG (Điều 198–211) ────────────────────
#     {"question": "Đình công là gì?",
#      "expected_article": "Điều 198", "type": "factual"},
#     {"question": "Người lao động có quyền đình công trong trường hợp nào?",
#      "expected_article": "Điều 199", "type": "condition"},
#     {"question": "Trình tự tiến hành đình công gồm những bước nào?",
#      "expected_article": "Điều 200", "type": "factual"},
#     {"question": "Khi nào cuộc đình công được coi là bất hợp pháp?",
#      "expected_article": "Điều 204", "type": "condition"},
#     {"question": "Nơi nào người lao động không được đình công?",
#      "expected_article": "Điều 209", "type": "negation"},
#     {"question": "Chính phủ có quyền hoãn cuộc đình công không?",
#      "expected_article": "Điều 210", "type": "negation"},
#     {"question": "Tiền lương của người không tham gia đình công nhưng phải ngừng việc?",
#      "expected_article": "Điều 207", "type": "factual"},
#     {"question": "Hành vi nào bị nghiêm cấm trong và sau đình công?",
#      "expected_article": "Điều 208", "type": "negation"},
 
#     # ── PHẦN 17: QUẢN LÝ NHÀ NƯỚC & XỬ LÝ VI PHẠM (Điều 212–220)
#     {"question": "Nội dung quản lý nhà nước về lao động gồm những gì?",
#      "expected_article": "Điều 212", "type": "factual"},
#     {"question": "Nội dung thanh tra lao động bao gồm gì?",
#      "expected_article": "Điều 214", "type": "factual"},
#     {"question": "Thanh tra lao động có những quyền gì?",
#      "expected_article": "Điều 216", "type": "factual"},
#     {"question": "Người vi phạm Bộ luật Lao động bị xử lý như thế nào?",
#      "expected_article": "Điều 217", "type": "factual"},
#     {"question": "Doanh nghiệp dưới 10 lao động được miễn giảm thủ tục gì?",
#      "expected_article": "Điều 218", "type": "factual"},
#     {"question": "Bộ luật Lao động 2019 có hiệu lực từ ngày nào?",
#      "expected_article": "Điều 220", "type": "numeric"},
# ]

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


# ── TẬP CÂU HỎI THÔNG TỤC (informal) để đo tác động của Query Rewriting ──
INFORMAL_BENCHMARK = [
    {"question": "nghỉ đẻ được mấy tháng",                                    "expected_article": "Điều 139", "type": "informal"},
    {"question": "bị đuổi việc oan thì được đền bao nhiêu tiền",               "expected_article": "Điều 41",  "type": "informal"},
    {"question": "lương thử việc có được thấp hơn lương chính không",          "expected_article": "Điều 26",  "type": "informal"},
    {"question": "công ty có được giữ bằng cấp chứng chỉ của nhân viên không", "expected_article": "Điều 17",  "type": "informal"},
    {"question": "ký hợp đồng miệng có hợp lệ không",                         "expected_article": "Điều 14",  "type": "informal"},
    {"question": "làm 3 năm bị cho nghỉ việc thì nhận được tiền gì",           "expected_article": "Điều 46",  "type": "informal"},
    {"question": "bao nhiêu ngày phải báo trước khi tự nghỉ việc",             "expected_article": "Điều 35",  "type": "informal"},
    {"question": "đi làm muộn bị trừ lương có đúng luật không",               "expected_article": "Điều 102", "type": "informal"},
    {"question": "công ty bắt làm thêm giờ mà không trả thêm tiền thì sao",   "expected_article": "Điều 98",  "type": "informal"},
    {"question": "đang mang thai có bị sa thải được không",                    "expected_article": "Điều 137", "type": "informal"},
    {"question": "thử việc xong công ty không nhận thì có được báo trước không","expected_article": "Điều 27",  "type": "informal"},
    {"question": "nghỉ việc không báo trước thì mất gì",                       "expected_article": "Điều 40",  "type": "informal"},
    {"question": "làm ca đêm được thêm bao nhiêu phần trăm lương",             "expected_article": "Điều 98",  "type": "informal"},
    {"question": "mỗi năm được nghỉ phép mấy ngày",                           "expected_article": "Điều 113", "type": "informal"},
    {"question": "công ty nhỏ dưới 10 người có cần nội quy không",             "expected_article": "Điều 118", "type": "informal"},
]


def run_pipeline_benchmark(questions=None):
    """
    Chạy benchmark qua LegalRAGPipeline thật sự — có QueryRewriter.
    So sánh với run_hybrid_benchmark() để đo tác động của Query Rewriting.
    Chỉ đo retrieval (retrieved_chunks), không tốn thêm token LLM.
    """
    print("\n" + "=" * 60)
    print("PIPELINE BENCHMARK (Hybrid + QueryRewriter)")
    print("=" * 60)

    from src.retriever.hybrid_retriever import HybridRetriever
    from src.pipeline.query_rewriter import QueryRewriter
    from src.llm.factory import get_llm_client

    retriever = HybridRetriever()
    reranker = BGEReranker()
    llm = get_llm_client()
    rewriter = QueryRewriter(llm)

    dataset = questions or INFORMAL_BENCHMARK

    hit = 0
    rewrite_count = 0
    fail_cases = []

    for item in dataset:
        q = item["question"]

        # Bước rewrite — giống hệt rag_pipeline.py
        retrieval_query, was_rewritten = rewriter.rewrite(q)
        if was_rewritten:
            rewrite_count += 1

        candidates = retriever.retrieve(retrieval_query, top_k=20)
        reranked = reranker.rerank(retrieval_query, candidates, top_k=8)
        retrieved = [r["metadata"].get("article", "") for r in reranked]

        ok = item["expected_article"] in retrieved
        if ok:
            hit += 1
        else:
            fail_cases.append(item)

        status = "[OK]" if ok else "[X] "
        rewrite_tag = "[rewritten]" if was_rewritten else "[original] "
        top_score = reranked[0].get("rerank_score", "N/A") if reranked else "N/A"
        score_str = f"[score={top_score:.3f}]" if isinstance(top_score, float) else ""
        print(f"{status} {rewrite_tag} {q[:45]} {score_str}")
        if was_rewritten:
            print(f"         → {retrieval_query[:65]}")
        if not ok:
            print(f"    Expected : {item['expected_article']}")
            print(f"    Got      : {retrieved[:3]}")

    rate = hit / len(dataset)
    print(f"\nPipeline Hit Rate@8: {hit}/{len(dataset)} = {rate:.1%}")
    print(f"Queries rewritten  : {rewrite_count}/{len(dataset)}")

    if fail_cases:
        print(f"\n--- {len(fail_cases)} câu hỏi thất bại ---")
        for f in fail_cases:
            print(f"  - {f['question']} → {f['expected_article']}")

    return rate


if __name__ == "__main__":
    import sys

    # Chạy tất cả: python src/evaluation/benchmark.py
    # Chỉ chạy pipeline: python src/evaluation/benchmark.py --pipeline
    pipeline_only = "--pipeline" in sys.argv

    if pipeline_only:
        pipeline_rate = run_pipeline_benchmark()
        print("\n" + "=" * 60)
        print("PIPELINE (với QueryRewriter) — Informal Questions")
        print("=" * 60)
        print(f"Pipeline: {pipeline_rate:.1%}")
    else:
        bm25_rate = run_bm25_benchmark()
        vector_rate = run_vector_benchmark()
        hybrid_rate = run_hybrid_benchmark()
        pipeline_rate = run_pipeline_benchmark()

        print("\n" + "=" * 60)
        print("SUMMARY (Hit Rate@8) — Informal Questions")
        print("=" * 60)
        print(f"BM25              : {bm25_rate:.1%}")
        print(f"Vector            : {vector_rate:.1%}")
        print(f"Hybrid + Reranker : {hybrid_rate:.1%}")
        print(f"Pipeline (Rewrite): {pipeline_rate:.1%}  ← kết quả thực tế user nhận được")