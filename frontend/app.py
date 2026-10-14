import streamlit as st
import httpx

API_URL = "http://localhost:8000"

st.set_page_config(page_title="⚖️ Trợ lý Pháp luật VN", layout="wide")
st.title("⚖️ Trợ lý Pháp luật Việt Nam")
st.caption(
    "Tra cứu trong Bộ luật Lao động 2019 đã lập chỉ mục. Chưa xác minh đầy đủ sửa đổi và hiệu lực hiện hành."
)

tab1, tab2 = st.tabs(["💬 Hỏi đáp pháp luật", "📄 Kiểm tra văn bản / Hợp đồng"])

with tab1:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Hiển thị lịch sử chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            citations = msg.get("citations", [])
            if citations and isinstance(citations, list) and len(citations) > 0:
                with st.expander("📚 Căn cứ pháp lý"):
                    for c in citations:
                        st.markdown(f"- {c}")

    if prompt := st.chat_input("Nhập câu hỏi pháp luật..."):
        # Hiển thị câu hỏi
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Gọi API
        with st.chat_message("assistant"):
            with st.spinner("Đang tra cứu..."):
                try:
                    resp = httpx.post(
                        f"{API_URL}/chat",
                        json={"query": prompt, "domain": None},
                        timeout=httpx.Timeout(120, connect=10),
                    )

                    if resp.status_code != 200:
                        st.error(f"Lỗi API {resp.status_code}: {resp.text}")
                    else:
                        data = resp.json()
                        answer = data.get("answer", "Không có câu trả lời")
                        citations = data.get("citations", [])
                        latency = data.get("latency_ms", 0)

                        st.markdown(answer)

                        if isinstance(citations, list) and len(citations) > 0:
                            with st.expander("📚 Căn cứ pháp lý"):
                                for c in citations:
                                    st.markdown(f"- {c}")

                        st.caption(f"⏱ {latency:.0f}ms")

                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": answer,
                                "citations": (
                                    citations if isinstance(citations, list) else []
                                ),
                            }
                        )

                except httpx.TimeoutException:
                    st.error("Timeout — thử lại sau.")
                except Exception as e:
                    st.error(f"Lỗi kết nối: {e}")

with tab2:
    st.subheader("📄 Kiểm tra hợp đồng / văn bản pháp lý")
    st.caption(
        "Upload tối đa 3 file PDF hoặc Word — "
        "Đối chiếu từng phần với dữ liệu luật hiện có và nêu rõ phần chưa đủ căn cứ."
    )

    uploaded_files = st.file_uploader(
        "Chọn file (PDF hoặc Word)",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        help="Tối đa 3 file",
    )

    question = st.text_area(
        "Câu hỏi cụ thể (không bắt buộc)",
        placeholder=(
            "Ví dụ: Điều khoản thời gian báo trước có đúng luật không?\n"
            "Nếu để trống, AI sẽ tự kiểm tra toàn bộ văn bản."
        ),
        height=80,
    )

    if st.button("🔍 Kiểm tra", type="primary"):
        if not uploaded_files:
            st.warning("Vui lòng upload ít nhất 1 file")
        elif len(uploaded_files) > 3:
            st.error("Tối đa 3 file")
        else:
            with st.spinner("Đang phân tích văn bản và đối chiếu pháp luật..."):
                try:
                    files = [
                        (
                            "files",
                            (f.name, f.read(), f.type or "application/octet-stream"),
                        )
                        for f in uploaded_files
                    ]
                    data = {}
                    if question.strip():
                        data["question"] = question.strip()

                    resp = httpx.post(
                        f"{API_URL}/analyze-document",
                        files=files,
                        data=data,
                        timeout=httpx.Timeout(1800, connect=10),
                    )

                    if resp.status_code != 200:
                        st.error(
                            f"Lỗi API {resp.status_code}: {resp.json().get('detail', resp.text)}"
                        )
                    else:
                        result = resp.json()
                        answer = result.get("answer", "")
                        citations = result.get("citations", [])
                        latency = result.get("latency_ms", 0)
                        files_analyzed = result.get("files_analyzed", [])

                        if result.get("status") == "partial":
                            st.warning(
                                "Báo cáo chưa hoàn tất tất cả các phần. Xem các giới hạn bên dưới."
                            )
                        else:
                            st.success(
                                f"Đã xử lý {len(files_analyzed)} file trong {latency:.0f}ms"
                            )
                        st.markdown(answer)

                        if isinstance(citations, list) and len(citations) > 0:
                            with st.expander("📚 Điều luật đã tham chiếu"):
                                for c in citations:
                                    st.markdown(f"- {c}")

                except httpx.TimeoutException:
                    st.error("Timeout — file quá lớn hoặc server chậm. Thử lại.")
                except Exception as e:
                    st.error(f"Lỗi kết nối: {e}")
