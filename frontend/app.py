# frontend/app.py
import streamlit as st
import httpx

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Trợ lý Pháp luật VN", layout="wide")
st.title("⚖️ Trợ lý Pháp luật Việt Nam")

# Domain selector — chuẩn bị cho multi-domain sau
domain = st.sidebar.selectbox(
    "Chọn lĩnh vực pháp luật",
    ["Tất cả", "labor", "civil", "criminal", "traffic"],
    index=0
)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander("📚 Căn cứ pháp lý"):
                for c in msg["citations"]:
                    st.markdown(f"- {c}")

# Input
if prompt := st.chat_input("Nhập câu hỏi pháp luật..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Đang tra cứu..."):
            try:
                resp = httpx.post(
                    f"{API_URL}/chat",
                    json={
                        "query": prompt,
                        "domain": None if domain == "Tất cả" else domain
                    },
                    timeout=60
                )
                data = resp.json()

                st.markdown(data["answer"])

                if data["citations"]:
                    with st.expander("📚 Căn cứ pháp lý"):
                        for c in data["citations"]:
                            st.markdown(f"- {c}")

                st.caption(f"⏱ {data['latency_ms']:.0f}ms | 📄 {data['chunks_used']} điều luật")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": data["answer"],
                    "citations": data["citations"]
                })

            except Exception as e:
                st.error(f"Lỗi: {e}")