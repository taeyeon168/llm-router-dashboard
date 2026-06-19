import time
import psutil
import queue
import threading
import requests
import pandas as pd
import streamlit as st

# =========================
# MODEL CONFIG (HARDWARE SAFE)
# =========================
MODELS = {
    "HOT": "qwen2.5:7b",
    "WARM": "qwen2.5:14b",
    "COLD": "qwen2.5:27b"
}

FALLBACK = "qwen2.5:7b"
OLLAMA_URL = "http://localhost:11434/api/generate"

# =========================
# PRIORITY QUEUE (HFT STYLE)
# =========================
queue_high = queue.Queue()
queue_mid = queue.Queue()
queue_low = queue.Queue()

results = []

# =========================
# METRICS
# =========================
def get_ram():
    return psutil.virtual_memory().used / 1024**3


def get_gpu_dummy():
    # RTX 3050 not deeply queried in this version (safe abstraction)
    return 4  # GB VRAM assumed


# =========================
# HARDWARE-AWARE ENGINE
# =========================
def predict_safe_state(ram, gpu_vram):
    """
    predictive guard (not reactive)
    """

    # 🔥 strict safety boundary for 3050 4GB
    if gpu_vram <= 4 and ram > 22:
        return "HOT"

    if ram < 14:
        return "HOT"
    elif ram < 20:
        return "WARM"
    else:
        return "WARM"  # COLD disabled by default for safety


def select_queue():
    if not queue_high.empty():
        return queue_high.get(), "HIGH"
    if not queue_mid.empty():
        return queue_mid.get(), "MID"
    if not queue_low.empty():
        return queue_low.get(), "LOW"
    return None, None


# =========================
# OLLAMA CALL (SAFE WRAPPER)
# =========================
def call_ollama(model, prompt):
    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120
        )
        return r.json().get("response", "")
    except Exception as e:
        return f"[ERROR] {e}"


# =========================
# WORKER ENGINE
# =========================
def worker():
    while True:

        item, priority = select_queue()

        if item is None:
            time.sleep(0.1)
            continue

        ram = get_ram()
        state = predict_safe_state(ram, 4)

        model = MODELS.get(state, FALLBACK)

        response = call_ollama(model, item)

        results.append({
            "time": time.time(),
            "priority": priority,
            "model": model,
            "state": state,
            "response": response
        })

        time.sleep(0.1)


# =========================
# START WORKER
# =========================
threading.Thread(target=worker, daemon=True).start()

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="LLM v4.6 Hardware Aware", layout="wide")
st.title("🧠 LLM v4.6 Hardware-Aware Engine")

prompt = st.text_input("Prompt")

priority = st.selectbox("Priority", ["HIGH", "MID", "LOW"])

if st.button("Send"):

    if priority == "HIGH":
        queue_high.put(prompt)
    elif priority == "MID":
        queue_mid.put(prompt)
    else:
        queue_low.put(prompt)

placeholder = st.empty()

# =========================
# DASH LOOP
# =========================
while True:

    ram = get_ram()

    df = pd.DataFrame(results)

    with placeholder.container():

        col1, col2, col3 = st.columns(3)

        col1.metric("RAM (GB)", f"{ram:.2f}")
        col2.metric("Queue HIGH", queue_high.qsize())
        col3.metric("Queue MID", queue_mid.qsize())

        if not df.empty:
            st.subheader("📊 Execution Log")
            st.dataframe(df.tail(15))

    time.sleep(1)
