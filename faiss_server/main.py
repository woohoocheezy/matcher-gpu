# faiss_server/main.py
import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
import os

FAISS_DATA_DIR = "/app/faiss_data"
FAISS_INDEX_PATH = os.path.join(FAISS_DATA_DIR, "faiss_index.bin")
FAISS_IDS_PATH = os.path.join(FAISS_DATA_DIR, "faiss_ids.npy")

# --- Faiss 인덱스 로딩 ---
print("Loading Faiss index to GPU...")
start_time = time.time()
res = faiss.StandardGpuResources()
cpu_index = faiss.read_index(FAISS_INDEX_PATH)
gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)  # 0번 GPU
print(
    f"Index with {gpu_index.ntotal} vectors loaded in {time.time() - start_time:.2f} seconds."
)

print("Loading ID mapping...")
db_ids = np.load(FAISS_IDS_PATH)
print("ID mapping loaded.")
# -------------------------

app = FastAPI()


class FaissSearchRequest(BaseModel):
    descriptor_hex: str
    top_k: int = 100


def hex_to_numpy(hex_str: str) -> np.ndarray:
    if hex_str.startswith("\\x"):
        hex_str = hex_str[2:]
    byte_data = bytes.fromhex(hex_str)
    return np.frombuffer(byte_data, dtype=np.uint8).astype("float32").reshape(1, -1)


@app.post("/search")
def search(request: FaissSearchRequest):
    try:
        query_vector = hex_to_numpy(request.descriptor_hex)

        search_start_time = time.time()
        distances, indices = gpu_index.search(query_vector, request.top_k)
        search_time = time.time() - search_start_time

        results_indices = indices[0]
        matched_db_ids = [int(db_ids[i]) for i in results_indices if i != -1]

        print(
            f"Search found {len(matched_db_ids)} results in {search_time * 1000:.2f} ms"
        )

        return {"db_ids": matched_db_ids, "search_time_ms": search_time * 1000}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
