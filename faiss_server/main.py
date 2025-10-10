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
print("Loading Faiss index to CPU...")
start_time = time.time()
cpu_index = faiss.read_index(FAISS_INDEX_PATH)

print("Loading Faiss index to GPU...")
res = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
print(f"Index with {gpu_index.ntotal} vectors loaded in {time.time() - start_time:.2f} seconds.")

print("Loading ID mapping...")
db_ids = np.load(FAISS_IDS_PATH)
print("ID mapping loaded.")
# -------------------------

app = FastAPI()

# --- DTO 정의 ---
class KnnSearchRequest(BaseModel):
    descriptor_hex: str
    top_k: int = 100

class RangeSearchRequest(BaseModel):
    descriptor_hex: str
    threshold: float

# --- 유틸리티 함수 ---
def hex_to_numpy(hex_str: str) -> np.ndarray:
    if hex_str.startswith('\\x'):
        hex_str = hex_str[2:]
    byte_data = bytes.fromhex(hex_str)
    vector = np.frombuffer(byte_data, dtype=np.uint8).astype('float32').reshape(1, -1)
    faiss.normalize_L2(vector)
    return vector

# --- API 엔드포인트 ---

@app.post("/search/knn")
def search_knn(request: KnnSearchRequest):
    """ GPU를 사용하여 가장 가까운 K개의 이웃을 검색합니다. """
    try:
        query_vector = hex_to_numpy(request.descriptor_hex)
        
        search_start_time = time.time()
        distances, indices = gpu_index.search(query_vector, request.top_k)
        search_time = time.time() - search_start_time
        
        results_indices = indices[0]
        matched_db_ids = [int(db_ids[i]) for i in results_indices if i != -1]
        
        print(f"GPU k-NN search found {len(matched_db_ids)} results in {search_time * 1000:.2f} ms")
        
        return {"db_ids": matched_db_ids, "search_time_ms": search_time * 1000}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/range")
def search_range(request: RangeSearchRequest):
    """ CPU를 사용하여 지정된 거리(threshold) 내의 모든 결과를 검색합니다. """
    try:
        # 정확도를 높이기 위해 nprobe 값을 설정 (탐색할 클러스터 개수)
        cpu_index.nprobe = 32
        
        query_vector = hex_to_numpy(request.descriptor_hex)
        
        print(f"Received range search request with threshold: {request.threshold}")

        search_start_time = time.time()
        lims, distances, indices = cpu_index.range_search(query_vector, request.threshold)
        search_time = time.time() - search_start_time
        
        matched_db_ids = [int(db_ids[i]) for i in indices if i != -1]
        
        print(f"CPU Range search (nprobe={cpu_index.nprobe}) found {len(matched_db_ids)} results in {search_time * 1000:.2f} ms")
        
        return {"db_ids": matched_db_ids, "search_time_ms": search_time * 1000}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
