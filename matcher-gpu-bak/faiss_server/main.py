# faiss_server/main.py
import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
import os
import math
from typing import List, Dict

FAISS_DATA_DIR = "/app/faiss_data"
VECTOR_DIMENSION = 512 # 👈 [수정 완료] 이 줄을 추가했습니다.

# --- 다중 인덱스 로딩 ---
gpu_indexes: Dict[str, faiss.GpuIndex] = {}
id_maps: Dict[str, np.ndarray] = {}

print("Loading all monthly indexes to GPU...")
res = faiss.StandardGpuResources()
for filename in os.listdir(FAISS_DATA_DIR):
    if filename.endswith("_index.bin"):
        key = filename.replace("_index.bin", "")
        index_path = os.path.join(FAISS_DATA_DIR, filename)
        ids_path = os.path.join(FAISS_DATA_DIR, f"{key}_ids.npy")
        
        if not os.path.exists(ids_path): continue
        
        print(f"  -> Loading index for '{key}'...")
        cpu_index = faiss.read_index(index_path)
        cpu_index.nprobe = 32
        gpu_indexes[key] = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        id_maps[key] = np.load(ids_path)
        print(f"     ... {gpu_indexes[key].ntotal} vectors loaded.")
print("All indexes loaded.")
# --------------------------------

app = FastAPI()

class SearchRequest(BaseModel):
    descriptor_hex: str; threshold: float; index_keys: List[str]

A = 15.56840403; B = -18.32415094; NORMALIZATION_FACTOR = 128.0
def similarity_to_radius_sq(s):
    if s >= 1.0: return 0.0
    if s <= 0.0: return float('inf')
    d = (math.log(1.0/s - 1.0) - B) / A
    r = d * NORMALIZATION_FACTOR
    return r*r
def hex_to_numpy(h):
    if h.startswith('\\x'): h = h[2:]
    return np.frombuffer(bytes.fromhex(h), dtype=np.uint8).astype('float32').reshape(1, -1)

@app.post("/search")
def search(request: SearchRequest):
    try:
        target_keys = [key for key in request.index_keys if key in gpu_indexes]
        if not target_keys:
            return {"db_ids": [], "search_time_ms": 0}

        index_shards = faiss.IndexShards(VECTOR_DIMENSION, True, False)
        for key in target_keys:
            index_shards.add_shard(gpu_indexes[key])
        
        query_vector = hex_to_numpy(request.descriptor_hex)
        radius_sq = similarity_to_radius_sq(request.threshold)
        k_to_search = 10000

        search_start_time = time.time()
        D, I = index_shards.search(query_vector, k_to_search)
        search_time = time.time() - search_start_time
        
        distances = D[0]; indices = I[0]
        shard_indices = indices // 1000000000 
        local_indices = indices % 1000000000

        matched_db_ids = []
        for i in range(len(indices)):
            if indices[i] != -1 and distances[i] <= radius_sq:
                shard_num = shard_indices[i]
                local_idx = local_indices[i]
                key = target_keys[shard_num]
                matched_db_ids.append(int(id_maps[key][local_idx]))
        
        print(f"Searched {len(target_keys)} shards, found {len(matched_db_ids)} results. (Search time: {search_time * 1000:.2f} ms)")
        
        return {"db_ids": matched_db_ids, "search_time_ms": search_time * 1000}
    except Exception as e:
        import traceback; traceback.print_exc(); raise HTTPException(status_code=500, detail=str(e))
