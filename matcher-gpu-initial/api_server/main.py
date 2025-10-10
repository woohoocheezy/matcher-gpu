# api_server/main.py
import os
import datetime
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
import asyncpg
import httpx

load_dotenv()

DB_CONFIG = {
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT")),
    "database": os.getenv("DB_DATABASE"),
}
IMAGE_STORE_URL = os.getenv("IMAGE_STORE_PUBLIC_URL", "")
FAISS_SERVER_URL = "http://faiss-server:8001/search"

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.db_pool = await asyncpg.create_pool(**DB_CONFIG, min_size=5, max_size=20)
    app.state.http_client = httpx.AsyncClient()
    yield
    await app.state.db_pool.close()
    await app.state.http_client.aclose()

app = FastAPI(lifespan=lifespan)

class SearchRequest(BaseModel):
    org_group_id: str
    descriptor_hex: str
    start_date: datetime.datetime
    end_date: datetime.datetime

async def fetch_details(pool, org_group_id, ids: List[int]) -> List[dict]:
    if not ids: return []
    id_placeholders = ', '.join([f'${i+1}' for i in range(len(ids))])
    query = f'SELECT id, camera_id, created_at, face_image_path, body_image_path FROM "{org_group_id}"."events" WHERE id IN ({id_placeholders})'
    records = await pool.fetch(query, *ids)
    return [dict(r) for r in records]

@app.post("/search/face")
async def search_face(request: Request, data: SearchRequest):
    overall_start_time = time.time()
    try:
        # 1. Faiss 서버에 ID 목록 요청
        faiss_req_start = time.time()
        resp = await request.app.state.http_client.post(
            FAISS_SERVER_URL, json={"descriptor_hex": data.descriptor_hex, "top_k": 100}, timeout=10.0
        )
        resp.raise_for_status()
        faiss_result = resp.json()
        matched_ids = faiss_result.get("db_ids", [])
        print(f"Faiss search took {faiss_result.get('search_time_ms', 0):.2f} ms")

        if not matched_ids:
            return {"code": "success", "rows": []}

        # 2. 받아온 ID로 DB에서 상세 정보 조회
        details = await fetch_details(request.app.state.db_pool, data.org_group_id, matched_ids)

        # 3. 시간 필터링 및 최종 결과 조합
        final_results = []
        start_naive = data.start_date.replace(tzinfo=None)
        end_naive = data.end_date.replace(tzinfo=None)
        
        details_map = {d['id']: d for d in details}
        # Faiss 결과 순서를 유지하며 필터링
        for an_id in matched_ids:
            detail = details_map.get(an_id)
            if detail and start_naive <= detail['created_at'] < end_naive:
                if detail.get("face_image_path"):
                    detail["face_image_path"] = f"{IMAGE_STORE_URL}{detail['face_image_path']}"
                if detail.get("body_image_path"):
                    detail["body_image_path"] = f"{IMAGE_STORE_URL}{detail['body_image_path']}"
                final_results.append(detail)
        
        print(f"Total request processed in {(time.time() - overall_start_time) * 1000:.2f} ms.")
        return {"code": "success", "rows": final_results}
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Faiss server error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
