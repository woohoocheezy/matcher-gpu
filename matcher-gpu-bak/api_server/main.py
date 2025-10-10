# api_server/main.py
import os
import datetime
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
import asyncpg
import httpx
from dateutil.rrule import rrule, MONTHLY

# 👈 [수정 완료] app = FastAPI()를 먼저 정의
app = FastAPI()
load_dotenv()

DB_CONFIG = { "user": os.getenv("DB_USER"), "password": os.getenv("DB_PASSWORD"), "host": os.getenv("DB_HOST"), "port": int(os.getenv("DB_PORT")), "database": os.getenv("DB_DATABASE"), }
IMAGE_STORE_URL = os.getenv("IMAGE_STORE_PUBLIC_URL", "")
FAISS_SERVER_URL = "http://faiss-server:8001/search"

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.db_pool = await asyncpg.create_pool(**DB_CONFIG)
    app.state.http_client = httpx.AsyncClient()
    yield
    await app.state.db_pool.close()
    await app.state.http_client.aclose()

# lifespan을 app에 등록
app.router.lifespan_context = lifespan

class SearchRequest(BaseModel):
    org_group_id: str; descriptor_hex: str; start_date: datetime.datetime; end_date: datetime.datetime; threshold: float

async def fetch_details(pool, org_group_id, ids):
    if not ids: return []
    p = ', '.join([f'${i+1}' for i in range(len(ids))]); q = f'SELECT id, camera_id, created_at, face_image_path, body_image_path FROM "{org_group_id}"."events" WHERE id IN ({p})'
    return [dict(r) for r in await pool.fetch(q, *ids)]

def get_month_keys_for_range(start_date: datetime.datetime, end_date: datetime.datetime) -> List[str]:
    keys = []
    for dt in rrule(MONTHLY, dtstart=start_date, until=end_date):
        keys.append(f"{dt.year}-{dt.month:02d}")
    return sorted(list(set(keys)))

@app.post("/search/face")
async def search_face(request: Request, data: SearchRequest):
    overall_start_time = time.time()
    try:
        index_keys = get_month_keys_for_range(data.start_date, data.end_date)
        if not index_keys:
            return {"code": "success", "rows": []}
        
        print(f"Searching in monthly indexes: {index_keys}")

        resp = await request.app.state.http_client.post(
            FAISS_SERVER_URL, 
            json={ "descriptor_hex": data.descriptor_hex, "threshold": data.threshold, "index_keys": index_keys }, 
            timeout=30.0
        )
        resp.raise_for_status()
        faiss_result = resp.json()
        matched_ids = faiss_result.get("db_ids", [])
        
        if not matched_ids:
            return {"code": "success", "rows": []}

        details = await fetch_details(request.app.state.db_pool, data.org_group_id, matched_ids)
        
        final_results = []
        for detail in details:
            if detail.get("face_image_path"): detail["face_image_path"] = f"{IMAGE_STORE_URL}{detail['face_image_path']}"
            if detail.get("body_image_path"): detail["body_image_path"] = f"{IMAGE_STORE_URL}{detail['body_image_path']}"
            final_results.append(detail)
        
        print(f"Total request processed in {(time.time() - overall_start_time) * 1000:.2f} ms. Found {len(final_results)} results.")
        return {"code": "success", "rows": final_results}
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Faiss server error: {e}")
    except Exception as e:
        import traceback; traceback.print_exc(); raise HTTPException(status_code=500, detail=str(e))
