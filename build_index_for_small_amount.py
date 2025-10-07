# build_index.py
import asyncpg
import numpy as np
import faiss
import os
from dotenv import load_dotenv
import asyncio
import time
from datetime import datetime

load_dotenv()

# --- .env 설정 ---
DB_CONFIG = { "user": os.getenv("DB_USER"), "password": os.getenv("DB_PASSWORD"), "host": os.getenv("DB_HOST"), "port": int(os.getenv("DB_PORT")), "database": os.getenv("DB_DATABASE"), }
ORG_GROUP_ID = os.getenv("ORG_GROUP_ID")
START_DATE_STR = os.getenv("START_DATE")
END_DATE_STR = os.getenv("END_DATE")
VECTOR_DIMENSION = 512
FAISS_DATA_DIR = "/app/faiss_data"
FAISS_INDEX_PATH = os.path.join(FAISS_DATA_DIR, "faiss_index.bin")
FAISS_IDS_PATH = os.path.join(FAISS_DATA_DIR, "faiss_ids.npy")

async def main():
    if not all([ORG_GROUP_ID, START_DATE_STR, END_DATE_STR]):
        print("오류: .env 파일에 ORG_GROUP_ID, START_DATE, END_DATE가 모두 설정되어야 합니다.")
        return
    try:
        start_date = datetime.strptime(START_DATE_STR, "%Y-%m-%d %H:%M:%S")
        end_date = datetime.strptime(END_DATE_STR, "%Y-%m-%d %H:%M:%S")
    except ValueError as e:
        print(f"오류: 날짜 형식 오류 - {e}")
        return

    print(f"--- 인덱스 빌드 시작 (기간: {start_date} ~ {end_date}) ---")
    if not os.path.exists(FAISS_DATA_DIR): os.makedirs(FAISS_DATA_DIR)
    
    # 1. DB 데이터 로딩 및 변환
    print("1. DB 데이터 로딩 및 변환 중...")
    load_start_time = time.time()
    conn = await asyncpg.connect(**DB_CONFIG)
    main_query = f'SELECT id, face_descriptor FROM "{ORG_GROUP_ID}"."events" WHERE face_descriptor IS NOT NULL AND LENGTH(face_descriptor) = {VECTOR_DIMENSION} AND created_at >= $1 AND created_at < $2'
    records = await conn.fetch(main_query, start_date, end_date)
    await conn.close()
    
    if not records:
        print("유효한 벡터 데이터가 없습니다. 종료합니다.")
        return
        
    all_db_ids = np.array([r['id'] for r in records])
    all_vectors = np.array([np.frombuffer(r['face_descriptor'], dtype=np.uint8) for r in records]).astype('float32')
    print(f"  -> 데이터 로딩 및 변환 완료 ({len(records)}개, 소요 시간: {time.time() - load_start_time:.2f}초)")

    # 2. GPU 인덱스 생성 및 데이터 추가
    print("2. GPU 인덱스 생성 및 데이터 추가 중...")
    index_build_start_time = time.time()
    
    res = faiss.StandardGpuResources()
    cpu_index = faiss.IndexFlatL2(VECTOR_DIMENSION)
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    
    print("  -> GPU에 모든 벡터 추가 중...")
    gpu_index.add(all_vectors)
    
    print(f"  -> GPU 인덱스 빌드 완료 (소요 시간: {time.time() - index_build_start_time:.2f}초)")

    # 3. 인덱스 파일 저장
    print("3. 인덱스 파일 저장 중...")
    save_start_time = time.time()
    cpu_index_to_save = faiss.index_gpu_to_cpu(gpu_index)
    faiss.write_index(cpu_index_to_save, FAISS_INDEX_PATH)
    np.save(FAISS_IDS_PATH, all_db_ids)
    print(f"  -> 파일 저장 완료 (소요 시간: {time.time() - save_start_time:.2f}초)")
    print("--- 인덱스 빌드 성공적으로 완료! ---")

if __name__ == "__main__":
    asyncio.run(main())
