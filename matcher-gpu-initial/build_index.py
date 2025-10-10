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

# --- 메모리 관리 및 성능을 위한 설정 ---
BATCH_SIZE = 50000
TRAIN_SAMPLE_SIZE = 500000

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

    print(f"--- 인덱스 빌드 시작 (대용량 압축 모드, 기간: {start_date} ~ {end_date}) ---")
    if not os.path.exists(FAISS_DATA_DIR): os.makedirs(FAISS_DATA_DIR)
    
    conn = await asyncpg.connect(**DB_CONFIG)
    
    # 1. [훈련]
    print(f"1. 훈련용 데이터 샘플 가져오는 중 (최대 {TRAIN_SAMPLE_SIZE}개)...")
    train_start_time = time.time()
    train_query = f'''
        SELECT face_descriptor FROM "{ORG_GROUP_ID}"."events" TABLESAMPLE SYSTEM (5)
        WHERE face_descriptor IS NOT NULL AND LENGTH(face_descriptor) = {VECTOR_DIMENSION} AND created_at >= $1 AND created_at < $2
        LIMIT {TRAIN_SAMPLE_SIZE};
    '''
    train_records = await conn.fetch(train_query, start_date, end_date)
    
    if len(train_records) < 256:
        print("훈련용 데이터 샘플이 너무 적습니다. 종료합니다."); await conn.close(); return

    print(f"  -> 훈련용 데이터 가져오기 완료 ({len(train_records)}개, 소요 시간: {time.time() - train_start_time:.2f}초)")

    print(f"2. Faiss 인덱스 훈련 중 (CPU)...")
    train_start_time = time.time()
    train_vectors = np.array([np.frombuffer(r['face_descriptor'], dtype=np.uint8) for r in train_records]).astype('float32')
    
    # 👈 [수정 완료] m 값을 64에서 32로 줄여 GPU 아키텍처 한계에 맞춤
    nlist = min(16384, int(np.sqrt(len(train_records)) * 8))
    m = 32  # 벡터를 32개 조각으로 나눠서 압축
    bits = 8
    
    quantizer = faiss.IndexFlatL2(VECTOR_DIMENSION)
    cpu_index = faiss.IndexIVFPQ(quantizer, VECTOR_DIMENSION, nlist, m, bits)
    
    cpu_index.train(train_vectors)
    del train_records, train_vectors
    print(f"  -> CPU 인덱스 훈련 완료 (소요 시간: {time.time() - train_start_time:.2f}초)")

    # 3. [추가]
    print("3. 전체 데이터를 배치 단위로 GPU 인덱스에 추가하는 중...")
    add_start_time = time.time()
    
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    
    all_db_ids = []
    batch_vectors = []
    total_vectors_added = 0
    
    main_query = f'SELECT id, face_descriptor FROM "{ORG_GROUP_ID}"."events" WHERE face_descriptor IS NOT NULL AND LENGTH(face_descriptor) = {VECTOR_DIMENSION} AND created_at >= $1 AND created_at < $2'
    
    async with conn.transaction():
        async for record in conn.cursor(main_query, start_date, end_date):
            all_db_ids.append(record['id'])
            batch_vectors.append(np.frombuffer(record['face_descriptor'], dtype=np.uint8))
            
            if len(batch_vectors) == BATCH_SIZE:
                gpu_index.add(np.array(batch_vectors).astype('float32'))
                total_vectors_added += len(batch_vectors)
                print(f"  ... {total_vectors_added}개 벡터 추가됨 (진행 시간: {time.time() - add_start_time:.2f}초)")
                batch_vectors = []

    if batch_vectors:
        gpu_index.add(np.array(batch_vectors).astype('float32'))
        total_vectors_added += len(batch_vectors)
        print(f"  ... {total_vectors_added}개 벡터 추가됨 (진행 시간: {time.time() - add_start_time:.2f}초)")

    await conn.close()
    print(f"  -> 전체 데이터 추가 완료 ({total_vectors_added}개, 총 소요 시간: {time.time() - add_start_time:.2f}초)")

    # 4. [저장]
    print("4. 인덱스 파일 저장 중...")
    save_start_time = time.time()
    cpu_index_to_save = faiss.index_gpu_to_cpu(gpu_index)
    faiss.write_index(cpu_index_to_save, FAISS_INDEX_PATH)
    np.save(FAISS_IDS_PATH, np.array(all_db_ids))
    print(f"  -> 파일 저장 완료 (소요 시간: {time.time() - save_start_time:.2f}초)")
    print("--- 인덱스 빌드 성공적으로 완료! ---")

if __name__ == "__main__":
    asyncio.run(main())
