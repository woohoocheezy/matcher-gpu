# build_index.py
import asyncpg
import numpy as np
import faiss
import os
from dotenv import load_dotenv
import asyncio
import time
from datetime import datetime
from dateutil.rrule import rrule, MONTHLY

load_dotenv()

# --- .env 설정 ---
DB_CONFIG = { "user": os.getenv("DB_USER"), "password": os.getenv("DB_PASSWORD"), "host": os.getenv("DB_HOST"), "port": int(os.getenv("DB_PORT")), "database": os.getenv("DB_DATABASE"), }
ORG_GROUP_ID = os.getenv("ORG_GROUP_ID")
VECTOR_DIMENSION = 512
FAISS_DATA_DIR = "/app/faiss_data"

# --- 배치 설정 ---
BATCH_SIZE = 50000
TRAIN_SAMPLE_SIZE = 250000

async def build_index_for_month(conn, year, month):
    month_str = f"{year}-{month:02d}"
    index_path = os.path.join(FAISS_DATA_DIR, f"{month_str}_index.bin")
    ids_path = os.path.join(FAISS_DATA_DIR, f"{month_str}_ids.npy")

    if os.path.exists(index_path):
        print(f"--- [{month_str}] 월 인덱스가 이미 존재하므로 건너뜁니다. ---")
        return

    print(f"--- [{month_str}] 월 인덱스 빌드 시작 ---")

    start_date = datetime(year, month, 1)
    end_date = start_date.replace(month=start_date.month % 12 + 1, year=start_date.year + start_date.month // 12)

    # 1. [훈련]
    print(f"1. [{month_str}] 훈련용 데이터 샘플 가져오는 중...")
    train_start_time = time.time()
    train_query = f'''
        SELECT face_descriptor FROM "{ORG_GROUP_ID}"."events" TABLESAMPLE SYSTEM (20)
        WHERE face_descriptor IS NOT NULL AND LENGTH(face_descriptor) = {VECTOR_DIMENSION} AND created_at >= $1 AND created_at < $2
        LIMIT {TRAIN_SAMPLE_SIZE};
    '''
    train_records = await conn.fetch(train_query, start_date, end_date)
    
    if len(train_records) < 256:
        print(f"  -> [{month_str}] 훈련용 데이터가 부족하여 인덱스를 생성할 수 없습니다."); return

    print(f"  -> 가져오기 완료 ({len(train_records)}개, {time.time() - train_start_time:.2f}초)")

    print(f"2. [{month_str}] Faiss 인덱스 훈련 중 (CPU)...")
    train_start_time = time.time()
    train_vectors = np.array([np.frombuffer(r['face_descriptor'], dtype=np.uint8) for r in train_records]).astype('float32')
    
    nlist = min(4096, int(np.sqrt(len(train_records)) * 8)); m = 32; bits = 8
    quantizer = faiss.IndexFlatL2(VECTOR_DIMENSION)
    cpu_index = faiss.IndexIVFPQ(quantizer, VECTOR_DIMENSION, nlist, m, bits)
    cpu_index.train(train_vectors)
    del train_records, train_vectors
    print(f"  -> 훈련 완료 ({time.time() - train_start_time:.2f}초)")

    # 3. [추가]
    print(f"3. [{month_str}] 전체 데이터를 GPU 인덱스에 추가하는 중...")
    add_start_time = time.time()
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    
    all_db_ids = []; batch_vectors = []; total_vectors_added = 0
    main_query = f'SELECT id, face_descriptor FROM "{ORG_GROUP_ID}"."events" WHERE face_descriptor IS NOT NULL AND LENGTH(face_descriptor) = {VECTOR_DIMENSION} AND created_at >= $1 AND created_at < $2'
    
    async with conn.transaction():
        async for record in conn.cursor(main_query, start_date, end_date):
            all_db_ids.append(record['id'])
            batch_vectors.append(np.frombuffer(record['face_descriptor'], dtype=np.uint8))
            if len(batch_vectors) == BATCH_SIZE:
                gpu_index.add(np.array(batch_vectors).astype('float32'))
                total_vectors_added += len(batch_vectors); batch_vectors = []
                print(f"  ... {total_vectors_added}개 추가됨")
    if batch_vectors:
        gpu_index.add(np.array(batch_vectors).astype('float32'))
        total_vectors_added += len(batch_vectors)
    
    print(f"  -> 추가 완료 ({total_vectors_added}개, {time.time() - add_start_time:.2f}초)")

    # 4. [저장]
    print(f"4. [{month_str}] 인덱스 파일 저장 중...")
    cpu_index_to_save = faiss.index_gpu_to_cpu(gpu_index)
    faiss.write_index(cpu_index_to_save, os.path.join(FAISS_DATA_DIR, f"{month_str}_index.bin"))
    np.save(os.path.join(FAISS_DATA_DIR, f"{month_str}_ids.npy"), np.array(all_db_ids))
    print(f"--- [{month_str}] 월 인덱스 빌드 완료 ---")

async def main():
    if not os.path.exists(FAISS_DATA_DIR): os.makedirs(FAISS_DATA_DIR)
    conn = await asyncpg.connect(**DB_CONFIG)

    period_query = f'SELECT MIN(created_at) as min_date, MAX(created_at) as max_date FROM "{ORG_GROUP_ID}"."events";'
    period = await conn.fetchrow(period_query)
    if not period or not period['min_date']:
        print("DB에 데이터가 없습니다. 종료합니다."); await conn.close(); return

    start_date, end_date = period['min_date'], period['max_date']
    print(f"전체 데이터 기간: {start_date} ~ {end_date}")

    for dt in rrule(MONTHLY, dtstart=start_date, until=end_date):
        await build_index_for_month(conn, dt.year, dt.month)
        
    await conn.close()
    print("모든 인덱스 빌드 작업 완료.")

if __name__ == "__main__":
    asyncio.run(main())
