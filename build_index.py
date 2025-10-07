# build_index.py
import asyncpg
import numpy as np
import faiss
import os
from dotenv import load_dotenv
import asyncio
import time
from datetime import datetime  # 👈 [수정] datetime 모듈 추가

load_dotenv()

# --- .env 파일에서 모든 설정값 읽어오기 ---
DB_CONFIG = {
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT")),
    "database": os.getenv("DB_DATABASE"),
}
ORG_GROUP_ID = os.getenv("ORG_GROUP_ID")
# .env에서 날짜 문자열 읽어오기
START_DATE_STR = os.getenv("START_DATE")
END_DATE_STR = os.getenv("END_DATE")

# --- 중요 설정 ---
VECTOR_DIMENSION = 512
FAISS_DATA_DIR = "/app/faiss_data"
FAISS_INDEX_PATH = os.path.join(FAISS_DATA_DIR, "faiss_index.bin")
FAISS_IDS_PATH = os.path.join(FAISS_DATA_DIR, "faiss_ids.npy")

# --- 메모리 관리를 위한 배치 설정 ---
BATCH_SIZE = 100000
TRAIN_SAMPLE_SIZE = 1000000


async def main():
    # --- 설정값 유효성 검사 ---
    if not all([ORG_GROUP_ID, START_DATE_STR, END_DATE_STR]):
        print(
            "오류: .env 파일에 ORG_GROUP_ID, START_DATE, END_DATE가 모두 설정되어야 합니다."
        )
        return

    # 👈 [수정 완료] 문자열을 datetime 객체로 변환
    try:
        start_date = datetime.strptime(START_DATE_STR, "%Y-%m-%d %H:%M:%S")
        end_date = datetime.strptime(END_DATE_STR, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        print("오류: .env의 날짜 형식이 'YYYY-MM-DD HH:MI:SS'와 맞지 않습니다.")
        return

    print(f"--- 인덱스 빌드 시작 (기간: {start_date} ~ {end_date}) ---")

    if not os.path.exists(FAISS_DATA_DIR):
        os.makedirs(FAISS_DATA_DIR)

    print("Connecting to DB...")
    conn = await asyncpg.connect(**DB_CONFIG)

    # 1. [훈련] 지정된 기간 내 데이터의 일부만 가져와 인덱스 훈련
    print(f"1. 훈련용 데이터 샘플 가져오는 중 (최대 {TRAIN_SAMPLE_SIZE}개)...")
    train_start_time = time.time()

    train_query = f"""
        SELECT face_descriptor FROM "{ORG_GROUP_ID}"."events"
        WHERE 
            face_descriptor IS NOT NULL AND 
            LENGTH(face_descriptor) = {VECTOR_DIMENSION} AND
            created_at >= $1 AND created_at < $2
        ORDER BY RANDOM() 
        LIMIT {TRAIN_SAMPLE_SIZE};
    """
    # 👈 [수정 완료] 변환된 datetime 객체를 쿼리에 전달
    train_records = await conn.fetch(train_query, start_date, end_date)
    print(
        f"  -> 훈련용 데이터 가져오기 완료 ({len(train_records)}개, 소요 시간: {time.time() - train_start_time:.2f}초)"
    )

    if len(train_records) < 1000:
        print("훈련용 데이터 샘플이 너무 적습니다. 종료합니다.")
        await conn.close()
        return

    print(f"2. Faiss 인덱스 훈련 중...")
    train_start_time = time.time()
    train_vectors = np.array(
        [np.frombuffer(r["face_descriptor"], dtype=np.uint8) for r in train_records]
    ).astype("float32")

    res = faiss.StandardGpuResources()
    nlist = int(np.sqrt(len(train_records)) * 4)
    quantizer = faiss.IndexFlatL2(VECTOR_DIMENSION)
    gpu_index = faiss.GpuIndexIVFFlat(res, VECTOR_DIMENSION, nlist, faiss.METRIC_L2)
    gpu_index.train(train_vectors)
    del train_vectors
    print(f"  -> 인덱스 훈련 완료 (소요 시간: {time.time() - train_start_time:.2f}초)")

    # 2. [추가] 지정된 기간의 전체 데이터를 커서(Cursor)를 사용해 배치 단위로 처리
    print("3. 전체 데이터를 배치 단위로 인덱스에 추가하는 중...")
    add_start_time = time.time()
    all_db_ids = []
    total_vectors_added = 0

    main_query = f'SELECT id, face_descriptor FROM "{ORG_GROUP_ID}"."events" WHERE face_descriptor IS NOT NULL AND created_at >= $1 AND created_at < $2'

    async with conn.transaction():
        # 👈 [수정 완료] 변환된 datetime 객체를 쿼리에 전달
        async for record in conn.cursor(main_query, start_date, end_date):
            if len(record["face_descriptor"]) == VECTOR_DIMENSION:
                all_db_ids.append(record["id"])
                vector = (
                    np.frombuffer(record["face_descriptor"], dtype=np.uint8)
                    .astype("float32")
                    .reshape(1, -1)
                )
                gpu_index.add(vector)
                total_vectors_added += 1
                if total_vectors_added % BATCH_SIZE == 0:
                    print(
                        f"  ... {total_vectors_added}개 벡터 추가됨 (진행 시간: {time.time() - add_start_time:.2f}초)"
                    )

    await conn.close()
    print(
        f"  -> 전체 데이터 추가 완료 ({total_vectors_added}개, 총 소요 시간: {time.time() - add_start_time:.2f}초)"
    )

    # 3. [저장] 완성된 인덱스와 ID 맵 저장
    print("4. 인덱스 파일 저장 중...")
    save_start_time = time.time()
    cpu_index = faiss.index_gpu_to_cpu(gpu_index)
    faiss.write_index(cpu_index, FAISS_INDEX_PATH)
    np.save(FAISS_IDS_PATH, np.array(all_db_ids))
    print(f"  -> 파일 저장 완료 (소요 시간: {time.time() - save_start_time:.2f}초)")
    print("--- 인덱스 빌드 성공적으로 완료! ---")


if __name__ == "__main__":
    asyncio.run(main())
