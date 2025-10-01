import os
import json
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

# -----------------------
# 1. 환경변수 로드
# -----------------------
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_KEY:
    raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다. .env를 확인하세요.")

# -----------------------
# 2. ChromaDB 클라이언트 설정
# -----------------------
client = chromadb.PersistentClient(path="./data/chroma_openai")

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_KEY,
    model_name="text-embedding-3-small"   # 필요 시 "text-embedding-3-large" 로 변경 가능
)

collection = client.get_or_create_collection(
    name="process_meta",
    embedding_function=openai_ef
)

# -----------------------
# 3. meta_json 폴더 읽기
# -----------------------
meta_dir = Path("./data/meta_json")
if not meta_dir.exists():
    raise RuntimeError("data/meta_json 폴더가 없습니다. CSV를 먼저 처리하세요.")

files = list(meta_dir.glob("*.json"))
if not files:
    raise RuntimeError("meta_json 폴더에 JSON 파일이 없습니다. CSV → JSON 메타데이터 변환을 먼저 실행하세요.")

# -----------------------
# 4. JSON 파일을 컬렉션에 업로드
# -----------------------
for file in files:
    with open(file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    doc_id = file.stem
    document = json.dumps(meta, ensure_ascii=False)

    collection.upsert(
        ids=[doc_id],
        documents=[document],
        metadatas=[{"source": str(file)}]
    )
    print(f"[OK] {file.name} → ChromaDB 저장 완료")

print("=== 모든 메타데이터가 ChromaDB에 인덱싱 완료되었습니다. ===")
