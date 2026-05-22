import os
import shutil
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy





rag_base = os.getenv("ASRAY_BASE_DIR", "/home/ai/asraydata/rag_db")

embeddings = OpenAIEmbeddings(
    model="Qwen3-Embedding-4B",
    api_key=os.getenv("OPENAI_API_KEY", "123"),
    openai_api_base=os.getenv("OPENAI_API_BASE", "http://172.20.0.78:9997/v1")
)

_load_kwargs = {
    "index_name": "index",
    "distance_strategy": DistanceStrategy.COSINE,
}

try:
    FAISS.load_local("/tmp/__check", embeddings, **_load_kwargs, allow_dangerous_deserialization=True)
except TypeError:
    _load_kwargs["allow_dangerous_deserialization"] = True
except Exception:
    _load_kwargs["allow_dangerous_deserialization"] = True


class RAGManager:
    def __init__(self, base_dir=None):
        self.base_dir = base_dir or rag_base
        os.makedirs(self.base_dir, exist_ok=True)
        self._cache = {}

    def _path(self, name):
        return os.path.join(self.base_dir, name)

    def _exists_on_disk(self, name):
        return os.path.exists(os.path.join(self._path(name), "index.faiss"))

    def _save(self, name):
        if name not in self._cache:
            return
        path = self._path(name)
        os.makedirs(path, exist_ok=True)
        self._cache[name].save_local(path, index_name="index")

    def _load(self, name):
        if name in self._cache:
            return self._cache[name]
        vs = FAISS.load_local(self._path(name), embeddings, **_load_kwargs)
        self._cache[name] = vs
        return vs

    def _get_or_create(self, name):
        if name in self._cache:
            return self._cache[name]
        if self._exists_on_disk(name):
            return self._load(name)
        return None

    def add(self, name, texts, metadatas=None):
        if not texts:
            return
        vs = self._get_or_create(name)
        if vs is not None:
            vs.add_texts(texts, metadatas=metadatas)
        else:
            vs = FAISS.from_texts(
                texts, embeddings, metadatas=metadatas,
                distance_strategy=DistanceStrategy.COSINE
            )
        self._cache[name] = vs
        self._save(name)

    def search(self, name, query, k=5):
        vs = self._get_or_create(name)
        if vs is None:
            return []
        return vs.similarity_search(query, k=k)

    def search_with_score(self, name, query, k=5):
        vs = self._get_or_create(name)
        if vs is None:
            return []
        return vs.similarity_search_with_score(query, k=k)

    def delete(self, name, ids=None):
        if ids is None:
            self._cache.pop(name, None)
            path = self._path(name)
            if os.path.exists(path):
                shutil.rmtree(path)
        else:
            vs = self._get_or_create(name)
            if vs is not None:
                vs.delete(ids)
                self._save(name)

    def list_collections(self):
        if not os.path.exists(self.base_dir):
            return []
        return sorted([
            d for d in os.listdir(self.base_dir)
            if os.path.isdir(os.path.join(self.base_dir, d))
            and os.path.exists(os.path.join(self.base_dir, d, "index.faiss"))
        ])

    def count(self, name):
        vs = self._get_or_create(name)
        if vs is None:
            return 0
        return vs.index.ntotal


_manager = None

def get_manager():
    global _manager
    if _manager is None:
        _manager = RAGManager()
    return _manager




