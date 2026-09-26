from pathlib import Path
import chromadb

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "nlp_knowledge.md"
DB_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "nlp_knowledge"

class Retriever:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.client = chromadb.PersistentClient(path=str(DB_DIR))
        self.collection = self.client.get_or_create_collection(COLLECTION_NAME)
        self._build_if_needed()

    def _load_sections(self):
        text = DATA_FILE.read_text(encoding="utf-8")
        raw_sections = text.split("\n## ")
        sections = []
        for i, section in enumerate(raw_sections):
            section = section.strip()
            if not section:
                continue
            if i == 0 and section.startswith("# "):
                title = section.splitlines()[0].strip("# ").strip()
                body = "\n".join(section.splitlines()[1:]).strip()
            else:
                lines = section.splitlines()
                title = lines[0].strip().strip("#").strip()
                body = "\n".join(lines[1:]).strip()
            content = f"{title}\n{body}".strip()
            sections.append((str(len(sections)), title, content))
        return sections

    def _build_if_needed(self):
        if self.collection.count() > 0:
            return
        sections = self._load_sections()
        if not sections:
            raise RuntimeError("No NLP knowledge was found in data/nlp_knowledge.md")
        ids = [x[0] for x in sections]
        documents = [x[2] for x in sections]
        metadatas = [{"title": x[1]} for x in sections]
        embeddings = self.embedding_model.encode(documents).tolist()
        self.collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

    def search(self, query, k=4):
        query_embedding = self.embedding_model.encode(query).tolist()
        result = self.collection.query(query_embeddings=[query_embedding], n_results=k)
        return result.get("documents", [[]])[0]
