from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

def div_text_by_chunks():
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150,
                                              separators=["\n\n", "\n", ". ",  "? ", "! ", "  ", ""])
    chunks = []
    corpus = Path("corpus")
    for file_path in corpus.glob("*.txt"):
        file_text = file_path.read_text(encoding="utf-8")
        file_chunks = splitter.split_text(file_text)

        for i, chunk in enumerate(file_chunks):
            chunks.append({
                "id": f"{file_path}_{i}",
                "source": file_path.name,
                "chunk": chunk
            })

    return chunks

if __name__ == "__main__":
    corpus_chunks = div_text_by_chunks()
    print(f"Number of created chunks: {len(corpus_chunks)}")

    for chunk in corpus_chunks:
        print(chunk)
