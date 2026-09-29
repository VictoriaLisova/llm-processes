import os
import chromadb
from dotenv import load_dotenv
from google import genai
from chunking import div_text_by_chunks
from chromadb.utils.embedding_functions import GoogleGeminiEmbeddingFunction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

# QUESTIONS = ["Who produced the album Wiped Out!",
#              "Which label released the self-titled album The Neighbourhood in 2018",
#              "When was the deluxe edition of Chip Chrome & the Mono-Tones released"]

QUESTIONS = ["When did the band stop making music for a period of time",
             "What was the band`s position on the US Billboard 200"]

CHUNKS_AMOUNT = 5

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
chroma_client = chromadb.PersistentClient(path="./chroma_db")

def embedding(data):
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=data
    )

    return response.embeddings[0].values

def save_embedding(embedded_data):
    try:
        chroma_client.delete_collection("gemini-embedding")
    except ValueError:
        pass

    collection = chroma_client.create_collection(
        name="gemini-embedding",
        embedding_function=GoogleGeminiEmbeddingFunction(model_name='gemini-embedding-001')
    )

    for data in embedded_data:
        collection.add(
            ids=[data["id"]],
            documents=[data["chunk"]],
            embeddings=[data["embedding"]]
        )

    print(collection.get(include=["embeddings"]))

    return collection

def keywords_search(chunks):
    documents = [chunk["chunk"] for chunk in chunks]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    results = []
    for question in QUESTIONS:
        query_vector = vectorizer.transform([question])
        similarity = cosine_similarity(query_vector, tfidf_matrix)[0]
        top_indexes = similarity.argsort()[-CHUNKS_AMOUNT:][::-1]
        question_results = []
        for index in top_indexes:
            question_results.append({
                "id": chunks[index]["id"],
                "chunk": chunks[index]["chunk"],
                "score": similarity[index]
            })
        results.append({
            "question": question,
            "results": question_results
        })

    return results

def semantic_search(collection):
    results = []
    for question in QUESTIONS:
        query_embedding = embedding(question)
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=CHUNKS_AMOUNT
        )
        results.append({
            "question": question,
            "results": result
        })

    return results

def get_collection():
    return chroma_client.get_collection(
        name="gemini-embedding",
        embedding_function=GoogleGeminiEmbeddingFunction(model_name='gemini-embedding-001')
    )

if __name__ == "__main__":
    chunks = div_text_by_chunks()

    for chunk in chunks:
        embedding_results = embedding(chunk["chunk"])
        chunk["embedding"] = embedding_results
        print(embedding_results)

    collection = save_embedding(chunks)
    semantic_search_result = semantic_search(collection)

    print("SEMANTIC SEARCH RESULTS:")
    for result in semantic_search_result:
        print(result)
    keywords_search_result = keywords_search(chunks)
    print("KEYWORDS SEARCH RESULTS:")
    for result in keywords_search_result:
        print(result)
