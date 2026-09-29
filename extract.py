import json
import random
from pathlib import Path
from google import genai
from google.genai import errors
import os
import time
from dotenv import load_dotenv

from entities import Band

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def extract_facts(source_name, text):
    prompt = f"""
        Analyze the following text and extract all structured information about the band The Neighbourhood.
        Fill in the provided schema as accurately and comprehensively as possible based ONLY on the text.
        IMPORTANT RULES:
        1. Extract ONLY facts explicitly stated in the document.
        2. Do NOT invent, guess, or extrapolate any relationships or attributes.
        3. Fill in albums, members, events, songs, and reviews if they are mentioned.
        4. If some data fields are missing in the text, leave them empty or null.
        SOURCE FILE: {source_name}
        DOCUMENT TEXT:
        {text}
    """

    max_attempts_amount = 5
    for attempt in range(1, max_attempts_amount + 1):
        try:
            print(f"Attempt {attempt}")
            interaction = client.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": Band
                }
            )
            return interaction.parsed
        except errors.ServerError as e:
            print(e)
            if attempt < max_attempts_amount:
                wait_time = 2 ** attempt
                print(f"Waiting {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("Can`t release request")
                raise
    return None


def read_files():
    corpus = Path("corpus")
    facts = []

    for file_path in corpus.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")
        ext_facts = extract_facts(file_path.name, text)
        facts.append(ext_facts)
        time.sleep(15)

    output_facts = Path("facts.json")
    output_facts.write_text(
        json.dumps(
            [fact.model_dump() for fact in facts],
            ensure_ascii=False,
            indent=4,
        ),
        encoding="utf-8"
    )

def change_file_structure(facts_amount):
    path = Path("facts.json")
    data = path.read_text(encoding="utf-8")
    parsed_data = json.loads(data)

    facts = []
    for item in parsed_data:
        facts.append({
            "band_name": item["name"],
            "year_created": item["yearCreated"],
        })
        for album in item["albums"]:
            facts.append({
                "album_name": album["name"],
                "album_label": album["label"],
                "album_release_date": album["releaseDate"]
            })
            for song in album["songs"]:
                facts.append({
                    "song_name": song["name"],
                    "album_name": album["name"]
                })
            for producer in album["producers"]:
                facts.append({
                    "producer_name": producer["name"],
                    "producer_position": producer["position"],
                    "album_name": album["name"]
                })
            for review in album["reviews"]:
                if review is not None:
                    facts.append({
                        "review_name": review["publicationName"],
                        "summary": review["summary"]
                    })
                    if isinstance(review["critic"], list):
                        for critic in review["critic"]:
                            if isinstance(critic, dict):
                                facts.append({
                                    "critic_name": critic["name"],
                                    "critic_position": critic["position"]
                                })
        for member in item["members"]:
            facts.append({
                "member_name": member["name"],
                "member_position": member["position"]
            })
        for event in item["events"]:
            facts.append({
                "event_name": event["name"],
                "event_location": event["location"],
                "event_date": event["date"]
            })

    return random.sample(facts, facts_amount)

if __name__ == "__main__":
    # read_files()
    selected_facts = change_file_structure(20)
    for item in selected_facts:
        print(item)