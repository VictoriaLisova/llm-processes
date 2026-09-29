from typing import List, Optional

from pydantic import BaseModel

class Song(BaseModel):
    name: str

class Person(BaseModel):
    name: str
    position: str

class Review(BaseModel):
    publicationName: str
    critic: Optional[Person] = None
    summary: Optional[str] = None

class Album(BaseModel):
    name: str
    label: str
    releaseDate: str
    songs: List[Song] = []
    producers: List[Person] = []
    reviews: List[Review] = []

class Event(BaseModel):
    name: str
    location: Optional[str] = None
    date: Optional[str] = None

class Band(BaseModel):
    name: str
    yearCreated: Optional[int] = None
    albums: List[Album] = []
    members: List[Person] = []
    events: List[Event] = []
