import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from typing import Annotated
from fastapi import HTTPException, Depends, APIRouter

from schemas import TextInput

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')


router = APIRouter(
    prefix="/enpoints",
    tags=["NTKL"]
)


def tokenize_text(text: str):
    return word_tokenize(text)


def pos_tag_text(text: str):
    tokens = word_tokenize(text)
    return pos_tag(tokens)


def ner_text(text: str):
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    named_entities = ne_chunk(pos_tags)
    entities = []
    for chunk in named_entities:
        if hasattr(chunk, 'label'):
            entities.append((chunk.label(), ' '.join(c[0] for c in chunk)))
    return entities


@router.post("/tokenize")
async def tokenize(text_input: Annotated[TextInput, Depends()]):
    try:
        tokens = tokenize_text(text_input.text)
        return {"tokens": tokens}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pos_tag")
async def pos_tagging(text_input: Annotated[TextInput, Depends()]):
    try:
        pos_tags = pos_tag_text(text_input.text)
        return {"pos_tags": pos_tags}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ner")
async def named_entity_recognition(text_input: Annotated[TextInput, Depends()]):
    try:
        entities = ner_text(text_input.text)
        return {"entities": entities}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
