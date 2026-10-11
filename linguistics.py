import re
from collections import Counter
from config import ARCHAEO_TERMS, PERIOD_TERMS


def tokenize(text):
    return re.findall(r'[А-Яа-яA-Za-z\-]{2,}', text.lower())


def extract_entities(text):
    entities = []

    person_pattern = re.compile(r'\b[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+\b')
    year_pattern = re.compile(r'\b(18|19|20)\d{2}\b')
    loc_pattern = re.compile(r'\b(р\.|река|обл\.|область|район|село|деревня|город)\s+[А-ЯЁA-Z][^,.;:\n]+')

    for m in person_pattern.finditer(text):
        entities.append({'type': 'PERSON', 'text': m.group(0)})

    for m in year_pattern.finditer(text):
        entities.append({'type': 'YEAR', 'text': m.group(0)})

    for m in loc_pattern.finditer(text):
        entities.append({'type': 'LOCATION', 'text': m.group(0)})

    low = text.lower()

    for term in ARCHAEO_TERMS:
        if term in low:
            entities.append({'type': 'ARCHAEO_TERM', 'text': term})

    for term in PERIOD_TERMS:
        if term in low:
            entities.append({'type': 'PERIOD_TERM', 'text': term})

    return entities


def corpus_stats(texts):
    cnt = Counter()
    for t in texts:
        cnt.update(tokenize(t))

    return {
        'num_documents': len(texts),
        'num_tokens': int(sum(cnt.values())),
        'top_50_tokens': cnt.most_common(50)
    }