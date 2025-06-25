from agents.memory import MemoryKeeper, memory_tools
from world_state import reset
import milvus_client


def test_search_text_uses_milvus(monkeypatch):
    reset()
    events = {}

    class DummyCollection:
        def __init__(self):
            self.inserted = []

        def insert(self, data):
            self.inserted.append(data)
            return type('R', (), {'primary_keys': list(range(len(data[0])))})

        def flush(self):
            pass

        def search(self, data, field, param=None, limit=None, output_fields=None):
            events['limit'] = limit
            events['fields'] = output_fields
            class Hit:
                distance = 0.1
                entity = {
                    'doc_id': 'd1',
                    'title': 'doc',
                    'page': 0,
                    'text': 'the amazon basin is huge',
                    'citation': 'doc, A, 1900',
                }
                id = 0
            return [[Hit()]]

        def query(self, expr, output_fields=None, partition_names=None, timeout=None, **kwargs):
            events['query'] = expr
            return [{
                'id': 0,
                'doc_id': 'd1',
                'title': 'doc',
                'page': 0,
                'text': 'the amazon basin is huge',
                'citation': 'doc, A, 1900',
            }]

    monkeypatch.setattr('milvus_client.connect_milvus', lambda *a, **k: 'conn')
    monkeypatch.setattr('milvus_client.create_embeddings_collection', lambda: DummyCollection())
    monkeypatch.setattr(memory_tools, '_compute_embedding', lambda text: [0.0] * memory_tools.EMBED_DIM)

    mk = MemoryKeeper()
    docs = [{
        'id': 'd1',
        'title': 'doc',
        'author': 'A',
        'date': '1900',
        'text': 'the amazon basin is huge',
    }]
    mk.index_documents(docs)
    res = mk.search_text('amazon')

    assert events.get('limit') == 10
    assert 'doc_id' in events.get('fields', [])
    assert 'text' in events.get('fields', [])
    assert res and res[0]["doc_id"] == 'd1'
