from interface import create_app


def test_milvus_init_called(monkeypatch):
    called = {}

    def dummy_conn():
        called['connect'] = True
        return 'conn'

    def dummy_coll():
        called['coll'] = True
        return 'collection'

    def dummy_image_coll():
        called['image_coll'] = True
        return 'image_collection'

    monkeypatch.setattr('interface.backend.app.connect_milvus', dummy_conn)
    monkeypatch.setattr('interface.backend.app.create_embeddings_collection', dummy_coll)
    monkeypatch.setattr(
        'interface.backend.app.create_image_embeddings_collection',
        dummy_image_coll,
    )

    app = create_app()

    assert called.get('connect')
    assert called.get('coll')
    assert called.get('image_coll')
    assert app.extensions['milvus_conn'] == 'conn'

