from interface import create_app


def test_milvus_init_called(monkeypatch):
    called = {}

    def dummy():
        called['cnt'] = called.get('cnt', 0) + 1
        return 'conn'

    monkeypatch.setattr('interface.backend.app.connect_milvus', dummy)
    app = create_app()
    assert called.get('cnt') == 1
    assert app.extensions['milvus_conn'] == 'conn'

