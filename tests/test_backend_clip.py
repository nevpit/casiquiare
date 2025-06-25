from interface import create_app


def test_clip_model_loaded(monkeypatch):
    called = {}

    def dummy_load():
        called['load'] = True
        return 'model'

    monkeypatch.setattr('interface.backend.app.load_clip_model', dummy_load)

    app = create_app()

    assert called.get('load')
    assert app.extensions['clip_model'] == 'model'
