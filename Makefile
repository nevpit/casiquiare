.PHONY: dashboard

dashboard:
	pip install -r requirements.txt
	cd interface/frontend && npm install && npm run build
	cd interface && FLASK_APP=backend.app:create_app flask run
