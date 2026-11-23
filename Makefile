.PHONY: install test cov lint fix app docker reproduce

install:
	pip install -r requirements.txt

test:
	python -m pytest

cov:
	python -m pytest --cov=hitflop --cov-report=term-missing:skip-covered

lint:
	ruff check hitflop app tests

fix:
	ruff check --fix hitflop app tests

app:
	streamlit run app/streamlit_app.py

docker:
	docker build -t movie-hit-flop . && docker run --rm -p 8501:8501 movie-hit-flop

reproduce:          # phases 2-6; phase 1 re-scores saved test predictions, see README
	python -m hitflop.upgrade_phase2 && python -m hitflop.upgrade_phase3 && python -m hitflop.upgrade_phase4 && python -m hitflop.upgrade_phase5 && python -m hitflop.upgrade_phase6
