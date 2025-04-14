.PHONY: convert-notebooks run-notebooks build-notebooks \
	convert-notebook run-notebook build-notebook

convert-notebooks:
	@for notebook in notebooks/*.py; do \
		base=$$(basename $$notebook .py); \
		marimo export ipynb notebooks/$$base.py -o notebooks/$$base.ipynb; \
	done

run-notebooks:
	@for notebook in notebooks/*.ipynb; do \
		base=$$(basename $$notebook .ipynb); \
		jupyter nbconvert --to notebooks/$$base.ipynb --execute --inplace notebooks/$$base.ipynb; \
	done

build-notebooks: convert-notebooks run-notebooks

convert-notebook:
	@base=$$(basename $(NAME) .py); \
	marimo export ipynb notebooks/$$base.py -o notebooks/$$base.ipynb

run-notebook:
	@base=$$(basename $(NAME) .ipynb); \
	jupyter nbconvert --to notebook --execute --inplace notebooks/$$base.ipynb; \

build-notebook: convert-notebook run-notebook
