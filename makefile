PYTEST ?= pytest


all:

build: 
	python setup.py build_ext

dev: build
	python -m pip install --no-build-isolation -e .

test-cov:
	rm -rf coverage .coverage
	$(PYTEST) pyutil --showlocals -v

test: test-cov

doc:
	-rm -rf doc/build doc/source/generated
	cd doc; \
	python make.py clean; \
	python make.py html
