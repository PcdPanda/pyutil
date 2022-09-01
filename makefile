.PHONY : doc clean
PYTHON ?= python
PYTEST ?= pytest


all:


build: 
	python setup.py build_ext

clean:
	$(PYTHON) setup.py clean

doc:
	-rm -rf doc/build doc/source/generated
	cd doc; \
	$(MAKE) html

dev: build
	python -m pip install --no-build-isolation -e .

test-cov:
	rm -rf coverage .coverage
	$(PYTEST) pyutil --showlocals -v

test: test-cov

install: pip install .
