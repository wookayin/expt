.PHONY: all clean rustfmt build_rust benchmark

SRC := $(shell find src/ -type f -name '*.rs')

all: build_rust

# Need to "clean" the existing lib to avoid crash (see setuptools-rust#295)
build_rust: clean rustfmt Cargo.toml Cargo.lock $(SRC)
	python setup.py build_rust --release --inplace
	python -c 'import expt._internal; print(f"{expt._internal.__version__=}")'

benchmark: clean
	pip install -e .
	pytest -v -k TestLargeDataBenchmark -s --run-slow

rustfmt:
	cargo fmt --all --check --verbose --message-format=human

clean:
	rm -rf expt/_internal.*.so
