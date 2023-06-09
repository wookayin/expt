Fixtures for Test and Benchmarks
================================

This directory stores test/benchmark fixtures for fast tensorboard log parsing. See [data_loader_test.py] for the benchmarking code, and [data_loader.py] and [src/lib.rs] for the DataLoader implementations.

## Sample CSV data

A tiny CSV file `./fixtures/sample_csv/progress.csv`, which contains 50 rows.

## Small tensorboard logs (37 MB, 4 runs)

An example data from https://tensorboard.dev ([Tensorboard](https://tensorboard.dev/experiment/QFRIzZJpTZCNRzi8N7zomA/)).

See `//expt/data_loader_test.py:_setup_fixture()` or run `$ pytest expt/data_loader_test.py` to download the files.

```bash
8.2M    fixtures/lr_1E-03,conv=1,fc=2
 11M    fixtures/lr_1E-03,conv=2,fc=2
8.1M    fixtures/lr_1E-04,conv=1,fc=2
 10M    fixtures/lr_1E-04,conv=2,fc=2
```


## `edge_cgan` benchmark data (1.3 GB, 8 runs)

A benchmark data from [`gs://tensorboard-bench-logs`](https://console.cloud.google.com/storage/browser/tensorboard-bench-logs) ([README](https://storage.googleapis.com/tensorboard-bench-logs/edge_cgan/README)).

This dataset contains 8 runs across 11 event files, which total 1.3 GB of event
file data and 23 806 644 data points. There are 37 time series across 9 tags.
<!--
Tensorboard 1.x takes *18~20 minutes*, rustboard takes only *~2 seconds* to read the contents (across 8 runs in total) into memory.
-->

```bash
$ cd /path/to/expt/fixtures/
$ mkdir -p edge_cgan && gsutil rsync -x ".*png$" -r gs://tensorboard-bench-logs/edge_cgan edge_cgan/
```

### Dataset and benchmark result (`TestLargeDataBenchmark`):

Environment: Macbook Pro 2021 (M1 Pro), Python 3.11 (arm64). Single-threaded (no parallel).

| dataset                            | size   | #rows   | python    | rust      |
| ---------------------------------- | ------ | ------- | --------- | --------- |
| `edge_cgan/lifull_001`             |  27M   |  119345 |     12 s  |    0.6 s  |
| `edge_cgan/lifull_002`             |  75M   |  332740 |     32 s  |    1.5 s  |
| `edge_cgan/lifull_003`             |  97M   |  389632 |     38 s  |    1.8 s  |
| `edge_cgan/lifull_004`             | 112M   |  463309 |     45 s  |    2.0 s  |
| `edge_cgan/lifull_005`             |  96M   |  386142 |     37 s  |    1.7 s  |
| `edge_cgan/egraph_edge_cgan_001`   | 449M   | 3999766 |    172 s  |    8.6 s  |
| `edge_cgan/egraph_edge_cgan_002`   | 176M   | 1601450 |     69 s  |    3.3 s  |
| `edge_cgan/egraph_edge_cgan_003`   | 241M   | 2136493 |     92 s  |    5.0 s  |

Overall, [rust (`RustTensorboardLogReader`) implementation][src/lib.rs] is **~20x** faster than the naive
python (`TensorboardLogReader`) implementation.

Note that this result does not include serialization overhead that might be quite significant in parallel (multiprocess) loading,
and assumes that data is loaded in full (i.e., no subsampling) with a subsequent conversion to `pd.DataFrame`.
Parallel (multi-core) performance may vary.


[data_loader_test.py]: https://github.com/wookayin/expt/blob/master/expt/data_loader_test.py#L406
[data_loader.py]: https://github.com/wookayin/expt/blob/master/expt/data_loader.py
[src/lib.rs]: https://github.com/wookayin/expt/blob/master/src/lib.rs
