//! expt's rust extension: expt._internal module

extern crate pyo3;
extern crate rustboard_core;

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::Arc;

use pyo3::prelude::*;

use rustboard_core::commit::{Commit, ScalarValue, TagStore};
use rustboard_core::disk_logdir::DiskLogdir;
use rustboard_core::logdir::LogdirLoader;
use rustboard_core::reservoir::Capacity;
use rustboard_core::types::{PluginSamplingHint, Run};

#[pyclass(module = "expt._internal")]
struct TensorboardEventFileReader {
    /// The path to the log directory.
    #[pyo3(get)]
    log_dir: String,

    /// The rustboard commit object where all the eventfile data is stored.
    commit: Commit,
}

type SeriesName = String;
type StepValueMap = BTreeMap<i64, f32>;
type DataFrameLikeMap = HashMap<SeriesName, StepValueMap>;

macro_rules! collection {
    ($($k: expr => $v: expr),* $(,)?) => {{
        core::convert::From::from([$(($k, $v),)*])
    }};
}

#[pymethods]
impl TensorboardEventFileReader {
    #[new]
    pub fn new(log_dir: &str) -> Self {
        let commit = Commit::new();
        return Self {
            log_dir: String::from(log_dir),
            commit,
        };
    }

    pub fn get_data(&self) -> PyResult<DataFrameLikeMap> {
        // Create the reader object which has a short-term lifespan.
        // --samples_per_plugin: collect all scalars, no subsampling
        let plugin_sampling_hint: HashMap<String, Capacity> = collection! {
            "scalars".to_string() => Capacity::Unbounded,
        };
        // TODO: How to can make this a field of the object for longer-term lifespan?
        let mut loader = LogdirLoader::new(
            &self.commit,
            DiskLogdir::new(PathBuf::from(&self.log_dir)),
            0, // no reload threads
            Arc::new(PluginSamplingHint(plugin_sampling_hint)),
        );

        // Read the data from the filesystem.
        loader.reload();

        // For now, we assume that it does not have nested Runs (subdirs)
        // and all the event data is stored in the root of the log directory.
        let run = &Run(".".to_string());
        let run_map = self.commit.runs.read().unwrap();
        let rundata = run_map.get(run);
        if rundata.is_none() {
            // TODO: Maybe throw an exception?
            // rustboard::commit cannot distinguish no eventfiles v.s. empty summary.
            return Ok(DataFrameLikeMap::new());
        }

        // Enumerate the scalar data and collect into a desired Map structure.
        let scalars: &TagStore<ScalarValue> = &rundata.unwrap().read().unwrap().scalars;
        let ret: DataFrameLikeMap = scalars
            .iter()
            .map(|(tag, series)| {
                // (tag: str) -> SortedMap [step: int, value: float]
                let series_name = String::from(&tag.0);
                let s: StepValueMap = series
                    .valid_values()
                    .map(|(step, _, v)| (step.0, v.0))
                    .collect();
                (series_name, s)
            })
            .collect();

        Ok(ret)
    }
}

/// The rust extension module expt._internal (see Cargo.toml)
#[pymodule]
fn _internal(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    let version = env!("CARGO_PKG_VERSION")
        .to_string()
        .replace("-alpha", "a")
        .replace("-beta", "b");

    m.add("__version__", version)?;
    m.add_class::<TensorboardEventFileReader>()?;

    Ok(())
}
