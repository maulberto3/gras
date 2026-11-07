//! Data ROLES — which directory trains, which one ranks, which one only reports.

use crate::utils::tabular_data::{self, Dataset, PoolLayout};
use flodl::tensor::Result;

/// The pools a run start (or a resume) builds, plus the layout that produced them.
pub(crate) struct DataRoles {
    /// `train` (+ the ranking pool) concatenated — the one tensor every clock reads.
    /// `None` only for RL, which has no dataset.
    pub(crate) dataset: Option<Dataset>,
    /// Rows of the RANKING pool at the END of `dataset`; `None` = single-pool seeded
    /// split (`PoolSplit::of`), `Some` = pools by construction (`PoolSplit::explicit`).
    pub(crate) explicit_test_rows: Option<usize>,
    /// Which subdirectory ranks: `"val"` or `"test"`; `None` for a single pool.
    /// the fact `engine.json` records and a resume verifies.
    pub(crate) eval_split: Option<String>,
    /// The exam store: a dataset NO clock can reach, scored once at stop.
    pub(crate) exam: Option<Dataset>,
}

/// Resolve every role for `data_dir`, in the order the layout dictates.
pub(crate) fn resolve(data_dir: &std::path::Path, device: flodl::Device) -> Result<DataRoles> {
    let layout = tabular_data::detect_pool_layout(data_dir)?;
    let (dataset, eval_rows) = match layout {
        PoolLayout::TrainVal => {
            let (train, val) = tabular_data::resolve_train_val_datasets(data_dir)?;
            let n = val.len();
            (train.concat(&val)?, n)
        }
        PoolLayout::TrainTest => {
            let (train, test) = tabular_data::resolve_train_test_datasets(data_dir)?;
            let n = test.len();
            (train.concat(&test)?, n)
        }
        PoolLayout::Single => (tabular_data::resolve_dataset(data_dir)?, 0),
    };
    let train_rows = dataset.len() - eval_rows;
    // `Some(0)` would be a lie (`PoolSplit::explicit` with an empty eval pool); a single
    // pool is the only `None`, and there `eval_share` decides the split.
    let explicit_test_rows = match layout {
        PoolLayout::Single => None,
        _ => Some(eval_rows),
    };
    let exam = match tabular_data::exam_dir(data_dir, layout) {
        Some(dir) => Some(tabular_data::resolve_inputs_targets_datasets(&dir)?.to_device(device)?),
        None => None,
    };
    let exam_rows = exam.as_ref().map(|d| d.len());
    tracing::info!(
        "data │ roles: {} · train {} row(s) · eval ({}) {} row(s) · exam {} row(s)",
        match layout {
            PoolLayout::TrainVal => "train/ learns, val/ ranks, test/ reports",
            PoolLayout::TrainTest =>
                "train/ learns, test/ ranks (no val/ — nothing held out but exam/)",
            PoolLayout::Single => "one pool, split by eval_share",
        },
        train_rows,
        layout.eval_dir().unwrap_or("split of train"),
        eval_rows,
        match exam_rows {
            Some(n) => n.to_string(),
            None => "none".to_string(),
        },
    );
    Ok(DataRoles {
        dataset: Some(dataset.to_device(device)?),
        explicit_test_rows,
        eval_split: layout.eval_dir().map(|d| d.to_string()),
        exam,
    })
}
