//! Summary logging — robustness table display.

use crate::engine::RobustnessEntry;

/// Print the top-N repeated topologies.
///
/// `best = true`  → top N (appearances desc, std_dev asc, mean desc)
/// `best = false` → bottom N (appearances asc, std_dev desc, mean asc)
pub(crate) fn log_repeated_topologies(
    robustness: &std::collections::HashMap<String, RobustnessEntry>,
    top_n: usize,
    best: bool,
) {
    let mut entries: Vec<&RobustnessEntry> = robustness
        .values()
        .filter(|e| e.count > 1)
        .collect();

    if best {
        entries.sort_by(|a, b| {
            b.count
                .cmp(&a.count)
                .then_with(|| a.std_dev().partial_cmp(&b.std_dev()).unwrap_or(std::cmp::Ordering::Equal))
                .then_with(|| b.mean.partial_cmp(&a.mean).unwrap_or(std::cmp::Ordering::Equal))
        });
    } else {
        entries.sort_by(|a, b| {
            a.count
                .cmp(&b.count)
                .then_with(|| b.std_dev().partial_cmp(&a.std_dev()).unwrap_or(std::cmp::Ordering::Equal))
                .then_with(|| a.mean.partial_cmp(&b.mean).unwrap_or(std::cmp::Ordering::Equal))
        });
    }

    if entries.is_empty() {
        return;
    }

    let n = top_n.min(entries.len());
    let label = if best { "top" } else { "bottom" };
    let has_loss = entries.first().map_or(false, |e| e.has_loss());

    let header: Vec<&str> = if has_loss {
        vec![
            "rank", "appearances", "fit_mean", "fit_sd", "fit_min", "fit_max",
            "loss_mean", "loss_sd", "loss_min", "loss_max", "params", "topo_id",
        ]
    } else {
        vec!["rank", "appearances", "mean", "std_dev", "min", "max", "params", "topo_id"]
    };

    let mut rows: Vec<Vec<String>> = Vec::new();
    for (rank, entry) in entries.iter().take(n).enumerate() {
        let tid = topo_hash(&entry.topology_json);
        if has_loss {
            rows.push(vec![
                format!("#{}", rank + 1),
                entry.count.to_string(),
                format!("{:.4}", entry.mean),
                format!("{:.4}", entry.std_dev()),
                format!("{:.4}", entry.min_fitness),
                format!("{:.4}", entry.max_fitness),
                format!("{:.4}", entry.mean_loss.unwrap_or(0.0)),
                format!("{:.4}", entry.std_dev_loss()),
                format!("{:.4}", entry.min_loss.unwrap_or(0.0)),
                format!("{:.4}", entry.max_loss.unwrap_or(0.0)),
                entry.param_count.to_string(),
                tid,
            ]);
        } else {
            rows.push(vec![
                format!("#{}", rank + 1),
                entry.count.to_string(),
                format!("{:.4}", entry.mean),
                format!("{:.4}", entry.std_dev()),
                format!("{:.4}", entry.min_fitness),
                format!("{:.4}", entry.max_fitness),
                entry.param_count.to_string(),
                tid,
            ]);
        }
    }

    crate::utils::log_utils::log_section_table(
        &format!("repeated topologies ({label} {n})"),
        &header,
        &rows,
        crate::utils::log_utils::TABLE_WIDTH,
    );
}

/// xxh3 hash — 16 hex chars, deterministic, near-zero collisions.
pub(crate) fn topo_hash(topology_json: &str) -> String {
    format!("{:016x}", xxhash_rust::xxh3::xxh3_64(topology_json.as_bytes()))
}