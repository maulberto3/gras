//! Topology robustness tracking — Welford's online statistics.
//!
//! Tracks how often a topology appears across generations and maintains
//! running min/max/mean/stddev for fitness and loss via Welford's algorithm.

/// Tracks how often a topology appears across generations.
#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct RobustnessEntry {
    pub count: usize,
    pub min_fitness: f32,
    pub max_fitness: f32,
    pub mean: f32,
    pub m2: f32,
    pub min_loss: Option<f32>,
    pub max_loss: Option<f32>,
    pub mean_loss: Option<f32>,
    pub m2_loss: f32,
    pub param_count: usize,
    pub topology_json: String,
}

impl RobustnessEntry {
    pub fn new(fitness: f32, loss: Option<f32>, param_count: usize, topology_json: String) -> Self {
        Self {
            count: 1,
            min_fitness: fitness,
            max_fitness: fitness,
            mean: fitness,
            m2: 0.0,
            min_loss: loss,
            max_loss: loss,
            mean_loss: loss,
            m2_loss: 0.0,
            param_count,
            topology_json,
        }
    }

    /// Welford's online update for fitness.
    pub fn update(&mut self, fitness: f32, loss: Option<f32>) {
        self.count += 1;
        // Fitness stats
        self.min_fitness = self.min_fitness.min(fitness);
        self.max_fitness = self.max_fitness.max(fitness);
        let delta = fitness - self.mean;
        self.mean += delta / self.count as f32;
        let delta2 = fitness - self.mean;
        self.m2 += delta * delta2;
        // Loss stats
        if let Some(l) = loss {
            if let Some(ml) = self.mean_loss {
                let dl = l - ml;
                self.mean_loss = Some(ml + dl / self.count as f32);
                let dl2 = l - self.mean_loss.unwrap();
                self.m2_loss += dl * dl2;
            }
            self.min_loss = Some(self.min_loss.unwrap_or(l).min(l));
            self.max_loss = Some(self.max_loss.unwrap_or(l).max(l));
        }
    }

    /// Sample standard deviation of fitness across appearances.
    pub fn std_dev(&self) -> f32 {
        if self.count < 2 {
            0.0
        } else {
            (self.m2 / (self.count - 1) as f32).sqrt()
        }
    }

    /// Sample standard deviation of loss across appearances.
    pub fn std_dev_loss(&self) -> f32 {
        if self.count < 2 || self.mean_loss.is_none() {
            0.0
        } else {
            (self.m2_loss / (self.count - 1) as f32).sqrt()
        }
    }

    pub fn has_loss(&self) -> bool {
        self.mean_loss.is_some()
    }
}
