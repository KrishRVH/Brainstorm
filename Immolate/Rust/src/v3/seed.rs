use crate::seed::{SEED_SPACE, Seed};
use crate::v3::rng::RngState;

#[derive(Clone, Debug)]
pub struct V3State {
    pub seed: Seed,
    pub hashed_seed: f64,
    pub rng: RngState,
}

impl V3State {
    pub fn from_id(id: i64) -> Self {
        let mut seed = Seed::from_id(id.rem_euclid(SEED_SPACE));
        let hashed_seed = seed.pseudohash(0);
        Self {
            seed,
            hashed_seed,
            rng: RngState::default(),
        }
    }

    pub fn next(&mut self) {
        self.seed.next();
        self.hashed_seed = self.seed.pseudohash(0);
        self.rng.clear();
    }
}
