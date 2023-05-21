pub mod monte_carlo;
pub mod q_learning;
pub mod q_learning_lambda;
pub mod q_learning_beta;
pub mod sarsa;
pub mod sarsa_lambda;
pub mod value_iteration;

use std::collections::BTreeMap;

use rand_chacha::ChaCha20Rng;

use crate::mdp::{Action, Mdp, State};

pub trait StateActionAlgorithm {
    // default implementation
    fn run(
        &self,
        mdp: &Mdp,
        episodes: usize,
        rng: &mut ChaCha20Rng,
    ) -> BTreeMap<(State, Action), f64> {
        let mut q_map: BTreeMap<(State, Action), f64> = BTreeMap::new();

        mdp.transitions.keys().for_each(|state_action| {
            q_map.insert(*state_action, 0.0);
        });

        self.run_with_q_map(mdp, episodes, rng, &mut q_map);

        q_map
    }
    fn run_with_q_map(
        &self,
        mdp: &Mdp,
        episodes: usize,
        rng: &mut ChaCha20Rng,
        q_map: &mut BTreeMap<(State, Action), f64>,
    );
}

#[derive(Copy, Clone)]
pub enum Trace {
    Accumulating,
    Replacing,
    Dutch,
}

impl Trace {
    pub fn calculate(&self, value: f64, alpha: f64) -> f64 {
        match &self {
            Trace::Accumulating => value + 1.0,
            Trace::Replacing => 1.0,
            Trace::Dutch => (1.0 - alpha) * value + 1.0,
        }
    }
}
