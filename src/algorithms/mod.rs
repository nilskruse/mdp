pub mod dyna_q;
pub mod monte_carlo;
pub mod q_learning;
pub mod q_learning_beta;
pub mod q_learning_dynamic;
pub mod q_learning_lambda;
pub mod sarsa;
pub mod sarsa_lambda;
pub mod value_iteration;

use std::collections::BTreeMap;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

use crate::mdp::{
    GenericAction, GenericMdp, GenericState, IndexAction, IndexMdp, IndexState, MapMdp,
};

pub trait GenericStateActionAlgorithm {
    // default implementation
    fn run<M: GenericMdp<S, A>, S: GenericState, A: GenericAction, R: Rng + SeedableRng>(
        &self,
        mdp: &M,
        episodes: usize,
        rng: &mut R,
    ) -> BTreeMap<(S, A), f64> {
        let mut q_map: BTreeMap<(S, A), f64> = BTreeMap::new();

        mdp.get_all_state_actions().iter().for_each(|state_action| {
            q_map.insert(*state_action, 0.0);
        });

        self.run_with_q_map(mdp, episodes, rng, &mut q_map);

        q_map
    }

    fn run_with_q_map<
        M: GenericMdp<S, A>,
        S: GenericState,
        A: GenericAction,
        R: Rng + SeedableRng,
    >(
        &self,
        mdp: &M,
        episodes: usize,
        rng: &mut R,
        q_map: &mut BTreeMap<(S, A), f64>,
    );
}

pub trait GenericStateActionAlgorithmStateful {
    // default implementation
    fn run<M: GenericMdp<S, A>, S: GenericState, A: GenericAction, R: Rng + SeedableRng>(
        &mut self,
        mdp: &M,
        episodes: usize,
        rng: &mut R,
    ) -> BTreeMap<(S, A), f64> {
        let mut q_map: BTreeMap<(S, A), f64> = BTreeMap::new();

        mdp.get_all_state_actions().iter().for_each(|state_action| {
            q_map.insert(*state_action, 0.0);
        });

        self.run_with_q_map(mdp, episodes, rng, &mut q_map);

        q_map
    }

    fn run_with_q_map<
        M: GenericMdp<S, A>,
        S: GenericState,
        A: GenericAction,
        R: Rng + SeedableRng,
    >(
        &mut self,
        mdp: &M,
        episodes: usize,
        rng: &mut R,
        q_map: &mut BTreeMap<(S, A), f64>,
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
