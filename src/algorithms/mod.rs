pub mod dyna_q;
pub mod monte_carlo;
pub mod q_learning;
pub mod q_learning_beta;
pub mod q_learning_dynamic;
pub mod q_learning_lambda;
pub mod sarsa;
pub mod sarsa_lambda;
pub mod value_iteration;

use std::{collections::BTreeMap, fmt::Display};

use rand::Rng;

use crate::mdp::{GenericAction, GenericMdp, GenericState};

pub trait GenericStateActionAlgorithm {
    // default implementation
    fn run<M: GenericMdp<S, A>, S: GenericState, A: GenericAction, R: Rng>(
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

    fn run_with_q_map<M: GenericMdp<S, A>, S: GenericState, A: GenericAction, R: Rng>(
        &self,
        mdp: &M,
        episodes: usize,
        rng: &mut R,
        q_map: &mut BTreeMap<(S, A), f64>,
    );

    fn get_epsilon(&self) -> f64;

    #[allow(unused_variables)]
    fn step<S: GenericState, A: GenericAction, R: Rng>(
        &self,
        q_map: &mut BTreeMap<(S, A), f64>,
        possible_actions: &[A],
        current_state: S,
        selected_action: A,
        next_state: S,
        reward: f64,
        discount_factor: f64,
        rng: &mut R,
    ) -> bool {
        unimplemented!()
    }
}

pub trait GenericStateActionAlgorithmStateful {
    // default implementation
    fn run<M: GenericMdp<S, A>, S: GenericState, A: GenericAction, R: Rng>(
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

    fn run_with_q_map<M: GenericMdp<S, A>, S: GenericState, A: GenericAction, R: Rng>(
        &mut self,
        mdp: &M,
        episodes: usize,
        rng: &mut R,
        q_map: &mut BTreeMap<(S, A), f64>,
    );
}
#[derive(Copy, Clone, Debug)]
pub enum Trace {
    Accumulating,
    Replacing,
    Dutch,
}

impl Display for Trace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
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
