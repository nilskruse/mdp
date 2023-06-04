use rand::{Rng, SeedableRng};

use crate::{
    experiments::non_contractive::RiggedStateActionAlgorithm,
    mdp::{GenericAction, GenericMdp, GenericState},
};
use std::collections::BTreeMap;

use crate::{
    mdp::{IndexAction, IndexState},
    policies::{epsilon_greedy_policy, greedy_policy},
};

use super::GenericStateActionAlgorithm;

pub struct QLearningBeta {
    alpha: f64,
    epsilon: f64,
    max_steps: usize,
}

impl QLearningBeta {
    pub fn new(alpha: f64, epsilon: f64, max_steps: usize) -> Self {
        QLearningBeta {
            alpha,
            epsilon,
            max_steps,
        }
    }
}

impl RiggedStateActionAlgorithm for QLearningBeta {
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
        rig: (S, A),
    ) {
        for episode in 1..=episodes {
            let mut current_state = mdp.get_initial_state();
            let mut steps = 0;

            while !mdp.is_terminal(current_state) && steps < self.max_steps {
                let Some(mut selected_action) = epsilon_greedy_policy(mdp, q_map, current_state, self.epsilon, rng) else {break};
                let (mut next_state, mut reward) =
                    mdp.perform_action((current_state, selected_action), rng);

                if episode == 2 && steps == 0 {
                    println!("Rigging first action selection!!!");
                    selected_action = rig.1;
                    next_state = rig.0;
                    reward = 1000.0;
                }

                // update q_map
                let Some(best_action) = greedy_policy(mdp, q_map, next_state, rng) else {break};
                let best_q = *q_map
                    .get(&(next_state, best_action))
                    .expect("No qmap entry found");

                let current_q = q_map.entry((current_state, selected_action)).or_insert(0.0);
                let beta = 1.0 / episode as f64;
                *current_q = *current_q
                    + (self.alpha * (reward + mdp.get_discount_factor() * best_q - *current_q))
                        * (1.0 - beta);

                current_state = next_state;

                steps += 1;
            }
        }
    }
}
