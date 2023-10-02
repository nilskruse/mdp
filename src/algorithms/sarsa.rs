use std::collections::BTreeMap;

use rand::Rng;

use crate::{
    mdp::{GenericAction, GenericMdp, GenericState},
    policies::epsilon_greedy_policy,
};

use super::GenericStateActionAlgorithm;

pub struct Sarsa {
    alpha: f64,
    epsilon: f64,
    max_steps: usize,
}

impl Sarsa {
    pub fn new(alpha: f64, epsilon: f64, max_steps: usize) -> Self {
        Sarsa {
            alpha,
            epsilon,
            max_steps,
        }
    }
}

impl GenericStateActionAlgorithm for Sarsa {
    fn run_with_q_map<M: GenericMdp<S, A>, S: GenericState, A: GenericAction, R: Rng>(
        &self,
        mdp: &M,
        episodes: usize,
        rng: &mut R,
        q_map: &mut BTreeMap<(S, A), f64>,
    ) {
        for _ in 1..=episodes {
            let (mut current_state, mut current_action) = (
                mdp.get_initial_state(rng),
                epsilon_greedy_policy(mdp, q_map, mdp.get_initial_state(rng), self.epsilon, rng)
                    .unwrap(),
            );
            let mut steps = 0;

            while !mdp.is_terminal(current_state) && steps < self.max_steps {
                let (next_state, reward) = mdp.perform_action((current_state, current_action), rng);

                let next_action = epsilon_greedy_policy(mdp, q_map, next_state, self.epsilon, rng);
                let Some(next_action) = next_action else {
                    break;
                };

                // update q_map
                let next_q = *q_map.get(&(next_state, next_action)).unwrap_or(&0.0);
                let current_q = q_map.entry((current_state, current_action)).or_insert(0.0);
                *current_q = *current_q
                    + self.alpha * (reward + mdp.get_discount_factor() * next_q - *current_q);

                current_state = next_state;
                current_action = next_action;

                steps += 1;
            }
        }
    }

    fn get_epsilon(&self) -> f64 {
        self.epsilon
    }
}
