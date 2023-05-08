use std::collections::BTreeMap;

use crate::{
    mdp::{Action, State},
    policies::epsilon_greedy_policy,
};

use super::StateActionAlgorithm;

pub struct Sarsa {
    alpha: f64,
    gamma: f64,
    epsilon: f64,
    max_steps: usize,
}

impl Sarsa {
    pub fn new(alpha: f64, gamma: f64, epsilon: f64, max_steps: usize) -> Self {
        Sarsa {
            alpha,
            gamma,
            epsilon,
            max_steps,
        }
    }
}

impl StateActionAlgorithm for Sarsa {
    fn run_with_q_map(
        &self,
        mdp: &crate::mdp::Mdp,
        episodes: usize,
        rng: &mut rand_chacha::ChaCha20Rng,
        q_map: &mut std::collections::BTreeMap<(crate::mdp::State, crate::mdp::Action), f64>,
    ) {
        for _ in 1..=episodes {
            let (mut current_state, mut current_action) = (
                mdp.initial_state,
                epsilon_greedy_policy(mdp, q_map, mdp.initial_state, self.epsilon, rng).unwrap(),
            );
            let mut steps = 0;

            while !mdp.terminal_states.contains(&current_state) && steps < self.max_steps {
                let (next_state, reward) = mdp.perform_action((current_state, current_action), rng);

                let next_action = epsilon_greedy_policy(mdp, q_map, next_state, self.epsilon, rng);
                if let Some(next_action) = next_action {
                    // update q_map
                    let next_q = *q_map.get(&(next_state, next_action)).unwrap_or(&0.0);
                    let current_q = q_map.entry((current_state, current_action)).or_insert(0.0);
                    *current_q =
                        *current_q + self.alpha * (reward + self.gamma * next_q - *current_q);

                    current_state = next_state;
                    current_action = next_action;

                    steps += 1;
                } else {
                    break;
                }
            }
        }
    }
}
