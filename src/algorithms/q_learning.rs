use crate::mdp::GenericMdp;
use std::collections::BTreeMap;

use crate::{
    mdp::{Action, GenericAction, GenericState, MapMdp, State},
    policies::{
        epsilon_greedy_policy, epsilon_greedy_policy_generic, greedy_policy, greedy_policy_generic,
    },
};

use super::{GenericStateActionAlgorithm, StateActionAlgorithm};

pub struct QLearning {
    alpha: f64,
    gamma: f64,
    epsilon: f64,
    max_steps: usize,
}

impl QLearning {
    pub fn new(alpha: f64, gamma: f64, epsilon: f64, max_steps: usize) -> Self {
        QLearning {
            alpha,
            gamma,
            epsilon,
            max_steps,
        }
    }
}

impl StateActionAlgorithm for QLearning {
    fn run_with_q_map(
        &self,
        mdp: &crate::mdp::Mdp,
        episodes: usize,
        rng: &mut rand_chacha::ChaCha20Rng,
        q_map: &mut std::collections::BTreeMap<(crate::mdp::State, crate::mdp::Action), f64>,
    ) {
        for _ in 1..=episodes {
            let mut current_state = mdp.initial_state;
            let mut steps = 0;

            while !mdp.terminal_states.contains(&current_state) && steps < self.max_steps {
                let Some(selected_action) = epsilon_greedy_policy(mdp, q_map, current_state, self.epsilon, rng) else {break};
                let (next_state, reward) =
                    mdp.perform_action((current_state, selected_action), rng);

                // update q_map
                let Some(best_action) = greedy_policy(mdp, q_map, next_state, rng) else {break};
                let best_q = *q_map
                    .get(&(next_state, best_action))
                    .expect("No qmap entry found");

                let current_q = q_map.entry((current_state, selected_action)).or_insert(0.0);
                *current_q = *current_q + self.alpha * (reward + self.gamma * best_q - *current_q);

                current_state = next_state;

                steps += 1;
            }
        }
    }
}

pub struct QLearningGeneric {
    alpha: f64,
    gamma: f64,
    epsilon: f64,
    max_steps: usize,
}

impl QLearningGeneric {
    pub fn new(alpha: f64, gamma: f64, epsilon: f64, max_steps: usize) -> Self {
        QLearningGeneric {
            alpha,
            gamma,
            epsilon,
            max_steps,
        }
    }
}

impl GenericStateActionAlgorithm for QLearningGeneric {
    fn run_with_q_map<M: GenericMdp<S, A>, S: GenericState, A: GenericAction>(
        &self,
        mdp: &M,
        episodes: usize,
        rng: &mut rand_chacha::ChaCha20Rng,
        q_map: &mut BTreeMap<(S, A), f64>,
    ) {
        for _ in 1..=episodes {
            let mut current_state = mdp.get_initial_sate();
            let mut steps = 0;

            while !mdp.is_terminal(current_state) && steps < self.max_steps {
                let Some(selected_action) = epsilon_greedy_policy_generic(mdp, q_map, current_state, self.epsilon, rng) else {break};
                let (next_state, reward) =
                    mdp.perform_action((current_state, selected_action), rng);

                // update q_map
                let Some(best_action) = greedy_policy_generic(mdp, q_map, next_state, rng) else {break};
                let best_q = *q_map
                    .get(&(next_state, best_action))
                    .expect("No qmap entry found");

                let current_q = q_map.entry((current_state, selected_action)).or_insert(0.0);
                *current_q = *current_q + self.alpha * (reward + self.gamma * best_q - *current_q);

                current_state = next_state;

                steps += 1;
            }
        }
    }
}
