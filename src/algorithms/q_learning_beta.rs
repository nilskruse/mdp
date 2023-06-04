use crate::mdp::GenericMdp;
use std::collections::BTreeMap;

use crate::{
    mdp::{IndexAction, IndexState},
    policies::{epsilon_greedy_policy, greedy_policy},
};

use super::StateActionAlgorithm;

pub struct QLearningBeta {
    alpha: f64,
    gamma: f64,
    epsilon: f64,
    max_steps: usize,
}

impl QLearningBeta {
    pub fn new(alpha: f64, gamma: f64, epsilon: f64, max_steps: usize) -> Self {
        QLearningBeta {
            alpha,
            gamma,
            epsilon,
            max_steps,
        }
    }
}

impl StateActionAlgorithm for QLearningBeta {
    fn run_with_q_map(
        &self,
        mdp: &crate::mdp::IndexMdp,
        episodes: usize,
        rng: &mut rand_chacha::ChaCha20Rng,
        q_map: &mut std::collections::BTreeMap<
            (crate::mdp::IndexState, crate::mdp::IndexAction),
            f64,
        >,
    ) {
        for episode in 1..=episodes {
            let mut current_state = mdp.initial_state;
            let mut steps = 0;

            while !mdp.terminal_states.contains(&current_state) && steps < self.max_steps {
                let Some(mut selected_action) = epsilon_greedy_policy(mdp, q_map, current_state, self.epsilon, rng) else {break};
                let (mut next_state, mut reward) =
                    mdp.perform_action((current_state, selected_action), rng);

                if episode == 2 && steps == 0 {
                    println!("Rigging first action selection!!!");
                    selected_action = IndexAction(0);
                    next_state = IndexState(2);
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
                    + (self.alpha * (reward + self.gamma * best_q - *current_q)) * (1.0 - beta);

                current_state = next_state;

                steps += 1;
            }
        }
    }
}
