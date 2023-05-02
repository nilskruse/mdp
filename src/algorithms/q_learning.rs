use std::collections::BTreeMap;

use crate::{
    mdp::{Action, State},
    policies::{epsilon_greedy_policy, greedy_policy},
};

use super::TDAlgorithm;

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
impl TDAlgorithm for QLearning {
    fn run(
        &self,
        mdp: &crate::mdp::Mdp,
        episodes: usize,
        rng: &mut rand_chacha::ChaCha20Rng,
    ) -> std::collections::BTreeMap<(crate::mdp::State, crate::mdp::Action), f64> {
        let mut q_map: BTreeMap<(State, Action), f64> = BTreeMap::new();

        mdp.transitions.keys().for_each(|state_action| {
            q_map.insert(*state_action, 0.0);
        });

        self.run_with_q_map(mdp, episodes, rng, &mut q_map);

        q_map
    }

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
                let selected_action =
                    epsilon_greedy_policy(mdp, q_map, current_state, self.epsilon, rng);
                if let Some(selected_action) = selected_action {
                    let (next_state, reward) =
                        mdp.perform_action((current_state, selected_action), rng);

                    // update q_map
                    let best_action = greedy_policy(mdp, q_map, next_state, rng);
                    if let Some(best_action) = best_action {
                        let best_q = *q_map
                            .get(&(next_state, best_action))
                            .expect("No qmap entry found");

                        let current_q =
                            q_map.entry((current_state, selected_action)).or_insert(0.0);
                        *current_q =
                            *current_q + self.alpha * (reward + self.gamma * best_q - *current_q);

                        current_state = next_state;

                        steps += 1;
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
        }
    }
}