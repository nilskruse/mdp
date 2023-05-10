use std::collections::BTreeMap;

use crate::{
    mdp::{Action, State},
    policies::{epsilon_greedy_policy, greedy_policy},
};

use super::{StateActionAlgorithm, Trace};

pub struct QLearningLamda {
    alpha: f64,
    gamma: f64,
    epsilon: f64,
    lambda: f64,
    max_steps: usize,
    trace: Trace,
}

impl QLearningLamda {
    pub fn new(
        alpha: f64,
        gamma: f64,
        epsilon: f64,
        lambda: f64,
        max_steps: usize,
        trace: Trace,
    ) -> Self {
        QLearningLamda {
            alpha,
            gamma,
            epsilon,
            lambda,
            max_steps,
            trace,
        }
    }
}

impl StateActionAlgorithm for QLearningLamda {
    fn run_with_q_map(
        &self,
        mdp: &crate::mdp::Mdp,
        episodes: usize,
        rng: &mut rand_chacha::ChaCha20Rng,
        q_map: &mut std::collections::BTreeMap<(crate::mdp::State, crate::mdp::Action), f64>,
    ) {
        for _ in 0..episodes {
            let mut e_map: BTreeMap<(State, Action), f64> = BTreeMap::new();

            let (mut current_state, mut current_action) = (
                mdp.initial_state,
                epsilon_greedy_policy(mdp, q_map, mdp.initial_state, self.epsilon, rng).unwrap(),
            );
            let mut steps = 0;

            while !mdp.terminal_states.contains(&current_state) && steps < self.max_steps {
                let (next_state, reward) = mdp.perform_action((current_state, current_action), rng);

                if let Some(next_action) =
                    epsilon_greedy_policy(mdp, q_map, next_state, self.epsilon, rng)
                {
                    if let Some(best_action) = greedy_policy(mdp, q_map, next_state, rng) {
                        // update q_map
                        let next_q = *q_map.get(&(next_state, best_action)).unwrap();
                        let current_q = *q_map.get(&(current_state, current_action)).unwrap();

                        let delta = reward + self.gamma * next_q - current_q;

                        e_map
                            .entry((current_state, current_action))
                            .and_modify(|entry| *entry = self.trace.calculate(*entry, self.alpha));

                        // update q and e for all (state, action) pairs
                        mdp.transitions.keys().for_each(|key| {
                            let e_entry = e_map.entry(*key).or_default();

                            q_map.entry(*key).and_modify(|q_entry| {
                                *q_entry += self.alpha * delta * *e_entry;
                            });
                            if next_action == best_action {
                                *e_entry *= self.gamma * self.lambda;
                            } else {
                                *e_entry = 0.0;
                            }
                        });

                        current_state = next_state;
                        current_action = next_action;

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
