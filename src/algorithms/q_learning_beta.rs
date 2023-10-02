use rand::Rng;

use crate::mdp::{GenericAction, GenericMdp, GenericState};
use std::collections::BTreeMap;

use crate::policies::{epsilon_greedy_policy, greedy_policy};

use super::GenericStateActionAlgorithmStateful;

pub struct QLearningBeta {
    alpha: f64,
    epsilon: f64,
    max_steps: usize,
    rate: usize,
    beta_denom: f64,
    total_episodes: usize,
}

impl QLearningBeta {
    pub fn new(alpha: f64, epsilon: f64, max_steps: usize, rate: usize) -> Self {
        QLearningBeta {
            alpha,
            epsilon,
            max_steps,
            rate,
            beta_denom: 0.0_f64,
            total_episodes: 0,
        }
    }
}

impl GenericStateActionAlgorithmStateful for QLearningBeta {
    fn run_with_q_map<M: GenericMdp<S, A>, S: GenericState, A: GenericAction, R: Rng>(
        &mut self,
        mdp: &M,
        episodes: usize,
        rng: &mut R,
        q_map: &mut BTreeMap<(S, A), f64>,
    ) {
        for _ in 1..=episodes {
            let mut current_state = mdp.get_initial_state(rng);
            let mut steps = 0;

            if self.total_episodes % self.rate == 0 {
                self.beta_denom += 1.0;
                // let beta = 1.0 / (self.beta_denom + 1.0);
                // println!("beta={beta}");
            }

            let beta = 1.0 / (self.beta_denom + 1.0);

            while !mdp.is_terminal(current_state) && steps < self.max_steps {
                let Some(selected_action) =
                    epsilon_greedy_policy(mdp, q_map, current_state, self.epsilon, rng)
                else {
                    break;
                };
                let (next_state, reward) =
                    mdp.perform_action((current_state, selected_action), rng);

                // update q_map
                let Some(best_action) = greedy_policy(mdp, q_map, next_state, rng) else {
                    break;
                };
                let best_q = *q_map
                    .get(&(next_state, best_action))
                    .expect("No qmap entry found");

                let current_q = q_map.entry((current_state, selected_action)).or_insert(0.0);
                *current_q = (*current_q
                    + (self.alpha * (reward + mdp.get_discount_factor() * best_q - *current_q)))
                    * (1.0 - beta);

                current_state = next_state;

                steps += 1;
            }
            self.total_episodes += 1;
        }
    }
}
