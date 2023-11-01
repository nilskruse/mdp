use rand::Rng;

use crate::{mdp::GenericMdp, policies::greedy_policy_ma};
use std::collections::BTreeMap;

use crate::{
    mdp::{GenericAction, GenericState},
    policies::epsilon_greedy_policy,
};

use super::GenericStateActionAlgorithm;

pub struct QLearning {
    alpha: f64,
    epsilon: f64,
    max_steps: usize,
}

impl QLearning {
    pub fn new(alpha: f64, epsilon: f64, max_steps: usize) -> Self {
        QLearning {
            alpha,
            epsilon,
            max_steps,
        }
    }
}

impl GenericStateActionAlgorithm for QLearning {
    fn run_with_q_map<M: GenericMdp<S, A>, S: GenericState, A: GenericAction, R: Rng>(
        &self,
        mdp: &M,
        episodes: usize,
        rng: &mut R,
        q_map: &mut BTreeMap<(S, A), f64>,
    ) {
        for _ in 1..=episodes {
            let mut current_state = mdp.get_initial_state(rng);
            let mut steps = 0;

            while !mdp.is_terminal(current_state) && steps < self.max_steps {
                let Some(selected_action) =
                    epsilon_greedy_policy(mdp, q_map, current_state, self.epsilon, rng)
                else {
                    break;
                };
                let (next_state, reward) =
                    mdp.perform_action((current_state, selected_action), rng);

                // update q_map
                // let Some(best_action) = greedy_policy(mdp, q_map, next_state, rng) else {break};
                // let best_q = *q_map
                //     .get(&(next_state, best_action))
                //     .expect("No qmap entry found");

                // let current_q = q_map.entry((current_state, selected_action)).or_insert(0.0);
                // *current_q = *current_q
                //     + self.alpha * (reward + mdp.get_discount_factor() * best_q - *current_q);

                let step = self.step(
                    q_map,
                    &mdp.get_possible_actions(next_state),
                    current_state,
                    selected_action,
                    next_state,
                    reward,
                    mdp.get_discount_factor(),
                    rng,
                );
                // print_q_map(q_map);

                // break if step was not possible
                if !step {
                    println!("broke");
                    break;
                };

                current_state = next_state;

                steps += 1;
            }
        }
    }

    fn step<S: GenericState, A: GenericAction, R: Rng>(
        &self,
        q_map: &mut BTreeMap<(S, A), f64>,
        next_possible_actions: &[A],
        current_state: S,
        selected_action: A,
        next_state: S,
        reward: f64,
        discount_factor: f64,
        rng: &mut R,
    ) -> bool {
        // println!("state: {:?}, action: {:?}", current_state, selected_action);
        let Some(best_action) = greedy_policy_ma(next_possible_actions, q_map, next_state, rng)
        else {
            return false;
        };
        let best_q = *q_map
            .get(&(next_state, best_action))
            .expect("No qmap entry found");

        let current_q = q_map.entry((current_state, selected_action)).or_insert(0.0);
        *current_q = *current_q + self.alpha * (reward + discount_factor * best_q - *current_q);
        return true;
    }

    fn get_epsilon(&self) -> f64 {
        self.epsilon
    }
}
