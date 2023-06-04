use std::collections::BTreeMap;

use rand::{Rng, SeedableRng};

use crate::{
    mdp::{GenericAction, GenericMdp, GenericState, IndexAction, IndexState},
    policies::epsilon_greedy_policy,
};

use super::{GenericStateActionAlgorithm, Trace};

pub struct SarsaLambda {
    alpha: f64,
    epsilon: f64,
    lambda: f64,
    max_steps: usize,
    trace: Trace,
}

impl SarsaLambda {
    pub fn new(alpha: f64, epsilon: f64, lambda: f64, max_steps: usize, trace: Trace) -> Self {
        SarsaLambda {
            alpha,
            epsilon,
            lambda,
            max_steps,
            trace,
        }
    }
}

impl GenericStateActionAlgorithm for SarsaLambda {
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
    ) {
        for _ in 0..episodes {
            let mut e_map: BTreeMap<(S, A), f64> = BTreeMap::new();

            let (mut current_state, mut current_action) = (
                mdp.get_initial_state(),
                epsilon_greedy_policy(mdp, q_map, mdp.get_initial_state(), self.epsilon, rng)
                    .unwrap(),
            );
            let mut steps = 0;

            while !mdp.is_terminal(current_state) && steps < self.max_steps {
                let (next_state, reward) = mdp.perform_action((current_state, current_action), rng);

                // select action epsilon greedy and break if no action is possible (episode ends)
                let Some(next_action) = epsilon_greedy_policy(mdp, q_map, next_state, self.epsilon, rng) else {break};
                // update q_map
                let next_q = *q_map.get(&(next_state, next_action)).unwrap();
                let current_q = *q_map.get(&(current_state, current_action)).unwrap();

                let delta = reward + mdp.get_discount_factor() * next_q - current_q;

                e_map
                    .entry((current_state, current_action))
                    .and_modify(|entry| *entry = self.trace.calculate(*entry, self.alpha));

                // update q and e for all (state, action) pairs
                mdp.get_all_state_actions_iter().for_each(|key| {
                    let e_entry = e_map.entry(*key).or_default();
                    q_map.entry(*key).and_modify(|q_entry| {
                        *q_entry += self.alpha * delta * *e_entry;
                    });
                    *e_entry *= mdp.get_discount_factor() * self.lambda;
                });

                current_state = next_state;
                current_action = next_action;

                steps += 1;
            }
        }
    }
}
