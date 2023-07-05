use std::collections::BTreeMap;

use eframe::glow::DRAW_INDIRECT_BUFFER_BINDING;
use rand::{seq::IteratorRandom, Rng, SeedableRng};
use std::collections::Bound::{Included, Unbounded};

use crate::{
    mdp::{GenericAction, GenericMdp, GenericState, Reward},
    policies::{epsilon_greedy_policy, greedy_policy},
};

pub trait Dyna<S: GenericState, A: GenericAction> {
    fn run<M: GenericMdp<S, A>, R: Rng + SeedableRng>(
        &mut self,
        mdp: &M,
        episodes: usize,
        rng: &mut R,
    ) -> BTreeMap<(S, A), f64> {
        let mut q_map = BTreeMap::new();

        mdp.get_all_state_actions().iter().for_each(|state_action| {
            q_map.insert(*state_action, 0.0);
        });

        self.run_with_q_map(mdp, episodes, rng, &mut q_map);

        q_map
    }

    fn run_with_q_map<M: GenericMdp<S, A>, R: Rng + SeedableRng>(
        &mut self,
        mdp: &M,
        episodes: usize,
        rng: &mut R,
        q_map: &mut BTreeMap<(S, A), f64>,
    );
}

pub struct DynaQ<S: GenericState, A: GenericAction> {
    alpha: f64,
    epsilon: f64,
    k: usize,
    max_steps: usize,
    model: BTreeMap<(S, A), (f64, S)>,
    t_table: BTreeMap<(S, A), BTreeMap<S, usize>>,
    deterministic: bool,
}

impl<S: GenericState, A: GenericAction> DynaQ<S, A> {
    pub fn new<M: GenericMdp<S, A>>(
        alpha: f64,
        epsilon: f64,
        k: usize,
        max_steps: usize,
        deterministic: bool,
        _mdp: &M, // used for type inference
    ) -> Self {
        Self {
            alpha,
            epsilon,
            k,
            max_steps,
            model: BTreeMap::new(),
            t_table: BTreeMap::new(),
            deterministic,
        }
    }
}

impl<S: GenericState, A: GenericAction> Dyna<S, A> for DynaQ<S, A> {
    fn run_with_q_map<M: GenericMdp<S, A>, R: Rng + SeedableRng>(
        &mut self,
        mdp: &M,
        episodes: usize,
        rng: &mut R,
        q_map: &mut BTreeMap<(S, A), f64>,
    ) {
        for _ in 1..=episodes {
            let mut current_state = mdp.get_initial_state(rng);
            let mut steps = 0;

            while !mdp.is_terminal(current_state) && steps < self.max_steps {
                let Some(selected_action) = epsilon_greedy_policy(mdp, q_map, current_state, self.epsilon, rng) else {break};
                let (next_state, reward) =
                    mdp.perform_action((current_state, selected_action), rng);

                // update q_map
                let Some(best_action) = greedy_policy(mdp, q_map, next_state, rng) else {break};
                let best_q = *q_map
                    .get(&(next_state, best_action))
                    .expect("No qmap entry found");

                let current_q = q_map.entry((current_state, selected_action)).or_insert(0.0);
                *current_q = *current_q
                    + self.alpha * (reward + mdp.get_discount_factor() * best_q - *current_q);

                // determine reward value used for updating model
                let model_reward = if self.deterministic {
                    reward
                } else {
                    let mut state_count = 0;

                    // update t_table
                    self.t_table
                        .entry((current_state, selected_action))
                        .and_modify(|map| {
                            map.entry(next_state)
                                .and_modify(|count| {
                                    *count += 1;
                                    state_count = *count;
                                })
                                .or_insert(1);
                        })
                        .or_insert(BTreeMap::from([(next_state, 1)]));

                    state_count = *self
                        .t_table
                        .get(&(current_state, selected_action))
                        .unwrap()
                        .get(&next_state)
                        .unwrap();

                    let sum: usize = self
                        .t_table
                        .get(&(current_state, selected_action))
                        .unwrap()
                        .values()
                        .copied()
                        .sum();

                    reward * (state_count as f64 / sum as f64)
                };

                // update model
                self.model
                    .insert((current_state, selected_action), (model_reward, next_state));

                // run q on model
                for _ in 0..self.k {
                    let (key, (reward, next_state)) = self
                        .model
                        .iter()
                        .choose(rng)
                        .expect("Model should not be empty");

                    let Some(selected_action) = greedy_policy(mdp, q_map, *next_state, rng) else {
                        // no action possible
                        continue;
                    };

                    let best_q = *q_map
                        .get(&(*next_state, selected_action))
                        .expect("No qmap entry found");

                    let current_q = q_map.entry(*key).or_insert(0.0);

                    *current_q = *current_q
                        + self.alpha * (reward + mdp.get_discount_factor() * best_q - *current_q);
                }
                current_state = next_state;

                steps += 1;
            }
        }
    }
}
