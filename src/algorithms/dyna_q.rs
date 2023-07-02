use std::collections::BTreeMap;

use rand::{seq::IteratorRandom, Rng, SeedableRng};

use crate::{
    mdp::{GenericAction, GenericMdp, GenericState, Reward},
    policies::{epsilon_greedy_policy, greedy_policy},
};

pub trait Dyna {
    fn run<M: GenericMdp<S, A>, S: GenericState, A: GenericAction, R: Rng + SeedableRng>(
        &mut self,
        mdp: &M,
        episodes: usize,
        rng: &mut R,
    ) -> (BTreeMap<(S, A), f64>, BTreeMap<(S, A), (f64, S)>) {
        let mut q_map = BTreeMap::new();
        let mut model = BTreeMap::new();

        mdp.get_all_state_actions().iter().for_each(|state_action| {
            q_map.insert(*state_action, 0.0);
        });

        self.run_with_q_map(mdp, episodes, rng, &mut q_map, &mut model);

        (q_map, model)
    }

    fn run_with_q_map<
        M: GenericMdp<S, A>,
        S: GenericState,
        A: GenericAction,
        R: Rng + SeedableRng,
    >(
        &mut self,
        mdp: &M,
        episodes: usize,
        rng: &mut R,
        q_map: &mut BTreeMap<(S, A), f64>,
        model: &mut BTreeMap<(S, A), (f64, S)>,
    );
}

pub struct DynaQ {
    alpha: f64,
    epsilon: f64,
    k: usize,
    max_steps: usize,
}

impl DynaQ {
    pub fn new(alpha: f64, epsilon: f64, k: usize, max_steps: usize) -> Self {
        Self {
            alpha,
            epsilon,
            k,
            max_steps,
        }
    }
}

impl Dyna for DynaQ {
    fn run_with_q_map<
        M: GenericMdp<S, A>,
        S: GenericState,
        A: GenericAction,
        R: Rng + SeedableRng,
    >(
        &mut self,
        mdp: &M,
        episodes: usize,
        rng: &mut R,
        q_map: &mut BTreeMap<(S, A), f64>,
        model: &mut BTreeMap<(S, A), (f64, S)>,
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

                // update model
                model.insert((current_state, selected_action), (reward, next_state));

                // run q on model
                for _ in 0..self.k {
                    let (key, (reward, next_state)) =
                        model.iter().choose(rng).expect("Model should not be empty");

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
        // for _ in 1..=episodes {
        //     let mut current_state = mdp.get_initial_state(rng);
        //     let mut steps = 0;

        //     while !mdp.is_terminal(current_state) && steps < self.max_steps {
        //         let Some(selected_action) = epsilon_greedy_policy(mdp, q_map, current_state, self.epsilon, rng) else {break};
        //         let (next_state, reward) =
        //             mdp.perform_action((current_state, selected_action), rng);

        //         // update q_map
        //         let Some(selected_action) = greedy_policy(mdp, q_map, next_state, rng) else {break};
        //         let best_q = *q_map
        //             .get(&(next_state, selected_action))
        //             .expect("No qmap entry found");

        //         let current_q = q_map.entry((current_state, selected_action)).or_insert(0.0);
        //         *current_q = *current_q
        //             + self.alpha * (reward + mdp.get_discount_factor() * best_q - *current_q);

        //         println!("current_q: {:?} ", *current_q);
        //         // update model TODO: deal with non-determinism
        //         // model.insert((current_state, selected_action), (reward, next_state));
        //         // // planning
        //         // for _ in 0..self.k {
        //         //     let (key, (reward, next_state)) =
        //         //         model.iter().choose(rng).expect("Model should not be empty");

        //         //     let Some(selected_action) = greedy_policy(mdp, q_map, *next_state, rng) else {
        //         //         // no action possible
        //         //         continue;
        //         //     };
        //         //     let best_q = *q_map
        //         //         .get(&(*next_state, selected_action))
        //         //         .expect("No qmap entry found");
        //         //     let current_q = q_map.entry(*key).or_insert(0.0);
        //         //     *current_q = *current_q
        //         //         + self.alpha * (reward + mdp.get_discount_factor() * best_q - *current_q);
        //         // }

        //         current_state = next_state;

        //         steps += 1;
        //     }
        // }
    }
}
