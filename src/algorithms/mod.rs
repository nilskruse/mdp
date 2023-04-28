pub mod q_learning;
pub mod sarsa;

use std::collections::BTreeMap;

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use crate::{
    mdp::{Action, Mdp, State},
    policies::{epsilon_greedy_policy, greedy_policy, random_policy},
};

pub trait TDAlgorithm {
    fn run(
        &self,
        mdp: &Mdp,
        episodes: usize,
        rng: &mut ChaCha20Rng,
    ) -> BTreeMap<(State, Action), f64>;
    fn run_with_q_map(
        &self,
        mdp: &Mdp,
        episodes: usize,
        rng: &mut ChaCha20Rng,
        q_map: &mut BTreeMap<(State, Action), f64>,
    );
}

pub fn sarsa(
    mdp: &Mdp,
    alpha: f64,
    gamma: f64,
    epsilon: f64,
    initial: (State, Action),
    episodes: usize,
    max_steps: usize,
    rng: &mut ChaCha20Rng,
) -> BTreeMap<(State, Action), f64> {
    let mut q_map: BTreeMap<(State, Action), f64> = BTreeMap::new();

    mdp.transitions.keys().for_each(|state_action| {
        q_map.insert(*state_action, 0.0);
    });

    for _ in 1..=episodes {
        // let (mut current_state, mut current_action) = initial;
        let (mut current_state, mut current_action) = (
            mdp.initial_state,
            epsilon_greedy_policy(mdp, &q_map, mdp.initial_state, epsilon, rng).unwrap(),
        );
        let mut steps = 0;

        while !mdp.terminal_states.contains(&current_state) && steps < max_steps {
            let (next_state, reward) = mdp.perform_action((current_state, current_action), rng);

            let next_action = epsilon_greedy_policy(mdp, &q_map, next_state, epsilon, rng);
            if let Some(next_action) = next_action {
                // update q_map
                let next_q = *q_map.get(&(next_state, next_action)).unwrap_or(&0.0);
                let current_q = q_map.entry((current_state, current_action)).or_insert(0.0);
                *current_q = *current_q + alpha * (reward + gamma * next_q - *current_q);

                current_state = next_state;
                current_action = next_action;

                steps += 1;
            } else {
                break;
            }
        }
    }

    q_map
}

// pub fn q_learning(
//     mdp: &Mdp,
//     alpha: f64,
//     gamma: f64,
//     epsilon: f64,
//     initial: (State, Action),
//     episodes: usize,
//     max_steps: usize,
//     rng: &mut ChaCha20Rng,
// ) -> BTreeMap<(State, Action), f64> {
//     let mut q_map: BTreeMap<(State, Action), f64> = BTreeMap::new();

//     mdp.transitions.keys().for_each(|state_action| {
//         q_map.insert(*state_action, 0.0);
//     });

//     for _ in 1..=episodes {
//         let (mut current_state, _) = initial;
//         let mut steps = 0;

//         while !mdp.terminal_states.contains(&current_state) && steps < max_steps {
//             let selected_action = epsilon_greedy_policy(mdp, &q_map, current_state, epsilon, rng);
//             if let Some(selected_action) = selected_action {
//                 let (next_state, reward) =
//                     mdp.perform_action((current_state, selected_action), rng);

//                 // update q_map
//                 let best_action = greedy_policy(mdp, &q_map, next_state, rng);
//                 if let Some(best_action) = best_action {
//                     let best_q = *q_map
//                         .get(&(next_state, best_action))
//                         .expect("No qmap entry found");

//                     let current_q = q_map.entry((current_state, selected_action)).or_insert(0.0);
//                     *current_q = *current_q + alpha * (reward + gamma * best_q - *current_q);

//                     current_state = next_state;

//                     steps += 1;
//                 } else {
//                     break;
//                 }
//             } else {
//                 break;
//             }
//         }
//     }
//     q_map
// }

pub fn value_iteration(mdp: &Mdp, tolerance: f64, gamma: f64) -> BTreeMap<State, f64> {
    let mut value_map: BTreeMap<State, f64> = BTreeMap::new();
    let mut delta = f64::MAX;

    while delta > tolerance {
        delta = 0.0;

        for (state, _) in mdp.transitions.keys() {
            let old_value = *value_map.get(state).unwrap_or(&0.0);
            let new_value = best_action_value(mdp, *state, &value_map, gamma);

            value_map.insert(*state, new_value);
            delta = delta.max((old_value - new_value).abs());
        }
    }

    value_map
}

fn best_action_value(mdp: &Mdp, state: State, value_map: &BTreeMap<State, f64>, gamma: f64) -> f64 {
    mdp.transitions
        .iter()
        .filter_map(|((s, _), transitions)| {
            if *s == state {
                let expected_value: f64 = transitions
                    .iter()
                    .map(|(prob, next_state, reward)| {
                        prob * (reward + gamma * value_map.get(next_state).unwrap_or(&0.0))
                    })
                    .sum();
                Some(expected_value)
            } else {
                None
            }
        })
        .fold(f64::MIN, f64::max)
}
