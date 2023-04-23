use std::collections::HashMap;

use crate::{
    mdp::{Action, Mdp, State},
    policies::{epsilon_greedy_policy, greedy_policy},
};

pub fn sarsa(
    mdp: &Mdp,
    alpha: f64,
    gamma: f64,
    initial: (State, Action),
    episodes: u64,
    max_steps: u64,
) -> HashMap<(State, Action), f64> {
    let mut q_map: HashMap<(State, Action), f64> = HashMap::new();

    for episode in 1..=episodes {
        let (mut current_state, mut current_action) = initial;
        let mut steps = 0;

        while !mdp.terminal_states.contains(&current_state) && steps < max_steps {
            let (next_state, reward) = mdp.perform_action((current_state, current_action));

            let next_action = epsilon_greedy_policy(mdp, &q_map, next_state, 0.1);

            // update q_map
            let next_q = *q_map.get(&(next_state, next_action)).unwrap_or(&0.0);
            let current_q = q_map.entry((current_state, current_action)).or_insert(0.0);
            *current_q = *current_q + alpha * (reward + gamma * next_q - *current_q);

            current_state = next_state;
            current_action = next_action;

            steps += 1;
        }
        // println!("Terminated episode {} after {} steps!", episode, steps);
    }

    q_map
}

pub fn q_learning(
    mdp: &Mdp,
    alpha: f64,
    gamma: f64,
    initial: (State, Action),
    episodes: u64,
    max_steps: u64,
) -> HashMap<(State, Action), f64> {
    let mut q_map: HashMap<(State, Action), f64> = HashMap::new();

    for _ in 1..=episodes {
        let (mut current_state, _) = initial;
        let mut steps = 0;

        while !mdp.terminal_states.contains(&current_state) && steps < max_steps {
            let selected_action = epsilon_greedy_policy(mdp, &q_map, current_state, 0.1);
            let (next_state, reward) = mdp.perform_action((current_state, selected_action));

            // update q_map
            let best_action = greedy_policy(mdp, &q_map, current_state);
            let best_q = *q_map.get(&(next_state, best_action)).unwrap_or(&0.0);

            let current_q = q_map.entry((current_state, selected_action)).or_insert(0.0);
            *current_q = *current_q + alpha * (reward + gamma * best_q - *current_q);

            current_state = next_state;

            steps += 1;
        }
        // println!("Terminated episode {} after {} steps!", episode, steps);
    }

    q_map
}

pub fn value_iteration(mdp: &Mdp, tolerance: f64, gamma: f64) -> HashMap<State, f64> {
    let mut value_map: HashMap<State, f64> = HashMap::new();
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

fn best_action_value(mdp: &Mdp, state: State, value_map: &HashMap<State, f64>, gamma: f64) -> f64 {
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
