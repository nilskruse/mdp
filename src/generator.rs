use std::{collections::HashMap, hash::Hash, iter};

use rand::{seq::SliceRandom, Rng};

use crate::mdp::{Action, Mdp, State, Transition};

pub fn generate_random_mdp(
    n_states: usize,
    n_actions: usize,
    min_actions: usize,
    max_actions: usize,
    n_terminal_states: usize,
    min_transitions: usize,
    max_transitions: usize,
    min_reward: f64,
    max_reward: f64,
) -> Mdp {
    let mut rng = rand::thread_rng();
    let mut states = vec![];
    let mut actions = vec![];

    // create states
    for i in 0..n_states {
        states.push(State(i));
    }
    for i in 0..n_actions {
        actions.push(Action(i));
    }

    let mut states_actions = vec![];

    for state in &states {
        let n_actions = rng.gen_range(min_actions..=max_actions);
        for i in 0..max_actions {
            states_actions.push((state, Action(i)));
        }
    }

    let transitions_map: HashMap<(State, Action), Vec<Transition>> =
        HashMap::from_iter(states_actions.iter().map(|(state, action)| {
            let n_transitions = rng.gen_range(min_transitions..=max_transitions);
            let probabilities = random_probs(n_transitions);
            let mut outcomes = vec![];

            for i in 0..n_transitions {
                let reward = rng.gen_range(min_reward..=max_reward);
                let next_state = states.choose(&mut rng).unwrap();
                outcomes.push((probabilities[i], *next_state, reward));
            }
            ((**state, *action), outcomes)
        }));

    Mdp {
        transitions: transitions_map,
        terminal_states: vec![State(0)],
        initial_state: State(0),
    }
}

fn random_probs(n: usize) -> Vec<f64> {
    let mut rng = rand::thread_rng();

    // Generate n-1 random floats between 0 and 1
    let mut random_numbers: Vec<f64> = iter::repeat_with(|| rng.gen_range(0.0..1.0))
        .take(n - 1)
        .collect();

    // Sort the numbers and add 0 and 1
    random_numbers.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    random_numbers.insert(0, 0.0);
    random_numbers.push(1.0);

    // Calculate the differences between consecutive numbers
    random_numbers
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .collect()
}
