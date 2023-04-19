mod mdp;
use rand::Rng;

use crate::mdp::*;
use std::collections::HashMap;

fn main() {
    let mdp = Mdp::new_test_mdp();
    let value_map = value_iteration(&mdp, 0.01, 0.0);

    mdp.perform_action((State(0u8), Action::A));

    for (state, value) in value_map.iter() {
        println!("State {:?} has value: {:.4}", state, value);
    }

    let q_map = sarsa(&mdp, 0.5, 0.5, (State(0), Action::A));
    println!("Q: {:?}", q_map);
}

fn sarsa(
    mdp: &Mdp,
    alpha: f64,
    gamma: f64,
    initial: (State, Action),
) -> HashMap<(State, Action), f64> {
    let mut q_map: HashMap<(State, Action), f64> = HashMap::new();
    let (mut current_state, mut current_action) = initial;

    // println!("Next state: {:?} and reward: {:?}", next_state, reward);
    // println!("Possible actions: {:?}", possible_actions);
    // println!("Selected next_action: {:?}", selected_action);

    for _ in 0..100000 {
        let (mut next_state, mut reward) = mdp.perform_action((current_state, current_action));

        // TODO: actual policy
        let next_action = random_policy(&mdp, next_state);

        println!(
            "Quintuple: ({:?},{:?},{:?},{:?},{:?})",
            current_state, current_action, reward, next_state, next_action
        );

        // update q_map
        let next_q = q_map.get(&(next_state, next_action)).unwrap_or(&0.0).clone();
        let current_q = q_map.entry((current_state, current_action)).or_insert(0.0);
        *current_q = *current_q + alpha * (reward + gamma * next_q - *current_q);

        current_state = next_state;
        current_action = next_action;
    }

    q_map
}

fn random_policy(mdp: &Mdp, current_state: State) -> Action {
    let mut rng = rand::thread_rng();
    let possible_states = mdp.get_possible_actions(current_state);
    let selected_index = rng.gen_range(0..possible_states.len());

    possible_states[selected_index]
}

fn value_iteration(mdp: &Mdp, tolerance: f64, gamma: f64) -> HashMap<State, f64> {
    let mut value_map: HashMap<State, f64> = HashMap::new();
    let mut delta = f64::MAX;

    while delta > tolerance {
        delta = 0.0;

        for (state, _) in mdp.transition_probabilities.keys() {
            let old_value = *value_map.get(state).unwrap_or(&0.0);
            let new_value = best_action_value(mdp, *state, &value_map, gamma);

            value_map.insert(*state, new_value);
            delta = delta.max((old_value - new_value).abs());
        }
    }

    value_map
}

fn best_action_value(mdp: &Mdp, state: State, value_map: &HashMap<State, f64>, gamma: f64) -> f64 {
    mdp.transition_probabilities
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
