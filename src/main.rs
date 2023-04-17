mod mdp;
use crate::mdp::*;
use std::collections::HashMap;

fn main() {
    let mdp = Mdp::new_test_mdp(0.0);
    let tolerance = 0.01;
    let value_map = value_iteration(&mdp, tolerance);

    mdp.perform_action((State(0u8), Action::A));

    for (state, value) in value_map.iter() {
        println!("State {:?} has value: {:.4}", state, value);
    }

    sarsa(&mdp, 0.5, 0.5, (State(0), Action::A));
}

fn sarsa(
    mdp: &Mdp,
    alpha: f64,
    gamma: f64,
    initial: (State, Action),
) -> HashMap<(State, Action), f64> {
    let q_map: HashMap<(State, Action), f64> = HashMap::new();
    let next_state = mdp.perform_action(initial);
    println!("Next state: {:?}", next_state);
    q_map
}

fn value_iteration(mdp: &Mdp, tolerance: f64) -> HashMap<State, f64> {
    let mut value_map: HashMap<State, f64> = HashMap::new();
    let mut delta = f64::MAX;

    while delta > tolerance {
        delta = 0.0;

        for (state, _) in mdp.transition_probabilities.keys() {
            let old_value = *value_map.get(state).unwrap_or(&0.0);
            let new_value = best_action_value(mdp, *state, &value_map);

            value_map.insert(*state, new_value);
            delta = delta.max((old_value - new_value).abs());
        }
    }

    value_map
}

fn best_action_value(mdp: &Mdp, state: State, value_map: &HashMap<State, f64>) -> f64 {
    mdp.transition_probabilities
        .iter()
        .filter_map(|((s, _), transitions)| {
            if *s == state {
                let expected_value: f64 = transitions
                    .iter()
                    .map(|(prob, next_state, reward)| {
                        prob * (reward
                            + mdp.discount_factor * value_map.get(next_state).unwrap_or(&0.0))
                    })
                    .sum();
                Some(expected_value)
            } else {
                None
            }
        })
        .fold(f64::MIN, f64::max)
}
