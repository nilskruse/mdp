use std::collections::BTreeMap;

use crate::mdp::{Mdp, State};

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
