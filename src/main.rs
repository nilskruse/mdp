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
    let greedy_action = greedy_policy(&mdp, &q_map, State(0));
    println!("Selected greedy_action for State 0: {:?}", greedy_action);
    let greedy_action = greedy_policy(&mdp, &q_map, State(1));
    println!("Selected greedy_action for State 1: {:?}", greedy_action);
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

    for _ in 0..1000 {
        let (next_state, reward) = mdp.perform_action((current_state, current_action));

        // TODO: actual policy
        let next_action = epsilon_greedy_policy(mdp, &q_map, next_state, 0.1);

        println!(
            "Quintuple: ({:?},{:?},{:?},{:?},{:?})",
            current_state, current_action, reward, next_state, next_action
        );

        // update q_map
        let next_q = *q_map.get(&(next_state, next_action)).unwrap_or(&0.0);
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

fn greedy_policy(
    mdp: &Mdp,
    q_map: &HashMap<(State, Action), Reward>,
    current_state: State,
) -> Action {
    q_map
        .iter()
        .filter_map(|((state, action), reward)| {
            if state.eq(&current_state) {
                Some((*action, *reward))
            } else {
                None
            }
        })
        .fold(None, |prev, (current_action, current_reward)| {
            if let Some((_, prev_reward)) = prev {
                if current_reward > prev_reward {
                    Some((current_action, current_reward))
                } else {
                    prev
                }
            } else {
                Some((current_action, current_reward))
            }
        })
        .unwrap_or((random_policy(mdp, current_state), 0.0)) // random when no entry
        .0
}

fn epsilon_greedy_policy(
    mdp: &Mdp,
    q_map: &HashMap<(State, Action), Reward>,
    current_state: State,
    epsilon: f64,
) -> Action {
    let mut rng = rand::thread_rng();
    let random_value = rng.gen_range(0.0..1.0);
    if random_value < (1.0 - epsilon) {
        greedy_policy(mdp, q_map, current_state)
    } else {
        random_policy(mdp, current_state)
    }
}

fn value_iteration(mdp: &Mdp, tolerance: f64, gamma: f64) -> HashMap<State, f64> {
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
