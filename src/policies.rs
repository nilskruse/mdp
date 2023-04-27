use std::{cmp::Ordering, collections::BTreeMap};

use itertools::Itertools;
use rand::Rng;
use rand_chacha::ChaCha20Rng;

use crate::mdp::{Action, Mdp, Reward, State};

pub fn random_policy(mdp: &Mdp, current_state: State, rng: &mut ChaCha20Rng) -> Option<Action> {
    let mut possible_actions = mdp.get_possible_actions(current_state);

    // possible_actions.sort_by(|a1, a2| a1.0.cmp(&a2.0));

    if possible_actions.is_empty() {
        None
    } else {
        let selected_index = rng.gen_range(0..possible_actions.len());
        Some(possible_actions[selected_index])
    }
}

pub fn greedy_policy(
    mdp: &Mdp,
    q_map: &BTreeMap<(State, Action), Reward>,
    current_state: State,
    rng: &mut ChaCha20Rng,
) -> Option<Action> {
    let selected_action = q_map
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
                } else if current_reward < prev_reward {
                    prev
                } else {
                    None
                }
            } else {
                Some((current_action, current_reward))
            }
        })
        .unzip()
        .0;

    if let Some(action) = selected_action {
        Some(action)
    } else {
        let random_policy_action = random_policy(mdp, current_state, rng);
        random_policy_action
    }
}

pub fn epsilon_greedy_policy(
    mdp: &Mdp,
    q_map: &BTreeMap<(State, Action), Reward>,
    current_state: State,
    epsilon: f64,
    rng: &mut ChaCha20Rng,
) -> Option<Action> {
    let random_value = rng.gen_range(0.0..1.0);
    if random_value < (1.0 - epsilon) {
        greedy_policy(mdp, q_map, current_state, rng)
    } else {
        random_policy(mdp, current_state, rng)
    }
}
