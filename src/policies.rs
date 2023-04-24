use std::{cmp::Ordering, collections::HashMap};

use rand::Rng;
use rand_chacha::ChaCha20Rng;

use crate::{
    mdp::{Action, Reward, State},
    Mdp,
};

pub fn random_policy(mdp: &Mdp, current_state: State, rng: &mut ChaCha20Rng) -> Action {
    let mut possible_actions = mdp.get_possible_actions(current_state);

    possible_actions.sort_by(|a1, a2| {
        if a1.0 > a2.0 {
            Ordering::Greater
        } else if a1.0 < a2.0 {
            Ordering::Less
        } else {
            Ordering::Equal
        }
    });
    let selected_index = rng.gen_range(0..possible_actions.len());

    possible_actions[selected_index]
}

pub fn greedy_policy(
    mdp: &Mdp,
    q_map: &HashMap<(State, Action), Reward>,
    current_state: State,
    rng: &mut ChaCha20Rng,
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
                } else if current_reward < prev_reward {
                    prev
                } else {
                    None
                }
            } else {
                Some((current_action, current_reward))
            }
        })
        .unwrap_or((random_policy(mdp, current_state, rng), 0.0)) // random when no entry
        .0
}

pub fn epsilon_greedy_policy(
    mdp: &Mdp,
    q_map: &HashMap<(State, Action), Reward>,
    current_state: State,
    epsilon: f64,
    rng: &mut ChaCha20Rng,
) -> Action {
    let random_value = rng.gen_range(0.0..1.0);
    if random_value < (1.0 - epsilon) {
        greedy_policy(mdp, q_map, current_state, rng)
    } else {
        random_policy(mdp, current_state, rng)
    }
}
