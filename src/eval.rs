use std::collections::BTreeMap;

use rand_chacha::ChaCha20Rng;

use crate::{
    mdp::{Action, GenericAction, GenericMdp, GenericState, Mdp, State},
    policies::{
        epsilon_greedy_policy, epsilon_greedy_policy_generic, greedy_policy, greedy_policy_generic,
    },
};

pub fn evaluate_epsilon_greedy_policy(
    mdp: &Mdp,
    q_map: &BTreeMap<(State, Action), f64>,
    episodes: usize,
    max_steps: usize,
    epsilon: f64,
    rng: &mut ChaCha20Rng,
) -> f64 {
    let mut total_reward = 0.0;

    for _episode in 1..=episodes {
        let mut current_state = mdp.initial_state;
        let mut episode_reward = 0.0;
        let mut steps = 0;

        while !mdp.terminal_states.contains(&current_state) && steps < max_steps {
            let selected_action = epsilon_greedy_policy(mdp, q_map, current_state, epsilon, rng);
            if let Some(selected_action) = selected_action {
                let (next_state, reward) =
                    mdp.perform_action((current_state, selected_action), rng);
                episode_reward += reward;
                current_state = next_state;
                steps += 1;
            } else {
                break;
            }
        }
        total_reward += episode_reward;
    }
    total_reward / episodes as f64
}

pub fn evaluate_greedy_policy(
    mdp: &Mdp,
    q_map: &BTreeMap<(State, Action), f64>,
    episodes: usize,
    max_steps: usize,
    rng: &mut ChaCha20Rng,
) -> f64 {
    let mut total_reward = 0.0;

    for _episode in 1..=episodes {
        let mut current_state = mdp.initial_state;
        let mut episode_reward = 0.0;
        let mut steps = 0;

        while !mdp.terminal_states.contains(&current_state) && steps < max_steps {
            let selected_action = greedy_policy(mdp, q_map, current_state, rng);
            if let Some(selected_action) = selected_action {
                let (next_state, reward) =
                    mdp.perform_action((current_state, selected_action), rng);
                episode_reward += reward;
                current_state = next_state;
                steps += 1;
            } else {
                break;
            }
        }
        total_reward += episode_reward;
    }
    total_reward / episodes as f64
}

pub fn evaluate_epsilon_greedy_policy_generic<S: GenericState, A: GenericAction>(
    mdp: &GenericMdp<S, A>,
    q_map: &BTreeMap<(S, A), f64>,
    episodes: usize,
    max_steps: usize,
    epsilon: f64,
    rng: &mut ChaCha20Rng,
) -> f64 {
    let mut total_reward = 0.0;

    for _episode in 1..=episodes {
        let mut current_state = mdp.initial_state;
        let mut episode_reward = 0.0;
        let mut steps = 0;

        while !mdp.terminal_states.contains(&current_state) && steps < max_steps {
            let selected_action =
                epsilon_greedy_policy_generic(mdp, q_map, current_state, epsilon, rng);
            if let Some(selected_action) = selected_action {
                let (next_state, reward) =
                    mdp.perform_action((current_state, selected_action), rng);
                episode_reward += reward;
                current_state = next_state;
                steps += 1;
            } else {
                break;
            }
        }
        total_reward += episode_reward;
    }
    total_reward / episodes as f64
}

pub fn evaluate_greedy_policy_generic<S: GenericState, A: GenericAction>(
    mdp: &GenericMdp<S, A>,
    q_map: &BTreeMap<(S, A), f64>,
    episodes: usize,
    max_steps: usize,
    rng: &mut ChaCha20Rng,
) -> f64 {
    let mut total_reward = 0.0;

    for _episode in 1..=episodes {
        let mut current_state = mdp.initial_state;
        let mut episode_reward = 0.0;
        let mut steps = 0;

        while !mdp.terminal_states.contains(&current_state) && steps < max_steps {
            let selected_action = greedy_policy_generic(mdp, q_map, current_state, rng);
            if let Some(selected_action) = selected_action {
                let (next_state, reward) =
                    mdp.perform_action((current_state, selected_action), rng);
                episode_reward += reward;
                current_state = next_state;
                steps += 1;
            } else {
                break;
            }
        }
        total_reward += episode_reward;
    }
    total_reward / episodes as f64
}
