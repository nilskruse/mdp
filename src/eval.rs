use crate::{mdp::GenericMdp, policies::random_policy};
use std::collections::BTreeMap;

use rand_chacha::ChaCha20Rng;

use crate::{
    mdp::{GenericAction, GenericState},
    policies::{epsilon_greedy_policy, greedy_policy},
};

pub fn evaluate_epsilon_greedy_policy<M: GenericMdp<S, A>, S: GenericState, A: GenericAction>(
    mdp: &M,
    q_map: &BTreeMap<(S, A), f64>,
    episodes: usize,
    max_steps: usize,
    epsilon: f64,
    rng: &mut ChaCha20Rng,
) -> f64 {
    let mut total_reward = 0.0;

    for _episode in 1..=episodes {
        let mut current_state = mdp.get_initial_state(rng);
        let mut episode_reward = 0.0;
        let mut steps = 0;

        while !mdp.is_terminal(current_state) && steps < max_steps {
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

pub fn evaluate_greedy_policy<M: GenericMdp<S, A>, S: GenericState, A: GenericAction>(
    mdp: &M,
    q_map: &BTreeMap<(S, A), f64>,
    episodes: usize,
    max_steps: usize,
    rng: &mut ChaCha20Rng,
) -> f64 {
    let mut total_reward = 0.0;

    for _episode in 1..=episodes {
        let mut current_state = mdp.get_initial_state(rng);
        let mut episode_reward = 0.0;
        let mut steps = 0;

        while !mdp.is_terminal(current_state) && steps < max_steps {
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

pub fn evaluate_random_policy<M: GenericMdp<S, A>, S: GenericState, A: GenericAction>(
    mdp: &M,
    episodes: usize,
    max_steps: usize,
    rng: &mut ChaCha20Rng,
) -> f64 {
    let mut total_reward = 0.0;

    for _episode in 1..=episodes {
        let mut current_state = mdp.get_initial_state(rng);
        let mut episode_reward = 0.0;
        let mut steps = 0;

        while !mdp.is_terminal(current_state) && steps < max_steps {
            let selected_action = random_policy(mdp, current_state, rng);
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
