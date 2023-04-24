#[macro_use]
extern crate assert_float_eq;

mod algorithms;
mod generator;
mod mdp;
mod policies;
mod benchmarks;

#[cfg(test)]
mod tests;

use std::collections::HashMap;

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use crate::algorithms::{q_learning, sarsa, value_iteration};
use crate::generator::generate_random_mdp;
use crate::mdp::*;
use crate::policies::greedy_policy;

fn main() {
    let mdp = Mdp::new_test_mdp();
    let value_map = value_iteration(&mdp, 0.01, 0.0);


    for (state, value) in value_map.iter() {
        println!("State {:?} has value: {:.4}", state, value);
    }

    // let q_map = sarsa(&mdp, 0.5, 0.5, (State(0), Action(0)), 10, 2000);
    // println!("Q: {:?}", q_map);

    // let avg_reward = evaluate_policy(&mdp, q_map, 1000);
    // println!("sarsa average reward: {avg_reward}");

    // let q_map = q_learning(&mdp, 0.5, 0.5, (State(0), Action(0)), 10, 2000);
    // println!("Q: {:?}", q_map);

    // let avg_reward = evaluate_policy(&mdp, q_map, 1000);
    // println!("Q-learning average reward: {avg_reward}");
    // let greedy_action = greedy_policy(&mdp, &q_map, State(0));
    // println!("Selected greedy_action for State 0: {:?}", greedy_action);
    // let greedy_action = greedy_policy(&mdp, &q_map, State(1));
    // println!("Selected greedy_action for State 1: {:?}", greedy_action);

    // for _ in 0..100 {
    //     println!("Result: {:?}", mdp.perform_action((State(1), Action(1))));
    // }
    // println!("{:?}", mdp.transitions);

    let learning_episodes = 1000;
    let learning_max_steps = 2000;
    let eval_episodes = 1000;
    let eval_max_steps = 2000;

    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(1);
    let mdp = generate_random_mdp(5, 2, 1, (2, 2), (1, 3), (-1.0, 10.0), &mut rng);
    println!("Random generated mdp: {:?}", mdp.transitions);
    println!("Terminal states: {:?}", mdp.terminal_states);

    let q_map = sarsa(
        &mdp,
        0.5,
        0.5,
        (State(0), Action(0)),
        learning_episodes,
        learning_max_steps,
        &mut rng
    );
    println!("Q: {:?}", q_map);

    let avg_reward = evaluate_policy(&mdp, q_map, eval_episodes, eval_max_steps, &mut rng);
    println!("sarsa average reward: {avg_reward}");

    let q_map = q_learning(
        &mdp,
        0.5,
        0.5,
        (State(0), Action(0)),
        learning_episodes,
        learning_max_steps,
        &mut rng
    );
    println!("Q: {:?}", q_map);

    let avg_reward = evaluate_policy(&mdp, q_map, eval_episodes, eval_max_steps, &mut rng);
    println!("Q-learning average reward: {avg_reward}");
}

fn evaluate_policy(
    mdp: &Mdp,
    q_map: HashMap<(State, Action), f64>,
    episodes: usize,
    max_steps: usize,
    rng: &mut ChaCha20Rng,
) -> f64 {
    let mut total_reward = 0.0;

    for episode in 1..=episodes {
        let mut current_state = mdp.initial_state;
        let mut episode_reward = 0.0;
        let mut steps = 0;

        while !mdp.terminal_states.contains(&current_state) && steps < max_steps {
            let selected_action = greedy_policy(mdp, &q_map, current_state, rng);
            let (next_state, reward) = mdp.perform_action((current_state, selected_action), rng);
            episode_reward += reward;
            current_state = next_state;
            steps += 1;
        }
        total_reward += episode_reward;
    }
    total_reward / episodes as f64
}
