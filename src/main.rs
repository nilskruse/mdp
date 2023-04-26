#![feature(test)]
// disable these warnings for now
#![allow(dead_code)]
#![allow(unused_imports)]

#[macro_use]
extern crate assert_float_eq;

mod algorithms;
mod envs;
mod eval;
mod generator;
mod mdp;
mod policies;
mod utils;

#[cfg(test)]
mod tests;

mod benchmarks;

use std::collections::HashMap;

use benchmarks::bench_runtime_sarsa_2;
use policies::epsilon_greedy_policy;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use crate::algorithms::{q_learning, sarsa, value_iteration};
use crate::benchmarks::bench_runtime_q_learning_2;
use crate::eval::evaluate_epsilon_greedy_policy;
use crate::generator::generate_random_mdp;
use crate::mdp::*;
use crate::policies::greedy_policy;
use crate::utils::print_transition_map;

fn main() {
    // let mdp = Mdp::new_test_mdp();
    // let value_map = value_iteration(&mdp, 0.01, 0.0);

    // for (state, value) in value_map.iter() {
    //     println!("State {:?} has value: {:.4}", state, value);
    // }

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

    // let learning_episodes = 1000;
    // let learning_max_steps = 2000;
    // let eval_episodes = 1000;
    // let eval_max_steps = 2000;

    // let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(1);
    // let mdp = generate_random_mdp(3, 2, 1, (2, 2), (1, 3), (-1.0, 10.0), &mut rng);
    // println!("Random generated mdp:");
    // print_transition_map(&mdp);
    // println!("Terminal states: {:?}", mdp.terminal_states);

    // let q_map = sarsa(
    //     &mdp,
    //     0.5,
    //     0.5,
    //     (State(0), Action(0)),
    //     learning_episodes,
    //     learning_max_steps,
    //     &mut rng,
    // );
    // println!("Q: {:?}", q_map);

    // let avg_reward = evaluate_policy(&mdp, q_map, eval_episodes, eval_max_steps, &mut rng);
    // println!("sarsa average reward: {avg_reward}");

    // let q_map = q_learning(
    //     &mdp,
    //     0.5,
    //     0.5,
    //     (State(0), Action(0)),
    //     learning_episodes,
    //     learning_max_steps,
    //     &mut rng,
    // );
    // println!("Q: {:?}", q_map);

    // let avg_reward = evaluate_policy(&mdp, q_map, eval_episodes, eval_max_steps, &mut rng);
    // println!("Q-learning average reward: {avg_reward}");
    // run_benchmarks();
    run_cliff_walking();
}

fn run_benchmarks() {
    println!("Starting Benchmarks...");

    println!("Benching Q-Learning...");
    bench_runtime_q_learning_2();

    println!("Benching SARSA...");
    bench_runtime_sarsa_2();
}

fn run_cliff_walking() {
    let cliff_walking_mdp = envs::cliff_walking::build_mdp();
    // print_transition_map(&cliff_walking_mdp);
    println!(
        "Transition map length:{:?}",
        cliff_walking_mdp.transitions.len()
    );
    println!("Initial state: {:?}", cliff_walking_mdp.initial_state);
    println!("Terminal states: {:?}", cliff_walking_mdp.terminal_states);

    let learning_episodes = 1000;
    let eval_episodes = 1000;

    // run "indefinitely"
    let learning_max_steps = usize::MAX;
    let eval_max_steps = usize::MAX;

    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(1);

    let q_map = q_learning(
        &cliff_walking_mdp,
        0.1,
        0.9,
        (cliff_walking_mdp.initial_state, Action(0)),
        learning_episodes,
        learning_max_steps,
        &mut rng,
    );

    let avg_reward = evaluate_epsilon_greedy_policy(
        &cliff_walking_mdp,
        q_map,
        eval_episodes,
        eval_max_steps,
        &mut rng,
    );
    println!("Q-learning average reward: {avg_reward}");

    let q_map = sarsa(
        &cliff_walking_mdp,
        0.1,
        0.9,
        (cliff_walking_mdp.initial_state, Action(0)),
        learning_episodes,
        learning_max_steps,
        &mut rng,
    );

    let avg_reward = evaluate_epsilon_greedy_policy(
        &cliff_walking_mdp,
        q_map,
        eval_episodes,
        eval_max_steps,
        &mut rng,
    );
    println!("SARSA average reward: {avg_reward}");
}
