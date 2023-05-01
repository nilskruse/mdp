// disable these warnings for now
#![allow(dead_code)]
#![allow(unused_imports)]

use std::collections::BTreeMap;

use mdp::algorithms::q_learning::QLearning;
use mdp::algorithms::sarsa::Sarsa;
use mdp::benchmarks::run_benchmarks;
use mdp::policies::epsilon_greedy_policy;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use mdp::algorithms::{q_learning, sarsa, value_iteration, TDAlgorithm};
use mdp::eval::{evaluate_epsilon_greedy_policy, evaluate_greedy_policy};
use mdp::generator::generate_random_mdp;
use mdp::{mdp::*, experiments};
use mdp::policies::greedy_policy;
use mdp::utils::print_transition_map;

fn main() {

    // run_benchmarks();
    experiments::cliff_walking::run_cliff_walking();
    experiments::cliff_walking::run_slippery_cliff_walking();
    // experiments::cliff_walking::run_cliff_walking_episodic();
    // test();
}


fn test() {
    println!("Running slippy cliff walking!");
    let slippy_cliff_walking_mdp = mdp::envs::slippery_cliff_walking::build_mdp(0.2);
    // print_transition_map(&slippy_cliff_walking_mdp);
    //
    // println!(
    //     "Transition map length:{:?}",
    //     slippy_cliff_walking_mdp.transitions.len()
    // );
    // println!(
    //     "Initial state: {:?}",
    //     slippy_cliff_walking_mdp.initial_state
    // );
    // println!(
    //     "Terminal states: {:?}",
    //     slippy_cliff_walking_mdp.terminal_states
    // );

    let learning_episodes = 10000;
    let eval_episodes = 1000;

    // run "indefinitely"
    let learning_max_steps = usize::MAX;
    let eval_max_steps = usize::MAX;

    let alpha = 0.1;
    let gamma = 1.0;
    let epsilon = 0.1;

    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(1);

    let q_learning_algo = QLearning::new(alpha, gamma, epsilon, learning_max_steps);
    let q_map = q_learning_algo.run(&slippy_cliff_walking_mdp, learning_episodes, &mut rng);

    let avg_reward = evaluate_epsilon_greedy_policy(
        &slippy_cliff_walking_mdp,
        &q_map,
        eval_episodes,
        eval_max_steps,
        epsilon,
        &mut rng,
    );
    println!("Q-learning average reward with epsilon greedy: {avg_reward}");

    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(1);
    let q_learning_algo = QLearning::new(alpha, gamma, epsilon, learning_max_steps);
    let q_map = q_learning_algo.run(&slippy_cliff_walking_mdp, learning_episodes, &mut rng);

    let avg_reward = evaluate_epsilon_greedy_policy(
        &slippy_cliff_walking_mdp,
        &q_map,
        eval_episodes,
        eval_max_steps,
        epsilon,
        &mut rng,
    );
    println!("Q-Learning(struct) average reward with epsion greedy: {avg_reward}");

    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(1);
    let sarsa_algo = QLearning::new(alpha, gamma, epsilon, learning_max_steps);
    let mut q_map = sarsa_algo.run(&slippy_cliff_walking_mdp, learning_episodes / 2, &mut rng);
    sarsa_algo.run_with_q_map(
        &slippy_cliff_walking_mdp,
        learning_episodes / 2,
        &mut rng,
        &mut q_map,
    );

    let avg_reward = evaluate_epsilon_greedy_policy(
        &slippy_cliff_walking_mdp,
        &q_map,
        eval_episodes,
        eval_max_steps,
        epsilon,
        &mut rng,
    );
    println!("Q-Learning(struct) with q_map average reward with epsion greedy: {avg_reward}");

    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(1);
    let sarsa_algo = Sarsa::new(alpha, gamma, epsilon, learning_max_steps);
    let q_map = sarsa_algo.run(&slippy_cliff_walking_mdp, learning_episodes, &mut rng);

    let avg_reward = evaluate_epsilon_greedy_policy(
        &slippy_cliff_walking_mdp,
        &q_map,
        eval_episodes,
        eval_max_steps,
        epsilon,
        &mut rng,
    );
    println!("SARSA(struct) average reward with epsion greedy: {avg_reward}");

    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(1);
    let sarsa_algo = Sarsa::new(alpha, gamma, epsilon, learning_max_steps);
    let mut q_map = sarsa_algo.run(&slippy_cliff_walking_mdp, learning_episodes / 2, &mut rng);
    sarsa_algo.run_with_q_map(
        &slippy_cliff_walking_mdp,
        learning_episodes / 2,
        &mut rng,
        &mut q_map,
    );

    let avg_reward = evaluate_epsilon_greedy_policy(
        &slippy_cliff_walking_mdp,
        &q_map,
        eval_episodes,
        eval_max_steps,
        epsilon,
        &mut rng,
    );
    println!("SARSA(struct) with q_map average reward with epsion greedy: {avg_reward}");
    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(1);
    let q_map = sarsa_algo.run(&slippy_cliff_walking_mdp, learning_episodes, &mut rng);
    let avg_reward = evaluate_epsilon_greedy_policy(
        &slippy_cliff_walking_mdp,
        &q_map,
        eval_episodes,
        eval_max_steps,
        epsilon,
        &mut rng,
    );
    println!("SARSA average reward with epsion greedy: {avg_reward}");

    println!();
}
