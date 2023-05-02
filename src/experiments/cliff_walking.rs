use rand::SeedableRng;
use std::io;

use crate::{
    algorithms::{q_learning::QLearning, sarsa::Sarsa, TDAlgorithm},
    envs,
    eval::{evaluate_epsilon_greedy_policy, evaluate_greedy_policy},
};

pub fn run_cliff_walking() {
    println!("Running deterministic cliff walking!");
    let cliff_walking_mdp = envs::cliff_walking::build_mdp();

    let learning_episodes = 500;
    let eval_episodes = 1000;

    // run "indefinitely"
    let learning_max_steps = usize::MAX;
    let eval_max_steps = usize::MAX;

    let alpha = 0.1;
    let gamma = 1.0;
    let epsilon = 0.1;

    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(1);

    let q_learning_algo = QLearning::new(alpha, gamma, epsilon, learning_max_steps);
    let q_map = q_learning_algo.run(&cliff_walking_mdp, learning_episodes, &mut rng);

    let avg_reward = evaluate_epsilon_greedy_policy(
        &cliff_walking_mdp,
        &q_map,
        eval_episodes,
        eval_max_steps,
        epsilon,
        &mut rng,
    );
    println!("Q-learning average reward with epsilon greedy: {avg_reward}");

    let avg_reward = evaluate_greedy_policy(
        &cliff_walking_mdp,
        &q_map,
        eval_episodes,
        eval_max_steps,
        &mut rng,
    );
    println!("Q-learning average reward with greedy: {avg_reward}");

    let sarsa_algo = Sarsa::new(alpha, gamma, epsilon, learning_max_steps);
    let q_map = sarsa_algo.run(&cliff_walking_mdp, learning_episodes, &mut rng);

    let avg_reward = evaluate_epsilon_greedy_policy(
        &cliff_walking_mdp,
        &q_map,
        eval_episodes,
        eval_max_steps,
        epsilon,
        &mut rng,
    );
    println!("SARSA average reward with epsilon greedy: {avg_reward}");

    let avg_reward = evaluate_greedy_policy(
        &cliff_walking_mdp,
        &q_map,
        eval_episodes,
        eval_max_steps,
        &mut rng,
    );
    println!("SARSA average reward with greedy: {avg_reward}");
    println!();
}

pub fn run_slippery_cliff_walking() {
    println!("Running slippery cliff walking!");
    let slippy_cliff_walking_mdp = envs::slippery_cliff_walking::build_mdp(0.2);

    let learning_episodes = 500;
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

    let sum: f64 = q_map.values().sum();

    println!("Sum = {sum}");

    let avg_reward = evaluate_epsilon_greedy_policy(
        &slippy_cliff_walking_mdp,
        &q_map,
        eval_episodes,
        eval_max_steps,
        epsilon,
        &mut rng,
    );
    println!("Q-learning average reward with epsilon greedy: {avg_reward}");

    let avg_reward = evaluate_greedy_policy(
        &slippy_cliff_walking_mdp,
        &q_map,
        eval_episodes,
        eval_max_steps,
        &mut rng,
    );
    println!("Q-learning average reward with greedy: {avg_reward}");

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
    println!("SARSA average reward with epsilon greedy: {avg_reward}");

    let avg_reward = evaluate_greedy_policy(
        &slippy_cliff_walking_mdp,
        &q_map,
        eval_episodes,
        eval_max_steps,
        &mut rng,
    );
    println!("SARSA average reward with greedy: {avg_reward}");
    println!();
}

pub fn run_cliff_walking_episodic() {
    println!("Try replicating book results...");
    let cliff_walking_mdp = envs::cliff_walking::build_mdp();

    let eval_episodes = 1000;

    // run "indefinitely"
    let learning_max_steps = usize::MAX;
    let eval_max_steps = usize::MAX;

    let alpha = 0.3;
    let gamma = 1.0;
    let epsilon = 0.1;

    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(1);

    let q_learning_algo = QLearning::new(alpha, gamma, epsilon, learning_max_steps);

    let mut q_map = q_learning_algo.run(&cliff_walking_mdp, 1, &mut rng);
    let mut csv_writer = csv::Writer::from_path("q_learning.csv").expect("csv error");
    csv_writer
        .write_record(&["episode", "avg_reward"])
        .expect("csv error");

    for i in 2..=500 {
        q_learning_algo.run_with_q_map(&cliff_walking_mdp, 1, &mut rng, &mut q_map);

        if i % 10 == 0 {
            let avg_reward = evaluate_epsilon_greedy_policy(
                &cliff_walking_mdp,
                &q_map,
                eval_episodes,
                eval_max_steps,
                epsilon,
                &mut rng,
            );
            // println!("Average reward in episode {i}: {avg_reward}");
            csv_writer.serialize((i, avg_reward)).expect("csv error");
        }
    }

    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(1);

    let sarsa_algo = Sarsa::new(alpha, gamma, epsilon, learning_max_steps);

    let mut q_map = sarsa_algo.run(&cliff_walking_mdp, 1, &mut rng);

    let mut csv_writer = csv::Writer::from_path("sarsa.csv").expect("csv error");
    csv_writer
        .write_record(&["episode", "avg_reward"])
        .expect("csv error");

    for i in 2..=500 {
        sarsa_algo.run_with_q_map(&cliff_walking_mdp, 1, &mut rng, &mut q_map);

        if i % 10 == 0 {
            let avg_reward = evaluate_epsilon_greedy_policy(
                &cliff_walking_mdp,
                &q_map,
                eval_episodes,
                eval_max_steps,
                epsilon,
                &mut rng,
            );
            // println!("Average reward in episode {i}: {avg_reward}");
            csv_writer.serialize((i, avg_reward)).expect("csv error");
        }
    }

    println!();
}
