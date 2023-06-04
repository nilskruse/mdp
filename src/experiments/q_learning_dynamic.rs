use crate::{
    algorithms::GenericStateActionAlgorithm,
    eval::{evaluate_epsilon_greedy_policy, evaluate_greedy_policy},
};
use rand::SeedableRng;

use crate::{
    algorithms::{
        q_learning::QLearning, q_learning_dynamic::QLearningDynamic, StateActionAlgorithm,
    },
    envs,
    utils::print_q_map,
};

pub fn run_experiment() {
    println!("Running deterministic cliff walking with q_learning_dynamic!");
    let cliff_walking_mdp = envs::cliff_walking::build_mdp().unwrap();

    let learning_episodes = 310;
    let eval_episodes = 500;

    // run "indefinitely"
    let learning_max_steps = usize::MAX;
    let eval_max_steps = 2000;

    let alpha = 0.1;
    let gamma = 1.0;
    let epsilon = 0.1;

    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0);

    let q_learning_algo = QLearningDynamic::new(alpha, gamma, epsilon, learning_max_steps);
    let q_map = q_learning_algo.run(&cliff_walking_mdp, learning_episodes, &mut rng);

    let avg_reward = evaluate_epsilon_greedy_policy(
        &cliff_walking_mdp,
        &q_map,
        eval_episodes,
        eval_max_steps,
        epsilon,
        &mut rng,
    );
    println!("Q-learning dynamic average reward with epsilon greedy: {avg_reward}");

    let avg_reward = evaluate_greedy_policy(
        &cliff_walking_mdp,
        &q_map,
        eval_episodes,
        eval_max_steps,
        &mut rng,
    );
    println!("Q-learning dynamic average reward with greedy: {avg_reward}");

    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0);

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
    // print_q_map(&q_map);
}
