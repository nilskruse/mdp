use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use crate::{
    algorithms::{
        dyna_q::{Dyna, DynaQ},
        monte_carlo::MonteCarlo,
        q_learning::QLearning,
        q_learning_lambda::QLearningLambda,
        sarsa::Sarsa,
        sarsa_lambda::SarsaLambda,
        GenericStateActionAlgorithm, Trace,
    },
    envs::my_intersection::MyIntersectionMdp,
    eval::{evaluate_greedy_policy, evaluate_random_policy},
    experiments::intersection::fixed_cycle,
    mdp::{GenericAction, GenericMdp, GenericState},
};

pub fn compare_intersection() {
    let seed: u64 = 0;
    let num_seeds: usize = 100;
    // let cw_mdp = crate::envs::cliff_walking::build_mdp().unwrap();
    let ns_prob = 0.6;
    let ew_prob = 0.2;
    let max_cars = 10;
    let mdp = crate::envs::my_intersection::MyIntersectionMdp::new(ns_prob, ew_prob, max_cars);

    let alpha = 0.1;
    let epsilon = 0.1;
    let lambda = 0.7;
    let trace = Trace::Replacing;
    let k = 5;
    let deterministic = true;
    let max_steps = 2000;
    let train_episodes = 1000;
    let mut results: Vec<(String, f64)> = vec![];

    // monte carlo
    println!("MC");
    let mc_algo = MonteCarlo::new(epsilon, max_steps);
    let mc_reward =
        bench_average_strategy(&mdp, &mc_algo, seed, num_seeds, train_episodes, max_steps);
    results.push(("MC".to_owned(), mc_reward));

    // Q-Learning
    println!("Q");
    let q_algo = QLearning::new(alpha, epsilon, max_steps);
    let q_reward =
        bench_average_strategy(&mdp, &q_algo, seed, num_seeds, train_episodes, max_steps);
    results.push(("Q-Learning".to_owned(), q_reward));

    // SARSA
    println!("SARSA");
    let sarsa_algo = Sarsa::new(alpha, epsilon, max_steps);
    let sarsa_reward = bench_average_strategy(
        &mdp,
        &sarsa_algo,
        seed,
        num_seeds,
        train_episodes,
        max_steps,
    );
    results.push(("SARSA".to_owned(), sarsa_reward));

    // Q-Learning(lambda)
    println!("Q lambda");
    let q_lambda_algo = QLearningLambda::new(alpha, epsilon, lambda, max_steps, trace);
    let q_lambda_reward = bench_average_strategy(
        &mdp,
        &q_lambda_algo,
        seed,
        num_seeds,
        train_episodes,
        max_steps,
    );
    results.push(("Q-Learning(lambda)".to_owned(), q_lambda_reward));

    // Q-Learning(lambda)
    println!("SARSA lambda");
    let sarsa_lambda_algo = SarsaLambda::new(alpha, epsilon, lambda, max_steps, trace);
    let sarsa_lambda_reward = bench_average_strategy(
        &mdp,
        &sarsa_lambda_algo,
        seed,
        num_seeds,
        train_episodes,
        max_steps,
    );
    results.push(("SARSA(lambda)".to_owned(), sarsa_lambda_reward));

    // DynaQ
    println!("DynaQ");
    let mut dyna_q_algo = DynaQ::new(alpha, epsilon, k, max_steps, deterministic, true, &mdp);
    let dyna_q_reward = bench_average_dynaq(
        &mdp,
        &mut dyna_q_algo,
        seed,
        num_seeds,
        train_episodes,
        max_steps,
    );
    results.push(("DynaQ".to_owned(), dyna_q_reward));

    // Fixed cycle
    let ns_time = 6;
    let ew_time = 2;
    let fixed_cycle_reward =
        bench_average_fixed_cycle(&mdp, seed, num_seeds, max_steps, ns_time, ew_time);
    results.push(("Fixed cycle".to_owned(), fixed_cycle_reward));

    // random
    let random_reward = bench_average_random(&mdp, seed, num_seeds, max_steps);
    results.push(("Random".to_owned(), random_reward));

    //
    // let mut _rng = ChaCha20Rng::seed_from_u64(seed);
    // let _algo = ::new(alpha, epsilon, max_steps);
    // let _time = bench_runtime(env, &_algo, episodes, &mut _rng);
    // results.push(("", _time));
    dbg!(&results);
    let mut path =
        format!("results/intersection_reward_alpha{alpha}_epsilon{epsilon}_lambda{lambda}_{trace}_more_episodes");
    path = path.replace(".", "_");
    path.push_str(".csv");
    dbg!(&path);

    let mut csv_writer = csv::Writer::from_path(path).expect("csv file error");

    let record = results.iter().map(|(algo, _)| algo.clone());
    let result: Vec<f64> = results.iter().map(|(_, result)| *result).collect();

    csv_writer
        .write_record(record)
        .expect("csv write record error");
    csv_writer.serialize(result).expect("csv error");
}

fn bench_average_strategy<M: GenericMdp<S, A>, S: GenericState, A: GenericAction>(
    env: &M,
    algo: &impl GenericStateActionAlgorithm,
    seed: u64,
    num_seeds: usize,
    train_episodes: usize,
    max_steps: usize,
) -> f64 {
    let mut eval_rng = ChaCha20Rng::seed_from_u64(seed);
    let eval_episodes = 10;
    let mut total_rewards = 0.0;

    for i in 0..num_seeds {
        let mut rng = ChaCha20Rng::seed_from_u64(seed + i as u64);
        let q_map = algo.run(env, train_episodes, &mut rng);
        total_rewards +=
            evaluate_greedy_policy(env, &q_map, eval_episodes, max_steps, &mut eval_rng);
    }
    total_rewards / num_seeds as f64
}

fn bench_average_dynaq<M: GenericMdp<S, A>, S: GenericState, A: GenericAction>(
    env: &M,
    algo: &mut DynaQ<S, A>,
    seed: u64,
    num_seeds: usize,
    train_episodes: usize,
    max_steps: usize,
) -> f64 {
    let mut eval_rng = ChaCha20Rng::seed_from_u64(seed);
    let eval_episodes = 10;
    let mut total_rewards = 0.0;

    for i in 0..num_seeds {
        let mut rng = ChaCha20Rng::seed_from_u64(seed + i as u64);
        algo.clear_model();
        let q_map = algo.run(env, train_episodes, &mut rng);
        total_rewards +=
            evaluate_greedy_policy(env, &q_map, eval_episodes, max_steps, &mut eval_rng);
    }
    total_rewards / num_seeds as f64
}

fn bench_average_fixed_cycle(
    env: &MyIntersectionMdp,
    seed: u64,
    num_seeds: usize,
    max_steps: usize,
    ns_time: usize,
    ew_time: usize,
) -> f64 {
    let eval_episodes = 10;
    let mut total_rewards = 0.0;

    for i in 0..num_seeds {
        let mut rng = ChaCha20Rng::seed_from_u64(seed + i as u64);
        total_rewards += fixed_cycle(env, eval_episodes, max_steps, ns_time, ew_time, &mut rng);
    }
    total_rewards / num_seeds as f64
}

fn bench_average_random(
    env: &MyIntersectionMdp,
    seed: u64,
    num_seeds: usize,
    max_steps: usize,
) -> f64 {
    let eval_episodes = 10;
    let mut total_rewards = 0.0;

    for i in 0..num_seeds {
        let mut rng = ChaCha20Rng::seed_from_u64(seed + i as u64);
        total_rewards += evaluate_random_policy(env, eval_episodes, max_steps, &mut rng);
    }
    total_rewards / num_seeds as f64
}

pub fn test_intersection_params() {
    test_params(Trace::Accumulating);
    test_params(Trace::Replacing);
    test_params(Trace::Dutch);
}

fn test_params(trace: Trace) {
    let mut path = format!("results/intersection_param_test_{trace}");
    path = path.replace(".", "_");
    path.push_str(".csv");
    dbg!(&path);
    let mut csv_writer = csv::Writer::from_path(path).expect("csv file error");
    csv_writer
        .write_record(["Lambda", "Q", "SARSA"])
        .expect("csv write record error");
    for i in 0..=10 {
        let lambda = i as f64 * 0.1;
        dbg!(lambda, trace);
        let (q, sarsa) = test_params_lambda_trace(lambda, trace);

        csv_writer.serialize((lambda, q, sarsa)).expect("csv error");
    }
}

fn test_params_lambda_trace(lambda: f64, trace: Trace) -> (f64, f64) {
    let seed: u64 = 0;
    let num_seeds: usize = 1;
    let ns_prob = 0.6;
    let ew_prob = 0.2;
    let max_cars = 10;
    let mdp = crate::envs::my_intersection::MyIntersectionMdp::new(ns_prob, ew_prob, max_cars);

    let alpha = 0.1;
    let epsilon = 0.1;
    let max_steps = 2000;
    let train_episodes = 100;

    // Q-Learning(lambda)
    println!("Q lambda");
    let q_lambda_algo = QLearningLambda::new(alpha, epsilon, lambda, max_steps, trace);
    let q_lambda_reward = bench_average_strategy(
        &mdp,
        &q_lambda_algo,
        seed,
        num_seeds,
        train_episodes,
        max_steps,
    );

    // Q-Learning(lambda)
    println!("SARSA lambda");
    let sarsa_lambda_algo = SarsaLambda::new(alpha, epsilon, lambda, max_steps, trace);
    let sarsa_lambda_reward = bench_average_strategy(
        &mdp,
        &sarsa_lambda_algo,
        seed,
        num_seeds,
        train_episodes,
        max_steps,
    );

    (q_lambda_reward, sarsa_lambda_reward)
}
