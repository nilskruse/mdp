use crate::algorithms::dyna_q::{Dyna, DynaQ};
use crate::algorithms::monte_carlo::MonteCarlo;
use crate::algorithms::q_learning_lambda::QLearningLambda;
use crate::algorithms::sarsa_lambda::SarsaLambda;
use crate::algorithms::{GenericStateActionAlgorithm, Trace};
use crate::mdp::{GenericAction, GenericMdp, GenericState, IndexAction, IndexState};
use std::collections::HashMap;
use std::time::{Duration, Instant};

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use crate::{algorithms::sarsa::Sarsa, generator::generate_random_mdp};

use crate::algorithms::q_learning::QLearning;

/// Bench runtime of algorithm on an mdp given set number of episodes
fn bench_runtime<M: GenericMdp<S, A>, S: GenericState, A: GenericAction>(
    env: &M,
    algo: &impl GenericStateActionAlgorithm,
    episodes: usize,
    seed: u64,
    num_seeds: usize,
) -> Duration {
    let mut total_duration: Duration = Duration::new(0, 0);
    for i in 0..num_seeds {
        let mut rng = ChaCha20Rng::seed_from_u64(seed + i as u64);
        let start = Instant::now();
        algo.run(env, episodes, &mut rng);
        total_duration = total_duration.saturating_add(start.elapsed());
    }
    total_duration.div_f64(num_seeds as f64)
}

fn bench_runtime_dyna<M: GenericMdp<S, A>, S: GenericState, A: GenericAction>(
    env: &M,
    algo: &mut DynaQ<S, A>,
    episodes: usize,
    seed: u64,
    num_seeds: usize,
) -> Duration {
    let mut total_duration: Duration = Duration::new(0, 0);
    for i in 0..num_seeds {
        let mut rng = ChaCha20Rng::seed_from_u64(seed + i as u64);
        algo.clear_model();
        let start = Instant::now();
        algo.run(env, episodes, &mut rng);
        total_duration = total_duration.saturating_add(start.elapsed());
    }
    total_duration.div_f64(num_seeds as f64)
}

fn bench_all_algo<M: GenericMdp<S, A>, S: GenericState, A: GenericAction>(
    env: &M,
    episodes: usize,
    seed: u64,
    num_seeds: usize,
    deterministic: bool,
) -> Vec<(String, f64)> {
    // some general parameters used by all algos
    let alpha = 0.1;
    let epsilon = 0.1;
    let lambda = 0.9;
    let k = 5;
    let max_steps = 1000;
    let mut results: Vec<(String, f64)> = vec![];

    // monte carlo
    let mc_algo = MonteCarlo::new(epsilon, max_steps);
    let mc_time = bench_runtime(env, &mc_algo, episodes, seed, num_seeds);
    results.push(("MC".to_owned(), mc_time.as_secs_f64()));

    // Q-Learning
    let q_algo = QLearning::new(alpha, epsilon, max_steps);
    let q_time = bench_runtime(env, &q_algo, episodes, seed, num_seeds);
    results.push(("Q-Learning".to_owned(), q_time.as_secs_f64()));

    // SARSA
    let sarsa_algo = Sarsa::new(alpha, epsilon, max_steps);
    let sarsa_time = bench_runtime(env, &sarsa_algo, episodes, seed, num_seeds);
    results.push(("SARSA".to_owned(), sarsa_time.as_secs_f64()));

    // Q-Learning(lambda)
    let q_lambda_algo =
        QLearningLambda::new(alpha, epsilon, lambda, max_steps, Trace::Accumulating);
    let q_lambda_time = bench_runtime(env, &q_lambda_algo, episodes, seed, num_seeds);
    results.push(("Q-Learning(lambda)".to_owned(), q_lambda_time.as_secs_f64()));

    // Q-Learning(lambda)
    let sarsa_lambda_algo =
        SarsaLambda::new(alpha, epsilon, lambda, max_steps, Trace::Accumulating);
    let sarsa_lambda_time = bench_runtime(env, &sarsa_lambda_algo, episodes, seed, num_seeds);
    results.push(("SARSA(lambda)".to_owned(), sarsa_lambda_time.as_secs_f64()));

    // DynaQ
    let mut dyna_q_algo = DynaQ::new(alpha, epsilon, k, max_steps, deterministic, true, env);
    let dyna_q_time = bench_runtime_dyna(env, &mut dyna_q_algo, episodes, seed, num_seeds);
    results.push(("DynaQ".to_owned(), dyna_q_time.as_secs_f64()));

    //
    // let mut _rng = ChaCha20Rng::seed_from_u64(seed);
    // let _algo = ::new(alpha, epsilon, max_steps);
    // let _time = bench_runtime(env, &_algo, episodes, &mut _rng);
    // results.push(("", _time));
    results
}

fn bench_runtime_algo_random_mdp(
    algo: &impl GenericStateActionAlgorithm,
    episodes: usize,
    seed: u64,
    iterations: usize,
    num_seeds: usize,
) -> Duration {
    let mut total_duration: Duration = Duration::new(0, 0);

    for i in 0..num_seeds {
        let mut mdp_rng = ChaCha20Rng::seed_from_u64(seed + i as u64);
        for _ in 0..iterations {
            let mdp = generate_random_mdp(5, 2, 1, (2, 2), (1, 3), (-1.0, 10.0), &mut mdp_rng);
            let mut algo_rng = ChaCha20Rng::seed_from_u64(seed + i as u64);
            let start = Instant::now();
            algo.run(&mdp, episodes, &mut algo_rng);
            total_duration = total_duration.saturating_add(start.elapsed());
        }
    }

    total_duration.div_f64(num_seeds as f64)
}

fn bench_runtime_algo_random_mdp_dyna(
    algo: &mut DynaQ<IndexState, IndexAction>,
    episodes: usize,
    seed: u64,
    iterations: usize,
    num_seeds: usize,
) -> Duration {
    let mut total_duration: Duration = Duration::new(0, 0);

    for i in 0..num_seeds {
        let mut mdp_rng = ChaCha20Rng::seed_from_u64(seed + i as u64);
        for _ in 0..iterations {
            let mdp = generate_random_mdp(5, 2, 1, (2, 2), (1, 3), (-1.0, 10.0), &mut mdp_rng);
            let mut algo_rng = ChaCha20Rng::seed_from_u64(seed + i as u64);
            algo.clear_model();
            let start = Instant::now();
            algo.run(&mdp, episodes, &mut algo_rng);
            total_duration = total_duration.saturating_add(start.elapsed());
        }
    }

    total_duration.div_f64(num_seeds as f64)
}

fn bench_all_algo_random_mdp(
    episodes: usize,
    seed: u64,
    iterations: usize,
    num_seeds: usize,
    deterministic: bool,
) -> Vec<(String, f64)> {
    // some general parameters used by all algos
    let alpha = 0.1;
    let epsilon = 0.1;
    let lambda = 0.9;
    let k = 5;
    let max_steps = 1000;
    let mut results: Vec<(String, f64)> = vec![];

    // monte carlo
    let mc_algo = MonteCarlo::new(epsilon, max_steps);
    let mc_time = bench_runtime_algo_random_mdp(&mc_algo, episodes, seed, iterations, num_seeds);
    results.push(("MC".to_owned(), mc_time.as_secs_f64()));

    // Q-Learning
    let q_algo = QLearning::new(alpha, epsilon, max_steps);
    let q_time = bench_runtime_algo_random_mdp(&q_algo, episodes, seed, iterations, num_seeds);
    results.push(("Q-Learning".to_owned(), q_time.as_secs_f64()));

    // SARSA
    let sarsa_algo = Sarsa::new(alpha, epsilon, max_steps);
    let sarsa_time =
        bench_runtime_algo_random_mdp(&sarsa_algo, episodes, seed, iterations, num_seeds);
    results.push(("SARSA".to_owned(), sarsa_time.as_secs_f64()));

    // Q-Learning(lambda)
    let q_lambda_algo =
        QLearningLambda::new(alpha, epsilon, lambda, max_steps, Trace::Accumulating);
    let q_lambda_time =
        bench_runtime_algo_random_mdp(&q_lambda_algo, episodes, seed, iterations, num_seeds);
    results.push(("Q-Learning(lambda)".to_owned(), q_lambda_time.as_secs_f64()));

    // Q-Learning(lambda)
    let sarsa_lambda_algo =
        SarsaLambda::new(alpha, epsilon, lambda, max_steps, Trace::Accumulating);
    let sarsa_lambda_time =
        bench_runtime_algo_random_mdp(&sarsa_lambda_algo, episodes, seed, iterations, num_seeds);
    results.push(("SARSA(lambda)".to_owned(), sarsa_lambda_time.as_secs_f64()));

    // DynaQ
    let mut mdp_rng = ChaCha20Rng::seed_from_u64(seed);
    let mdp = generate_random_mdp(5, 2, 1, (2, 2), (1, 3), (-1.0, 10.0), &mut mdp_rng);
    let mut dyna_q_algo = DynaQ::new(alpha, epsilon, k, max_steps, deterministic, true, &mdp);
    let dyna_q_time =
        bench_runtime_algo_random_mdp_dyna(&mut dyna_q_algo, episodes, seed, iterations, num_seeds);
    results.push(("DynaQ".to_owned(), dyna_q_time.as_secs_f64()));

    //
    // let mut _rng = ChaCha20Rng::seed_from_u64(seed);
    // let _algo = ::new(alpha, epsilon, max_steps);
    // let _time = bench_runtime(env, &_algo, episodes, &mut _rng);
    // results.push(("", _time));
    results
}

pub fn bench_environment<M: GenericMdp<S, A>, S: GenericState, A: GenericAction>(
    env: &M,
    episodes: usize,
    seed: u64,
    num_seeds: usize,
    deterministic: bool,
) -> Vec<(String, f64)> {
    let algo_results = bench_all_algo(env, episodes, seed, num_seeds, deterministic);
    algo_results
}
// pub fn bench_runtime_all_env() -> Vec<(String, Vec<f64>)> {
pub fn bench_runtime_all_env() {
    let episodes: usize = 200;
    let seed: u64 = 0;
    let iterations: usize = 100;
    let num_seeds: usize = 100;

    // let results: Vec<(String, Vec<f64>)> = vec![];
    let mut results: HashMap<&str, Vec<f64>> = HashMap::new();

    // regular cliff walking
    let cw_mdp = crate::envs::cliff_walking::build_mdp().unwrap();
    let cw_results = bench_environment(&cw_mdp, episodes, seed, num_seeds, true);

    cw_results.iter().for_each(|(algo, time)| {
        results
            .entry(algo)
            .and_modify(|vec| vec.push(*time))
            .or_insert(vec![*time]);
    });

    // slippery cliff walking
    let scw_mdp = crate::envs::slippery_cliff_walking::build_mdp(0.1).unwrap();
    let scw_results = bench_environment(&scw_mdp, episodes, seed, num_seeds, false);

    scw_results.iter().for_each(|(algo, time)| {
        results
            .entry(algo)
            .and_modify(|vec| vec.push(*time))
            .or_insert(vec![*time]);
    });

    // intersection
    let intersection_mdp = crate::envs::my_intersection::MyIntersectionMdp::new(0.6, 0.2, 10);
    let intersection_results =
        bench_environment(&intersection_mdp, episodes, seed, num_seeds, false);

    intersection_results.iter().for_each(|(algo, time)| {
        results
            .entry(algo)
            .and_modify(|vec| vec.push(*time))
            .or_insert(vec![*time]);
    });

    // random mdps
    let random_mdp_results =
        bench_all_algo_random_mdp(episodes, seed, iterations, num_seeds, false);
    random_mdp_results.iter().for_each(|(algo, time)| {
        results
            .entry(algo)
            .and_modify(|vec| vec.push(*time))
            .or_insert(vec![*time]);
    });

    let mut csv_writer = csv::Writer::from_path("results/runtime.csv").expect("csv file error");
    csv_writer
        .write_record([
            "Algorithm",
            "Cliff Walking",
            "Slippery Cliff Walking",
            "Intersection",
            "Arbitrary MDPs",
        ])
        .expect("csv write record error");

    results.iter().for_each(|(col, value)| {
        csv_writer.serialize((col, value)).expect("csv error");
    });
    println!("Results: {:?}", results);
}

fn write_result_to_csv(results: &Vec<(String, f64)>) {
    let mut csv_writer = csv::Writer::from_path("results/runtime.csv").expect("csv file error");
    csv_writer
        .write_record(["algorithm", "runtime"])
        .expect("csv write recored error");

    results.iter().for_each(|(col, value)| {
        csv_writer.serialize((col, *value)).expect("csv error");
    });
}

// fn bench_runtime_stateful<M: GenericMdp<S, A>, S: GenericState, A: GenericAction>(
//     env: &M,
//     algo: &impl GenericStateActionAlgorithmStateful,
//     episodes: usize,
//     rng: &mut impl Rng,
// ) -> Duration {
//     let mut total_duration: Duration = Duration::new(0, 0);
//     let start = Instant::now();
//     algo.run(env, episodes, rng);
//     total_duration = total_duration.saturating_add(start.elapsed());
//     total_duration
// }
