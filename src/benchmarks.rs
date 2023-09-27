use crate::algorithms::dyna_q::{Dyna, DynaQ};
use crate::algorithms::monte_carlo::MonteCarlo;
use crate::algorithms::q_learning_lambda::QLearningLambda;
use crate::algorithms::sarsa_lambda::SarsaLambda;
use crate::algorithms::{GenericStateActionAlgorithm, Trace};
use crate::mdp::{GenericAction, GenericMdp, GenericState, IndexAction, IndexState};
use std::collections::HashMap;
use std::time::{Duration, Instant};

use itertools::Itertools;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

use crate::{algorithms::sarsa::Sarsa, generator::generate_random_mdp};

use crate::algorithms::q_learning::QLearning;

const BENCH_EPISODES: usize = 1000;
const BENCH_MAX_STEPS: usize = 2000; // max steps per episode
const BENCH_ALPHA: f64 = 0.1; // learning rate
const BENCH_GAMMA: f64 = 0.9; // discount factor
const BENCH_EPSILON: f64 = 0.1; // learning rate
const BENCH_SEED: u64 = 2;

const BENCH_ITERATIONS: usize = 1000;

/// Bench runtime of algorithm on an mdp given set number of episodes
fn bench_runtime<M: GenericMdp<S, A>, S: GenericState, A: GenericAction>(
    env: &M,
    algo: &impl GenericStateActionAlgorithm,
    episodes: usize,
    rng: &mut impl Rng,
) -> Duration {
    let mut total_duration: Duration = Duration::new(0, 0);
    let start = Instant::now();
    algo.run(env, episodes, rng);
    total_duration = total_duration.saturating_add(start.elapsed());
    total_duration
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

fn bench_runtime_dyna<M: GenericMdp<S, A>, D: Dyna<S, A>, S: GenericState, A: GenericAction>(
    env: &M,
    algo: &mut D,
    episodes: usize,
    rng: &mut impl Rng,
) -> Duration {
    let mut total_duration: Duration = Duration::new(0, 0);
    let start = Instant::now();
    algo.run(env, episodes, rng);
    total_duration = total_duration.saturating_add(start.elapsed());
    total_duration
}

fn bench_all_algo<M: GenericMdp<S, A>, S: GenericState, A: GenericAction>(
    env: &M,
    episodes: usize,
    seed: u64,
) -> Vec<(String, f64)> {
    // some general parameters used by all algos
    let alpha = 0.1;
    let epsilon = 0.1;
    let lambda = 0.1;
    let k = 5;
    let max_steps = 2000;
    let mut results: Vec<(String, f64)> = vec![];

    // monte carlo
    let mut mc_rng = ChaCha20Rng::seed_from_u64(seed);
    let mc_algo = MonteCarlo::new(epsilon, max_steps);
    let mc_time = bench_runtime(env, &mc_algo, episodes, &mut mc_rng);
    results.push(("MC".to_owned(), mc_time.as_secs_f64()));

    // Q-Learning
    let mut q_rng = ChaCha20Rng::seed_from_u64(seed);
    let q_algo = QLearning::new(alpha, epsilon, max_steps);
    let q_time = bench_runtime(env, &q_algo, episodes, &mut q_rng);
    results.push(("Q-Learning".to_owned(), q_time.as_secs_f64()));

    // SARSA
    let mut sarsa_rng = ChaCha20Rng::seed_from_u64(seed);
    let sarsa_algo = Sarsa::new(alpha, epsilon, max_steps);
    let sarsa_time = bench_runtime(env, &sarsa_algo, episodes, &mut sarsa_rng);
    results.push(("SARSA".to_owned(), sarsa_time.as_secs_f64()));

    // Q-Learning(lambda)
    let mut q_lambda_rng = ChaCha20Rng::seed_from_u64(seed);
    let q_lambda_algo =
        QLearningLambda::new(alpha, epsilon, lambda, max_steps, Trace::Accumulating);
    let q_lambda_time = bench_runtime(env, &q_lambda_algo, episodes, &mut q_lambda_rng);
    results.push(("Q-Learning(lambda)".to_owned(), q_lambda_time.as_secs_f64()));

    // Q-Learning(lambda)
    let mut sarsa_lambda_rng = ChaCha20Rng::seed_from_u64(seed);
    let sarsa_lambda_algo =
        SarsaLambda::new(alpha, epsilon, lambda, max_steps, Trace::Accumulating);
    let sarsa_lambda_time = bench_runtime(env, &sarsa_lambda_algo, episodes, &mut sarsa_lambda_rng);
    results.push(("SARSA(lambda)".to_owned(), sarsa_lambda_time.as_secs_f64()));

    // DynaQ
    let mut dyna_q_rng = ChaCha20Rng::seed_from_u64(seed);
    let mut dyna_q_algo = DynaQ::new(alpha, epsilon, max_steps, k, true, true, env);
    let dyna_q_time = bench_runtime_dyna(env, &mut dyna_q_algo, episodes, &mut dyna_q_rng);
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
) -> Duration {
    let mut mdp_rng = ChaCha20Rng::seed_from_u64(seed);
    let mut total_duration: Duration = Duration::new(0, 0);

    for _ in 0..iterations {
        let mdp = generate_random_mdp(5, 2, 1, (2, 2), (1, 3), (-1.0, 10.0), &mut mdp_rng);
        let mut algo_rng = ChaCha20Rng::seed_from_u64(seed);
        let start = Instant::now();
        algo.run(&mdp, episodes, &mut algo_rng);
        total_duration = total_duration.saturating_add(start.elapsed());
    }

    total_duration
}

fn bench_runtime_algo_random_mdp_dyna<D: Dyna<IndexState, IndexAction>>(
    algo: &mut D,
    episodes: usize,
    seed: u64,
    iterations: usize,
) -> Duration {
    let mut mdp_rng = ChaCha20Rng::seed_from_u64(seed);
    let mut total_duration: Duration = Duration::new(0, 0);

    for _ in 0..iterations {
        let mdp = generate_random_mdp(5, 2, 1, (2, 2), (1, 3), (-1.0, 10.0), &mut mdp_rng);
        let mut algo_rng = ChaCha20Rng::seed_from_u64(seed);
        let start = Instant::now();
        algo.run(&mdp, episodes, &mut algo_rng);
        total_duration = total_duration.saturating_add(start.elapsed());
    }

    total_duration
}

fn bench_all_algo_random_mdp(episodes: usize, seed: u64, iterations: usize) -> Vec<(String, f64)> {
    // some general parameters used by all algos
    let alpha = 0.1;
    let epsilon = 0.1;
    let lambda = 0.1;
    let k = 5;
    let max_steps = 2000;
    let mut results: Vec<(String, f64)> = vec![];

    // monte carlo
    let mc_algo = MonteCarlo::new(epsilon, max_steps);
    let mc_time = bench_runtime_algo_random_mdp(&mc_algo, episodes, seed, iterations);
    results.push(("MC".to_owned(), mc_time.as_secs_f64()));

    // Q-Learning
    let q_algo = QLearning::new(alpha, epsilon, max_steps);
    let q_time = bench_runtime_algo_random_mdp(&q_algo, episodes, seed, iterations);
    results.push(("Q-Learning".to_owned(), q_time.as_secs_f64()));

    // SARSA
    let sarsa_algo = Sarsa::new(alpha, epsilon, max_steps);
    let sarsa_time = bench_runtime_algo_random_mdp(&sarsa_algo, episodes, seed, iterations);
    results.push(("SARSA".to_owned(), sarsa_time.as_secs_f64()));

    // Q-Learning(lambda)
    let q_lambda_algo =
        QLearningLambda::new(alpha, epsilon, lambda, max_steps, Trace::Accumulating);
    let q_lambda_time = bench_runtime_algo_random_mdp(&q_lambda_algo, episodes, seed, iterations);
    results.push(("Q-Learning(lambda)".to_owned(), q_lambda_time.as_secs_f64()));

    // Q-Learning(lambda)
    let sarsa_lambda_algo =
        SarsaLambda::new(alpha, epsilon, lambda, max_steps, Trace::Accumulating);
    let sarsa_lambda_time =
        bench_runtime_algo_random_mdp(&sarsa_lambda_algo, episodes, seed, iterations);
    results.push(("SARSA(lambda)".to_owned(), sarsa_lambda_time.as_secs_f64()));

    // DynaQ
    let mut mdp_rng = ChaCha20Rng::seed_from_u64(seed);
    let mdp = generate_random_mdp(5, 2, 1, (2, 2), (1, 3), (-1.0, 10.0), &mut mdp_rng);
    let mut dyna_q_algo = DynaQ::new(alpha, epsilon, max_steps, k, true, true, &mdp);
    let dyna_q_time =
        bench_runtime_algo_random_mdp_dyna(&mut dyna_q_algo, episodes, seed, iterations);
    results.push(("DynaQ".to_owned(), dyna_q_time.as_secs_f64()));

    //
    // let mut _rng = ChaCha20Rng::seed_from_u64(seed);
    // let _algo = ::new(alpha, epsilon, max_steps);
    // let _time = bench_runtime(env, &_algo, episodes, &mut _rng);
    // results.push(("", _time));
    results
}

pub fn bench_slippery_cliff_walking() {
    let mdp = crate::envs::slippery_cliff_walking::build_mdp(0.1).unwrap();
    let results = bench_all_algo(&mdp, 100, 0);
    println!("Results: {:?}", results);
    write_result_to_csv(&results)
}

pub fn bench_environment<M: GenericMdp<S, A>, S: GenericState, A: GenericAction>(
    env: &M,
    episodes: usize,
    seed: u64,
) -> Vec<(String, f64)> {
    let algo_results = bench_all_algo(env, episodes, seed);
    algo_results
}
// pub fn bench_runtime_all_env() -> Vec<(String, Vec<f64>)> {
pub fn bench_runtime_all_env() {
    let episodes: usize = 1000;
    let seed: u64 = 0;
    let iterations: usize = 100;

    // let results: Vec<(String, Vec<f64>)> = vec![];
    let mut results: HashMap<&str, Vec<f64>> = HashMap::new();

    // regular cliff walking
    let cw_mdp = crate::envs::cliff_walking::build_mdp().unwrap();
    let cw_results = bench_environment(&cw_mdp, episodes, seed);

    cw_results.iter().for_each(|(algo, time)| {
        results
            .entry(algo)
            .and_modify(|vec| vec.push(*time))
            .or_insert(vec![*time]);
    });

    // slippery cliff walking
    let scw_mdp = crate::envs::slippery_cliff_walking::build_mdp(0.1).unwrap();
    let scw_results = bench_environment(&scw_mdp, episodes, seed);

    scw_results.iter().for_each(|(algo, time)| {
        results
            .entry(algo)
            .and_modify(|vec| vec.push(*time))
            .or_insert(vec![*time]);
    });

    // intersection
    let intersection_mdp = crate::envs::my_intersection::MyIntersectionMdp::new(0.6, 0.2, 10);
    let intersection_results = bench_environment(&intersection_mdp, episodes, seed);

    intersection_results.iter().for_each(|(algo, time)| {
        results
            .entry(algo)
            .and_modify(|vec| vec.push(*time))
            .or_insert(vec![*time]);
    });

    // random mdps
    let random_mdp_results = bench_all_algo_random_mdp(episodes, seed, iterations);
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
