use crate::algorithms::GenericStateActionAlgorithm;
use std::time::{Duration, Instant};

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use crate::{
    algorithms::{sarsa::Sarsa, StateActionAlgorithm},
    generator::generate_random_mdp,
};

use crate::algorithms::q_learning::QLearning;

const BENCH_EPISODES: usize = 1000;
const BENCH_MAX_STEPS: usize = 2000; // max steps per episode
const BENCH_ALPHA: f64 = 0.1; // learning rate
const BENCH_GAMMA: f64 = 0.9; // discount factor
const BENCH_EPSILON: f64 = 0.1; // learning rate
const BENCH_SEED: u64 = 2;

const BENCH_ITERATIONS: usize = 1000;

pub fn run_benchmarks() {
    println!("Starting Benchmarks...");

    println!("Benching Q-Learning...");
    let q_learning_time = bench_runtime_q_learning();

    println!("Benching SARSA...");
    let sarsa_time = bench_runtime_sarsa();

    let mut csv_writer = csv::Writer::from_path("runtime.csv").expect("csv error");
    csv_writer
        .write_record(["algorithm", "runtime"])
        .expect("csv error");

    csv_writer
        .serialize(("Q-Learning", q_learning_time.as_secs_f64()))
        .expect("csv error");
    csv_writer
        .serialize(("SARSA", sarsa_time.as_secs_f64()))
        .expect("csv error");
}

fn bench_runtime_q_learning() -> Duration {
    let mut mdp_rng = ChaCha20Rng::seed_from_u64(BENCH_SEED);
    let mut total_duration: Duration = Duration::new(0, 0);
    let algo = QLearning::new(BENCH_ALPHA, BENCH_GAMMA, BENCH_EPSILON, BENCH_MAX_STEPS);

    for _ in 0..BENCH_ITERATIONS {
        let mdp = generate_random_mdp(5, 2, 1, (2, 2), (1, 3), (-1.0, 10.0), &mut mdp_rng);
        let mut algo_rng = ChaCha20Rng::seed_from_u64(BENCH_SEED);
        let start = Instant::now();

        algo.run(&mdp, BENCH_EPISODES, &mut algo_rng);
        total_duration = total_duration.saturating_add(start.elapsed());
    }

    println!("q_learning total_duration: {:?}", total_duration);
    total_duration
}

fn bench_runtime_sarsa() -> Duration {
    let mut mdp_rng = ChaCha20Rng::seed_from_u64(BENCH_SEED);
    let mut total_duration: Duration = Duration::new(0, 0);
    let algo = Sarsa::new(BENCH_ALPHA, BENCH_GAMMA, BENCH_EPSILON, BENCH_MAX_STEPS);

    for _ in 0..BENCH_ITERATIONS {
        let mdp = generate_random_mdp(5, 2, 1, (2, 2), (1, 3), (-1.0, 10.0), &mut mdp_rng);
        let mut algo_rng = ChaCha20Rng::seed_from_u64(BENCH_SEED);
        let start = Instant::now();
        algo.run(&mdp, BENCH_EPISODES, &mut algo_rng);
        total_duration = total_duration.saturating_add(start.elapsed());
    }

    println!("sarsa total_duration: {:?}", total_duration);
    total_duration
}
