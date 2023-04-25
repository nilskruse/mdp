use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use crate::{
    algorithms::{q_learning, sarsa},
    generator::generate_random_mdp,
    mdp::{Action, State},
};

extern crate test;

const BENCH_EPISODES: usize = 2000;
const BENCH_MAX_STEPS: usize = 2000; // max steps per episode
const BENCH_ALPHA: f64 = 0.1; // learning rate
const BENCH_GAMMA: f64 = 0.9; // discount factor
const BENCH_SEED: u64 = 0;

#[bench]
fn bench_runtime_q_learning(b: &mut test::Bencher) {
    let mut rng = ChaCha20Rng::seed_from_u64(BENCH_SEED);
    let mdp = generate_random_mdp(5, 2, 1, (2, 2), (1, 3), (-1.0, 10.0), &mut rng);
    b.iter(|| {
        q_learning(
            &mdp,
            BENCH_ALPHA,
            BENCH_GAMMA,
            (State(0), Action(0)),
            BENCH_EPISODES,
            BENCH_MAX_STEPS,
            &mut rng,
        );
    });
}

#[bench]
fn bench_runtime_sarsa(b: &mut test::Bencher) {
    let mut rng = ChaCha20Rng::seed_from_u64(BENCH_SEED);
    let mdp = generate_random_mdp(5, 2, 1, (2, 2), (1, 3), (-1.0, 10.0), &mut rng);
    b.iter(|| {
        sarsa(
            &mdp,
            BENCH_ALPHA,
            BENCH_GAMMA,
            (State(0), Action(0)),
            BENCH_EPISODES,
            BENCH_MAX_STEPS,
            &mut rng,
        )
    });
}
