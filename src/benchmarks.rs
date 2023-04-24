use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use crate::{
    algorithms::{q_learning, sarsa},
    generator::generate_random_mdp,
    mdp::{Action, State},
};

extern crate test;
// #[allow(soft_unstable)]
#[bench]
fn bench_runtime_q_learning(b: &mut test::Bencher) {
    let mut rng = ChaCha20Rng::seed_from_u64(1);
    let mdp = generate_random_mdp(5, 2, 1, (2, 2), (1, 3), (-1.0, 10.0), &mut rng);
    b.iter(|| {
        let mut rng = ChaCha20Rng::seed_from_u64(1);
        q_learning(&mdp, 0.1, 0.9, (State(0), Action(0)), 100, 2000, &mut rng);
    });
}

#[bench]
fn bench_runtime_sarsa(b: &mut test::Bencher) {
    let mut rng = ChaCha20Rng::seed_from_u64(1);
    let mdp = generate_random_mdp(5, 2, 1, (2, 2), (1, 3), (-1.0, 10.0), &mut rng);
    b.iter(|| {
        let mut rng = ChaCha20Rng::seed_from_u64(1);
        sarsa(&mdp, 0.1, 0.9, (State(0), Action(0)), 100, 2000, &mut rng)
    });
}
