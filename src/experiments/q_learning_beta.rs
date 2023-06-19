use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use crate::{
    algorithms::q_learning_beta::QLearningBeta,
    experiments::non_contractive::RiggedStateActionAlgorithm,
    mdp::{IndexAction, IndexState},
    utils::{print_q_map, print_transition_map},
};

pub fn run_q_beta_experiment() {
    // mdp with unlikely high reward transition
    let mdp = crate::experiments::non_contractive::build_mdp(0.001);
    // force this transition once in the beginning
    let rig = (IndexState(2), IndexAction(0));

    println!("Q-Learning Beta");
    let q_beta_algo = QLearningBeta::new(0.1, 0.2, usize::MAX, 100);
    let mut rng = ChaCha20Rng::seed_from_u64(0);
    let q_map = q_beta_algo.run(&mdp, 2000, &mut rng, rig);
    println!("Q-Table:");
    print_q_map(&q_map);
    println!();

    println!("\nTransitions");
    print_transition_map(&mdp);
    println!();
}
