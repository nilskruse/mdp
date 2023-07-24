use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use crate::{
    algorithms::{
        dyna_q::{Dyna, DynaQ},
        q_learning::QLearning,
        q_learning_beta::QLearningBeta,
        GenericStateActionAlgorithm,
    },
    eval::evaluate_greedy_policy,
    experiments::non_contractive::{QLearningClipped, RiggedStateActionAlgorithm},
    mdp::{GenericAction, GenericMdp, GenericState, IndexAction, IndexState},
    utils::{print_q_map, print_transition_map},
};

pub fn run_q_beta_experiment() {
    // mdp with unlikely high reward transition
    let mdp = crate::experiments::non_contractive::build_mdp(0.001);
    // force this transition once in the beginning
    let rig = Some((IndexState(2), IndexAction(0)));

    println!("Q-Learning Beta");
    let mut q_beta_algo = QLearningBeta::new(0.1, 0.2, usize::MAX, 10);
    let mut rng = ChaCha20Rng::seed_from_u64(0);
    let q_map = q_beta_algo.run(&mdp, 2000, &mut rng, rig);
    println!("Q-Table:");
    print_q_map(&q_map);
    println!();

    println!("\nTransitions");
    print_transition_map(&mdp);
    println!();
}

pub fn run_equivalence_experiment() {
    let mdp = crate::envs::cliff_walking::build_mdp().unwrap();
    let mdp = crate::envs::slippery_cliff_walking::build_mdp(0.2).unwrap();

    let n = 100;
    let mut total_q = 0.0;
    let mut total_q_beta = 0.0;
    let mut total_q_clipped = 0.0;
    let mut total_dyna_q = 0.0;

    let optimal_reward = -14.0;

    for seed in 0..n {
        let (q_eps, q_beta_eps, q_clipped_eps, dyna_q_eps) =
            run_equivalence_experiment_seed(&mdp, seed, optimal_reward);
        total_q += q_eps as f64;
        total_q_beta += q_beta_eps as f64;
        total_q_clipped += q_clipped_eps as f64;
        total_dyna_q += dyna_q_eps as f64;

        if seed % 10 == 0 {
            println!("Run: {seed}");
        }
    }

    total_q /= n as f64;
    total_q_beta /= n as f64;
    total_q_clipped /= n as f64;
    total_dyna_q /= n as f64;

    println!();
    println!("Q average episodes for optimal greedy policy: {total_q}");
    println!("Q-beta average episodes for optimal greedy policy: {total_q_beta}");
    println!("Q-clipped average episodes for optimal greedy policy: {total_q_clipped}");
    println!("DynaQ average episodes for optimal greedy policy: {total_dyna_q}");
}

pub fn run_equivalence_experiment_seed<M, S, A>(mdp: &M, seed: u64, optimal_reward: f64) -> (usize, usize, usize, usize)
where
    M: GenericMdp<S, A>,
    S: GenericState,
    A: GenericAction,
{
    let alpha = 0.1;
    let epsilon = 0.1;
    let max_steps = 200;
    let beta_rate = 10;
    let k = 10;

    let eval_episodes = 1;

    let q_algo = QLearning::new(alpha, epsilon, max_steps);
    let mut q_beta_algo = QLearningBeta::new(alpha, epsilon, max_steps, beta_rate);
    let mut q_clipped_algo = QLearningClipped::new(alpha, epsilon, max_steps, 50.0);
    let mut dyna_q_algo = DynaQ::new(alpha, epsilon, k, max_steps);

    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let mut eval_rng = ChaCha20Rng::seed_from_u64(0);
    let mut q_map = q_algo.run(mdp, 1, &mut rng);

    let mut q_counter = 1;

    loop {
        let avg_reward =
            evaluate_greedy_policy(mdp, &q_map, eval_episodes, max_steps, &mut eval_rng);
        q_algo.run_with_q_map(mdp, 1, &mut rng, &mut q_map);
        if avg_reward == optimal_reward {
            break;
        }
        // println!("episode: {q_counter}");
        q_counter += 1;
    }
    // println!("Q-Learning found optimal strategy after {q_counter} steps");

    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let mut eval_rng = ChaCha20Rng::seed_from_u64(0);
    let mut q_map = q_beta_algo.run(mdp, 1, &mut rng, None);

    let mut q_beta_counter = 1;

    loop {
        let avg_reward =
            evaluate_greedy_policy(mdp, &q_map, eval_episodes, max_steps, &mut eval_rng);
        q_beta_algo.run_with_q_map(mdp, 1, &mut rng, &mut q_map, None);
        if avg_reward == optimal_reward {
            break;
        }
        // println!("beta episode: {q_beta_counter}");
        q_beta_counter += 1;
    }
    // println!("Q-Learning-beta found optimal strategy after {q_beta_counter} steps");

    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let mut eval_rng = ChaCha20Rng::seed_from_u64(0);
    let mut q_map = q_clipped_algo.run(mdp, 1, &mut rng, None);

    let mut q_clipped_counter = 1;

    loop {
        let avg_reward =
            evaluate_greedy_policy(mdp, &q_map, eval_episodes, max_steps, &mut eval_rng);
        q_clipped_algo.run_with_q_map(mdp, 1, &mut rng, &mut q_map, None);
        if avg_reward == optimal_reward {
            break;
        }
        q_clipped_counter += 1;
    }

    // println!("dynaq");
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let mut eval_rng = ChaCha20Rng::seed_from_u64(0);
    let (mut q_map, mut model) = dyna_q_algo.run(mdp, 1, &mut rng);

    let mut dyna_q_counter = 1;

    loop {
        let avg_reward =
            evaluate_greedy_policy(mdp, &q_map, eval_episodes, max_steps, &mut eval_rng);
        // println!("dyna_q_counter: {dyna_q_counter}, avg_reward: {avg_reward}");
        dyna_q_algo.run_with_q_map(mdp, 1, &mut rng, &mut q_map, &mut model);
        if avg_reward == optimal_reward {
            break;
        }
        dyna_q_counter += 1;
    }
    // println!("Q-Learning-clipped found optimal strategy after {q_beta_counter} steps");
    (q_counter, q_beta_counter, q_clipped_counter, dyna_q_counter)
}
