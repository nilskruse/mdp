use mdp::algorithms::GenericStateActionAlgorithm;
use mdp::eval::{
    evaluate_epsilon_greedy_policy, evaluate_epsilon_greedy_policy_generic,
    evaluate_greedy_policy_generic,
};
use mdp::{algorithms::q_learning::QLearningGeneric, envs, experiments, visualisation};
use rand::SeedableRng;
fn main() {
    // run_benchmarks();
    // experiments::cliff_walking::run_cliff_walking();
    // experiments::cliff_walking::run_slippery_cliff_walking();
    // experiments::cliff_walking::run_cliff_walking_episodic();
    // test();
    // visualisation::vis_test();
    // experiments::non_contractive::run_experiment();
    // experiments::q_learning_dynamic::run_experiment();
    // envs::intersection::build_mdp(3);
    let generic_mdp = envs::cliff_walking::build_generic_mdp().unwrap();
    let generic_q_learning = QLearningGeneric::new(0.1, 0.9, 0.1, usize::MAX);
    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0);
    let q_map = generic_q_learning.run(&generic_mdp, 500, &mut rng);

    let avg_reward_epsilon = evaluate_epsilon_greedy_policy_generic(
        &generic_mdp,
        &q_map,
        500,
        usize::MAX,
        0.1,
        &mut rng,
    );
    let avg_reward_greedy =
        evaluate_greedy_policy_generic(&generic_mdp, &q_map, 500, usize::MAX, &mut rng);
    println!("{:#?}", generic_mdp);
    println!("{:?}", avg_reward_epsilon);
    println!("{:?}", avg_reward_greedy);
}
