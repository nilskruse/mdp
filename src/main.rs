use mdp::algorithms::GenericStateActionAlgorithm;
use mdp::eval::{evaluate_epsilon_greedy_policy, evaluate_greedy_policy};
use mdp::mdp::GenericMdp;
use mdp::{algorithms::q_learning::QLearning, envs, experiments};
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
    // let generic_mdp = envs::cliff_walking::build_mdp().unwrap();
    // let generic_q_learning = QLearning::new(0.1, 0.1, usize::MAX);
    // let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0);
    // let q_map = generic_q_learning.run(&generic_mdp, 1000, &mut rng);

    // let avg_reward_epsilon =
    //     evaluate_epsilon_greedy_policy(&generic_mdp, &q_map, 500, usize::MAX, 0.1, &mut rng);
    // let avg_reward_greedy = evaluate_greedy_policy(&generic_mdp, &q_map, 500, usize::MAX, &mut rng);
    // // println!("{:#?}", generic_mdp);
    // println!("{:?}", avg_reward_epsilon);
    // println!("{:?}", avg_reward_greedy);

    // let generic_mdp = envs::slippery_cliff_walking::build_mdp(0.2).unwrap();
    // let generic_q_learning = QLearning::new(0.1, 0.1, usize::MAX);
    // let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(1);
    // let q_map = generic_q_learning.run(&generic_mdp, 1000, &mut rng);

    // let avg_reward_epsilon =
    //     evaluate_epsilon_greedy_policy(&generic_mdp, &q_map, 500, usize::MAX, 0.1, &mut rng);
    // let avg_reward_greedy = evaluate_greedy_policy(&generic_mdp, &q_map, 500, usize::MAX, &mut rng);
    // // println!("{:#?}", generic_mdp);
    // println!("{:?}", avg_reward_epsilon);
    // println!("{:?}", avg_reward_greedy);
    // experiments::cliff_walking::run_cliff_walking();
    // experiments::cliff_walking::run_slippery_cliff_walking();

    // let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(1);
    // let blackjack_mdp = envs::blackjack::BlackjackMdp::new();

    // for _ in 0..100 {
    //     println!("{:?}", blackjack_mdp.get_initial_state(&mut rng));
    // }
    // let arr1: [usize; 4] = [1, 3, 3, 3];
    // let arr2: [usize; 4] = [1, 2, 3, 4];

    // let compare = arr1 < arr2;
    // println!("{compare}");
    //
    let eval_episodes = 100;
    let train_episodes = 500;
    let generic_mdp = envs::my_intersection::MyIntersectionMdp::new(0.4, 0.2, 10);
    let generic_q_learning = QLearning::new(0.1, 0.1, 2000);
    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(1);
    let mut q_map = generic_q_learning.run(&generic_mdp, 1, &mut rng);

    println!("first episode");
    let avg_reward_epsilon =
        evaluate_epsilon_greedy_policy(&generic_mdp, &q_map, eval_episodes, 2000, 0.1, &mut rng);
    let avg_reward_greedy =
        evaluate_greedy_policy(&generic_mdp, &q_map, eval_episodes, 2000, &mut rng);
    // println!("{:#?}", generic_mdp);
    println!("{:?}", avg_reward_epsilon);
    println!("{:?}", avg_reward_greedy);
    println!("500 episode");
    generic_q_learning.run_with_q_map(&generic_mdp, 499, &mut rng, &mut q_map);
    let avg_reward_epsilon =
        evaluate_epsilon_greedy_policy(&generic_mdp, &q_map, eval_episodes, 2000, 0.1, &mut rng);
    let avg_reward_greedy =
        evaluate_greedy_policy(&generic_mdp, &q_map, eval_episodes, 2000, &mut rng);
    // println!("{:#?}", generic_mdp);
    println!("{:?}", avg_reward_epsilon);
    println!("{:?}", avg_reward_greedy);
}
