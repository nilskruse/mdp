use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use crate::{
    algorithms::{
        dyna_q::{Dyna, DynaQ},
        q_learning::QLearning,
        q_learning_lambda::QLearningLambda,
        sarsa_lambda::SarsaLambda,
        GenericStateActionAlgorithm, Trace,
    },
    envs,
    eval::evaluate_greedy_policy,
    mdp::{GenericAction, GenericMdp, GenericState},
};

fn bench_until_optimal<M: GenericMdp<S, A>, S: GenericState, A: GenericAction>(
    env: &M,
    algo: &impl GenericStateActionAlgorithm,
    seed: u64,
    num_seeds: usize,
    optimal_reward: f64,
) -> f64 {
    let mut eval_rng = ChaCha20Rng::seed_from_u64(seed);
    let eval_max_steps = 200;
    let eval_episodes = 10;

    let mut total_episodes = 0;
    for i in 0..num_seeds {
        let mut rng = ChaCha20Rng::seed_from_u64(seed + i as u64);
        let mut q_map = algo.run(env, 1, &mut rng);

        loop {
            let avg_reward =
                evaluate_greedy_policy(env, &q_map, eval_episodes, eval_max_steps, &mut eval_rng);
            // dbg!(total_episodes);
            algo.run_with_q_map(env, 1, &mut rng, &mut q_map);
            if avg_reward == optimal_reward {
                break;
            }
            // println!("episode: {q_counter}");
            total_episodes += 1;
        }
    }
    total_episodes as f64 / num_seeds as f64
}

fn bench_until_optimal_dynaq<M: GenericMdp<S, A>, S: GenericState, A: GenericAction>(
    env: &M,
    algo: &mut DynaQ<S, A>,
    seed: u64,
    num_seeds: usize,
    optimal_reward: f64,
) -> f64 {
    let mut eval_rng = ChaCha20Rng::seed_from_u64(seed + 1);
    let eval_max_steps = 200;
    let eval_episodes = 10;

    let mut total_episodes = 0;
    for i in 0..num_seeds {
        let mut rng = ChaCha20Rng::seed_from_u64(seed + i as u64);
        algo.clear_model();
        let mut q_map = algo.run(env, 1, &mut rng);

        loop {
            let avg_reward =
                evaluate_greedy_policy(env, &q_map, eval_episodes, eval_max_steps, &mut eval_rng);
            algo.run_with_q_map(env, 1, &mut rng, &mut q_map);
            if avg_reward == optimal_reward {
                break;
            }
            // println!("episode: {q_counter}");
            total_episodes += 1;
            // dbg!(i, total_episodes, avg_reward);
        }
    }
    total_episodes as f64 / num_seeds as f64
}

pub fn bench_algos_until_optimal(lambda: f64, trace: Trace) {
    let seed: u64 = 0;
    let num_seeds: usize = 100;
    // let cw_mdp = crate::envs::cliff_walking::build_mdp().unwrap();
    let cw_mdp = crate::envs::grid_world::build_mdp().unwrap();

    let alpha = 0.1;
    let epsilon = 0.1;
    // let lambda = 1.0;
    // let trace = Trace::Replacing;
    let _k = 5;
    let _deterministic = true;
    let max_steps = 500;
    let optimal_reward = -13.0;
    let mut results: Vec<(String, f64)> = vec![];

    // // monte carlo
    // println!("MC");
    // let mc_algo = MonteCarlo::new(epsilon, max_steps);
    // let mc_episodes = bench_until_optimal(&cw_mdp, &mc_algo, seed, num_seeds, optimal_reward);
    // results.push(("MC".to_owned(), mc_episodes));

    // // Q-Learning
    // println!("Q");
    // let q_algo = QLearning::new(alpha, epsilon, max_steps);
    // let q_episodes = bench_until_optimal(&cw_mdp, &q_algo, seed, num_seeds, optimal_reward);
    // results.push(("Q-Learning".to_owned(), q_episodes));

    // // SARSA
    // println!("SARSA");
    // let sarsa_algo = Sarsa::new(alpha, epsilon, max_steps);
    // let sarsa_episodes = bench_until_optimal(&cw_mdp, &sarsa_algo, seed, num_seeds, optimal_reward);
    // results.push(("SARSA".to_owned(), sarsa_episodes));

    // Q-Learning(lambda)
    println!("Q lambda");
    let q_lambda_algo = QLearningLambda::new(alpha, epsilon, lambda, max_steps, trace);
    let q_lambda_episodes =
        bench_until_optimal(&cw_mdp, &q_lambda_algo, seed, num_seeds, optimal_reward);
    results.push(("Q-Learning(lambda)".to_owned(), q_lambda_episodes));

    // Q-Learning(lambda)
    println!("SARSA lambda");
    let sarsa_lambda_algo = SarsaLambda::new(alpha, epsilon, lambda, max_steps, trace);
    let sarsa_lambda_episodes =
        bench_until_optimal(&cw_mdp, &sarsa_lambda_algo, seed, num_seeds, optimal_reward);
    results.push(("SARSA(lambda)".to_owned(), sarsa_lambda_episodes));

    // // DynaQ
    // println!("DynaQ");
    // let mut dyna_q_algo = DynaQ::new(alpha, epsilon, k, max_steps, deterministic, true, &cw_mdp);
    // let dyna_q_episodes = bench_until_optimal_dynaq(&cw_mdp, &mut dyna_q_algo, seed, num_seeds, optimal_reward);
    // results.push(("DynaQ".to_owned(), dyna_q_episodes));

    //
    // let mut _rng = ChaCha20Rng::seed_from_u64(seed);
    // let _algo = ::new(alpha, epsilon, max_steps);
    // let _time = bench_runtime(env, &_algo, episodes, &mut _rng);
    // results.push(("", _time));
    dbg!(&results);
    let mut path =
        format!("results/grid_world_optimal_alpha{alpha}_epsilon{epsilon}_lambda{lambda}_{trace}");
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

pub fn run_benchmark() {
    let trace = Trace::Replacing;
    for i in 0..=10 {
        let lambda = i as f64 * 0.1;
        dbg!(lambda, trace);
        bench_algos_until_optimal(lambda, trace);
    }

    let trace = Trace::Accumulating;
    for i in 0..=10 {
        let lambda = i as f64 * 0.1;
        dbg!(lambda, trace);
        bench_algos_until_optimal(lambda, trace);
    }

    let trace = Trace::Dutch;
    for i in 0..=10 {
        let lambda = i as f64 * 0.1;
        dbg!(lambda, trace);
        bench_algos_until_optimal(lambda, trace);
    }
}

pub fn grid_world() {
    let mut rng = ChaCha20Rng::seed_from_u64(0);
    let alpha = 0.1;
    let epsilon = 0.1;
    let max_steps = 1000;
    let mdp = envs::grid_world::build_mdp().unwrap();
    let q_algo = QLearning::new(alpha, epsilon, max_steps);
    let q_map = q_algo.run(&mdp, 1000, &mut rng);
    let avg_reward = evaluate_greedy_policy(&mdp, &q_map, 10, 1000, &mut rng);
    dbg!(avg_reward);
}
