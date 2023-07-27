use rand::SeedableRng;

use crate::{algorithms::q_learning::QLearning, multiagent::intersection::MAIntersectionRunner};

pub fn main() {
    let max_steps = 5000;
    let q_algo_1 = QLearning::new(0.1, 0.1, max_steps);
    let q_algo_2 = QLearning::new(0.1, 0.1, max_steps);

    let runner = MAIntersectionRunner::new(0.2, 0.2, 0.2, 0.2, 10, q_algo_1, q_algo_2, max_steps);

    let (mut q_map_1, mut q_map_2) = runner.gen_q_maps();

    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0);
    let avg_reward = runner.eval_greedy(100, &q_map_1, &q_map_2, max_steps, &mut rng);
    println!("avg reward before: {}", avg_reward);

    runner.run(1000, &mut q_map_1, &mut q_map_2, &mut rng);

    let avg_reward = runner.eval_greedy(100, &q_map_1, &q_map_2, max_steps, &mut rng);
    println!("avg reward after: {}", avg_reward);

    // print_q_map(&q_map_1);
    // print_q_map(&q_map_2);
}
