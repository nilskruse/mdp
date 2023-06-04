use rand::SeedableRng;

use crate::{
    algorithms::{sarsa::Sarsa, GenericStateActionAlgorithm},
    envs,
};

pub mod cliff_walking;

pub fn vis_test() {
    let cliff_walking_mdp = envs::cliff_walking::build_mdp().unwrap();

    let learning_episodes = 500;

    // run "indefinitely"
    let learning_max_steps = usize::MAX;

    let alpha = 0.1;
    let epsilon = 0.1;

    let mut csv_writer = csv::Writer::from_path("cliff_walking.csv").expect("csv error");
    csv_writer
        .write_record(["policy", "avg_reward"])
        .expect("csv error");

    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0);

    let algo = Sarsa::new(alpha, epsilon, learning_max_steps);
    let q_map = algo.run(&cliff_walking_mdp, learning_episodes, &mut rng);

    cliff_walking::show_strategy(&cliff_walking_mdp, &q_map).expect("some gui error");
}
