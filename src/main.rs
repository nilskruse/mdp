use itertools::iproduct;
use mdp::{
    algorithms::q_learning::QLearning,
    envs::{self, my_intersection::LightState},
    experiments, multiagent,
    utils::print_q_map,
};
use rand::SeedableRng;

fn main() {
    // mdp::benchmarks::run_benchmarks();
    experiments::cliff_walking::run_cliff_walking();
    // experiments::cliff_walking::run_slippery_cliff_walking();
    // experiments::cliff_walking::run_cliff_walking_episodic();
    // mdp::visualisation::vis_test();
    // experiments::non_contractive::run_experiment();
    // test();
    // visualisation::vis_test();
    // experiments::non_contractive::run_experiment();
    // experiments::q_learning_dynamic::run_experiment();
    // experiments::intersection::run_experiment();
    // experiments::q_learning_beta::run_q_beta_experiment();
    // experiments::q_learning_beta::run_equivalence_experiment();
    // let lightstates = [
    //     LightState::NorthSouthOpen,
    //     LightState::EastWestOpen,
    //     LightState::ChangingToNS,
    //     LightState::ChangingToEW,
    // ];
    // let car_iter = 0..=5;
    // let iter = iproduct!(
    //     lightstates.iter(),
    //     lightstates.iter(),
    //     car_iter.clone().into_iter(),
    //     car_iter.clone().into_iter(),
    //     car_iter.clone().into_iter(),
    //     car_iter.clone().into_iter()
    // );

    // iter.clone().for_each(|t| {
    //     println!("{:?}", t);
    // });
    // println!("state count: {:?}", iter.clone().count());

    // envs::my_intersection::MyIntersectionMdp::new(0.5, 0.5, 10);
    // multiagent::intersection::MAIntersectionMdp::new(0.5, 0.5, 0.5, 0.5, 10);
    let q_algo_1 = QLearning::new(0.1, 0.1, 2000);
    let q_algo_2 = QLearning::new(0.1, 0.1, 2000);

    let runner = multiagent::intersection::MAIntersectionRunner::new(
        0.2, 0.2, 0.2, 0.2, 1, q_algo_1, q_algo_2, 2000,
    );

    let (mut q_map_1, mut q_map_2) = runner.gen_q_maps();

    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0);
    print_q_map(&q_map_1);
    runner.run(1, &mut q_map_1, &mut q_map_2, &mut rng);
    // print_q_map(&q_map_2);
}
