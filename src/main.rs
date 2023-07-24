use itertools::iproduct;
use mdp::{
    envs::{self, my_intersection::LightState},
    experiments, multiagent,
};

fn main() {
    // mdp::benchmarks::run_benchmarks();
    // experiments::cliff_walking::run_cliff_walking();
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
    let lightstates = [
        LightState::NorthSouthOpen,
        LightState::EastWestOpen,
        LightState::ChangingToNS,
        LightState::ChangingToEW,
    ];
    let car_iter = 0..=5;
    let iter = iproduct!(
        lightstates.iter(),
        lightstates.iter(),
        car_iter.clone().into_iter(),
        car_iter.clone().into_iter(),
        car_iter.clone().into_iter(),
        car_iter.clone().into_iter()
    );

    iter.clone().for_each(|t| {
        println!("{:?}", t);
    });
    println!("state count: {:?}", iter.clone().count());

    envs::my_intersection::MyIntersectionMdp::new(0.5, 0.5, 10);
    multiagent::intersection::MyIntersectionMdp::new(0.5, 0.5, 0.5, 0.5, 10);
}
