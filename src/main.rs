use mdp::{envs, experiments, visualisation};

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
    println!("{:#?}", generic_mdp);
}
