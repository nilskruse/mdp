use mdp::{experiments, visualisation};

fn main() {
    // run_benchmarks();
    experiments::cliff_walking::run_cliff_walking();
    // experiments::cliff_walking::run_slippery_cliff_walking();
    // experiments::cliff_walking::run_cliff_walking_episodic();
    // test();
    visualisation::vis_test();
}
