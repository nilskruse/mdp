use mdp::experiments;

fn main() {
    // mdp::benchmarks::run_benchmarks();
    // experiments::cliff_walking::run_cliff_walking();
    // experiments::cliff_walking::run_slippery_cliff_walking();
    // experiments::cliff_walking::run_cliff_walking_episodic();
    // mdp::visualisation::vis_test();
    experiments::non_contractive::run_experiment();
    // experiments::q_learning_dynamic::run_experiment();
    // experiments::intersection::run_experiment();
    // experiments::q_learning_beta::run_q_beta_experiment();
    // experiments::q_learning_beta::run_equivalence_experiment();
}
