use mdp::algorithms::GenericStateActionAlgorithm;
use mdp::eval::{evaluate_epsilon_greedy_policy, evaluate_greedy_policy};
use mdp::mdp::GenericMdp;
use mdp::{algorithms::q_learning::QLearning, envs, experiments};
use rand::SeedableRng;

fn main() {
    // run_benchmarks();
    experiments::cliff_walking::run_cliff_walking();
    experiments::cliff_walking::run_slippery_cliff_walking();
    // experiments::cliff_walking::run_cliff_walking_episodic();
    // test();
    // visualisation::vis_test();
    // experiments::non_contractive::run_experiment();
    // experiments::q_learning_dynamic::run_experiment();
    experiments::intersection::run_experiments();
}
