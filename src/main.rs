use clap::Command;
use mdp::{
    algorithms::Trace,
    benchmarks,
    experiments::{self},
    visualisation,
};

fn cli() -> Command {
    Command::new("mdp")
        .about("MDP tool")
        .subcommand_required(false)
        .arg_required_else_help(true)
        .allow_external_subcommands(true)
        .subcommand(
            Command::new("experiment")
                .about("Run experiments")
                .arg_required_else_help(true)
                .subcommand(
                    Command::new("noncontractive")
                        .about("Tests various algorithms on non-contractive mdp"),
                )
                .subcommand(
                    Command::new("multiagent_single")
                        .about("Run single-agent RL on multi-agent intersection environment"),
                )
                .subcommand(
                    Command::new("multiagent_agent_aware")
                        .about("Run agent-aware RL on multi-agent intersection environment"),
                ),
        )
        .subcommand(
            Command::new("bench")
                .about("Run benchmarks and write results to CSV files")
                .arg_required_else_help(true)
                .subcommand(Command::new("runtime").about("Run runtime benchmarks"))
                .subcommand(
                    Command::new("optimal_episodes")
                        .about("Run episodes required for optimal policy benchmarks"),
                )
                .subcommand(
                    Command::new("intersection")
                        .about("Run strategy comparison benchmark on intersection environment"),
                ),
        )
        .subcommand(
            Command::new("visual")
                .about("Run benchmarks and write results to CSV files")
                .arg_required_else_help(true)
                .subcommand(
                    Command::new("multiagent_intersection")
                        .about("Run visualisation of multi-agent intersection"),
                )
                .subcommand(
                    Command::new("cliff_walking")
                        .about("Run Q-Learning on cliff walking environment and visualize policy"),
                ),
        )
}

fn main() {
    let matches = cli().get_matches();

    match matches.subcommand() {
        Some(("experiment", experiment)) => match experiment.subcommand() {
            Some(("noncontractive", _)) => experiments::non_contractive::run_experiment(),
            Some(("multiagent_single", _)) => experiments::multiagent::regular_rl(),
            Some(("multiagent_agent_aware", _)) => experiments::multiagent::single_agent_rl(),
            _ => println!("Invalid command."),
        },
        Some(("bench", benchmark)) => match benchmark.subcommand() {
            Some(("runtime", _)) => benchmarks::runtime::bench_runtime_all_env(),
            Some(("optimal_episodes", _)) => benchmarks::optimal_episodes::run_benchmark(),
            Some(("intersection", _)) => benchmarks::strategies::compare_intersection(),
            _ => println!("Invalid command."),
        },
        Some(("visual", vis)) => match vis.subcommand() {
            Some(("multiagent_intersection", _)) => visualisation::ma_intersection::main().unwrap(),
            Some(("cliff_walking", _)) => visualisation::vis_test(),
            _ => println!("Invalid command."),
        },
        Some((_, _)) => {
            println!("Invalid command.");
        }
        _ => panic!("we should not reach this"),
    }
}
