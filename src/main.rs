use clap::{arg, Command};
use mdp::{
    experiments::{self, multiagent},
    visualisation,
};

fn cli() -> Command {
    Command::new("mdp")
        .about("MDP tool")
        .subcommand_required(false)
        .arg_required_else_help(false)
        .allow_external_subcommands(true)
        .subcommand(
            Command::new("noncontractive").about("Tests various algorithms on non-contractive mdp"),
        )
}

fn main() {
    let matches = cli().get_matches();

    match matches.subcommand() {
        Some(("noncontractive", _)) => {
            println!("doing stuff");
            experiments::non_contractive::run_experiment();
        }
        Some((_, _)) => {
            println!("Invalid command.");
        }
        _ => default_main(), // If all subcommands are defined above, anything else is unreachable!()
    }
}

fn default_main() {
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
    // experiments::multiagent::main();
    visualisation::ma_intersection::main().unwrap();
}
