mod algorithms;
mod mdp;
mod policies;

use crate::algorithms::{sarsa, value_iteration};
use crate::mdp::*;
use crate::policies::greedy_policy;

fn main() {
    let mdp = Mdp::new_test_mdp();
    let value_map = value_iteration(&mdp, 0.01, 0.0);

    mdp.perform_action((State(0), Action(0)));

    for (state, value) in value_map.iter() {
        println!("State {:?} has value: {:.4}", state, value);
    }

    let q_map = sarsa(&mdp, 0.5, 0.5, (State(0), Action(0)));
    println!("Q: {:?}", q_map);
    let greedy_action = greedy_policy(&mdp, &q_map, State(0));
    println!("Selected greedy_action for State 0: {:?}", greedy_action);
    let greedy_action = greedy_policy(&mdp, &q_map, State(1));
    println!("Selected greedy_action for State 1: {:?}", greedy_action);

    // for _ in 0..100 {
    //     println!("Result: {:?}", mdp.perform_action((State(1), Action(1))));
    // }
    // println!("{:?}", mdp.transitions);
}
