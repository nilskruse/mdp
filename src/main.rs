mod algorithms;
mod mdp;
mod policies;

use std::collections::HashMap;

use crate::algorithms::{q_learning, sarsa, value_iteration};
use crate::mdp::*;
use crate::policies::greedy_policy;

fn main() {
    let mdp = Mdp::new_test_mdp();
    let value_map = value_iteration(&mdp, 0.01, 0.0);

    mdp.perform_action((State(0), Action(0)));

    for (state, value) in value_map.iter() {
        println!("State {:?} has value: {:.4}", state, value);
    }

    let q_map = sarsa(&mdp, 0.5, 0.5, (State(0), Action(0)), 1000, 2000);
    println!("Q: {:?}", q_map);

    let avg_reward = evaluate_policy(&mdp, q_map, 1000);
    println!("sarsa average reward: {avg_reward}");

    let q_map = q_learning(&mdp, 0.5, 0.5, (State(0), Action(0)), 1000, 2000);
    println!("Q: {:?}", q_map);

    let avg_reward = evaluate_policy(&mdp, q_map, 1000);
    println!("Q-learning average reward: {avg_reward}");
    // let greedy_action = greedy_policy(&mdp, &q_map, State(0));
    // println!("Selected greedy_action for State 0: {:?}", greedy_action);
    // let greedy_action = greedy_policy(&mdp, &q_map, State(1));
    // println!("Selected greedy_action for State 1: {:?}", greedy_action);

    // for _ in 0..100 {
    //     println!("Result: {:?}", mdp.perform_action((State(1), Action(1))));
    // }
    // println!("{:?}", mdp.transitions);
}

fn evaluate_policy(mdp: &Mdp, q_map: HashMap<(State, Action), f64>, episodes: u64) -> f64 {
    let mut total_reward = 0.0;
    for _ in 1..=episodes {
        let mut current_state = mdp.initial_state;
        let mut episode_reward = 0.0;
        while !mdp.terminal_states.contains(&current_state) {
            let selected_action = greedy_policy(mdp, &q_map, current_state);
            let (next_state, reward) = mdp.perform_action((current_state, selected_action));
            episode_reward += reward;
            current_state = next_state;
        }
        total_reward += episode_reward;
    }
    total_reward / episodes as f64
}
