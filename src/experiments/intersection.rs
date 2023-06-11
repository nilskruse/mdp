use rand::{Rng, SeedableRng};

use crate::{
    algorithms::{q_learning::QLearning, GenericStateActionAlgorithm},
    envs::{
        self,
        my_intersection::{Action, MyIntersectionMdp},
    },
    eval::{evaluate_epsilon_greedy_policy, evaluate_greedy_policy},
    mdp::GenericMdp,
};

pub fn run_experiments() {
    let eval_episodes = 100;
    let train_episodes = 1000;
    let episode_length = 2000;
    let generic_mdp = MyIntersectionMdp::new(0.2, 0.2, 50);
    let generic_q_learning = QLearning::new(0.1, 0.1, episode_length);
    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(1);
    let mut q_map = generic_q_learning.run(&generic_mdp, 1, &mut rng);

    println!(
        "number of (state, action): {:?}",
        generic_mdp.get_all_state_actions().len()
    );
    println!("first episode");
    let avg_reward_epsilon = evaluate_epsilon_greedy_policy(
        &generic_mdp,
        &q_map,
        eval_episodes,
        episode_length,
        0.1,
        &mut rng,
    );
    let avg_reward_greedy = evaluate_greedy_policy(
        &generic_mdp,
        &q_map,
        eval_episodes,
        episode_length,
        &mut rng,
    );
    // println!("{:#?}", generic_mdp);
    println!("epsilon: {:?}", avg_reward_epsilon);
    println!("greedy: {:?}", avg_reward_greedy);
    println!("{train_episodes} episodes");
    generic_q_learning.run_with_q_map(&generic_mdp, train_episodes, &mut rng, &mut q_map);
    let avg_reward_epsilon = evaluate_epsilon_greedy_policy(
        &generic_mdp,
        &q_map,
        eval_episodes,
        episode_length,
        0.1,
        &mut rng,
    );
    let avg_reward_greedy = evaluate_greedy_policy(
        &generic_mdp,
        &q_map,
        eval_episodes,
        episode_length,
        &mut rng,
    );
    // println!("{:#?}", generic_mdp);
    println!("epsilon: {:?}", avg_reward_epsilon);
    println!("greedy: {:?}", avg_reward_greedy);
    // q_map.iter().for_each(|e| println!("{:?}", e));
    //
    let avg_reward = fixed_cycle(
        &generic_mdp,
        train_episodes,
        episode_length,
        30,
        30,
        &mut rng,
    );
    println!("fixed cycle reward: {:?}", avg_reward);
}

fn fixed_cycle(
    mdp: &MyIntersectionMdp,
    episodes: usize,
    max_steps: usize,
    ns_time: usize,
    ew_time: usize,
    rng: &mut (impl SeedableRng + Rng),
) -> f64 {
    let mut total_reward = 0.0;
    for _ in 0..episodes {
        let mut steps = 0;
        let mut cycle_counter = 0;
        let mut state = mdp.get_initial_state(rng);
        let mut episode_reward = 0.0;

        while steps < max_steps {
            let action = match state.light_state {
                envs::my_intersection::LightState::NorthSouthOpen => {
                    if cycle_counter < ns_time {
                        cycle_counter += 1;
                        Action::Stay
                    } else {
                        cycle_counter = 0;
                        Action::Change
                    }
                }
                envs::my_intersection::LightState::EastWestOpen => {
                    if cycle_counter < ew_time {
                        cycle_counter += 1;
                        Action::Stay
                    } else {
                        cycle_counter = 0;
                        Action::Change
                    }
                }
                envs::my_intersection::LightState::ChangingToNS => Action::WaitForChange,
                envs::my_intersection::LightState::ChangingToEW => Action::WaitForChange,
            };

            // println!("step: {:?}, action: {:?}", steps, action);

            let (next_state, reward) = mdp.perform_action((state, action), rng);
            state = next_state;
            episode_reward += reward;

            steps += 1;
        }
        total_reward += episode_reward;
    }
    total_reward / episodes as f64
}
