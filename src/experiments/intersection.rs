use rand::{Rng, SeedableRng};

use crate::{
    algorithms::{q_learning::QLearning, GenericStateActionAlgorithm},
    envs::{
        self,
        my_intersection::{LightAction, MyIntersectionMdp},
    },
    eval::{evaluate_epsilon_greedy_policy, evaluate_greedy_policy, evaluate_random_policy},
    mdp::GenericMdp,
};

pub fn run_experiment() {
    let eval_episodes = 100;
    let train_episodes = 1000;
    let episode_length = 2000;
    let generic_mdp = MyIntersectionMdp::new(0.5, 0.3, 50);
    let generic_q_learning = QLearning::new(0.1, 0.1, episode_length);
    // let generic_q_learning = SarsaLambda::new(0.1, 0.1, 0.1, episode_length, Trace::Accumulating);
    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(1);
    let q_map = generic_q_learning.run(&generic_mdp, train_episodes, &mut rng);

    println!(
        "number of (state, action): {:?}",
        generic_mdp.get_all_state_actions().len()
    );
    // println!("{:#?}", generic_mdp);
    println!("{train_episodes} episodes");
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

    let avg_reward_random =
        evaluate_random_policy(&generic_mdp, eval_episodes, episode_length, &mut rng);

    // q_map.iter().for_each(|e| println!("{:?}", e));
    //
    let avg_reward_fixed_cycle = fixed_cycle(
        &generic_mdp,
        train_episodes,
        episode_length,
        50,
        30,
        &mut rng,
    );

    println!("epsilon: {:?}", avg_reward_epsilon);
    println!("greedy: {:?}", avg_reward_greedy);
    println!("random: {:?}", avg_reward_random);
    println!("fixed cycle reward: {:?}", avg_reward_fixed_cycle);
}

pub fn fixed_cycle(
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
                        LightAction::Stay
                    } else {
                        cycle_counter = 0;
                        LightAction::Change
                    }
                }
                envs::my_intersection::LightState::EastWestOpen => {
                    if cycle_counter < ew_time {
                        cycle_counter += 1;
                        LightAction::Stay
                    } else {
                        cycle_counter = 0;
                        LightAction::Change
                    }
                }
                envs::my_intersection::LightState::ChangingToNS => LightAction::WaitForChange,
                envs::my_intersection::LightState::ChangingToEW => LightAction::WaitForChange,
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
