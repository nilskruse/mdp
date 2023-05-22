use std::collections::BTreeMap;

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use crate::{
    algorithms::{
        q_learning::QLearning, q_learning_beta::QLearningBeta, value_iteration::value_iteration,
        StateActionAlgorithm,
    },
    mdp::{Action, Mdp, State, Transition},
    policies::{epsilon_greedy_policy, greedy_policy},
    utils::{print_q_map, print_transition_map},
};

fn build_mdp(p: f64) -> Mdp {
    let transition_probabilities: BTreeMap<(State, Action), Vec<Transition>> = BTreeMap::from([
        (
            (State(0), Action(0)),
            vec![(p, State(2), 1000.0), (1.0 - p, State(3), 1.0)],
        ),
        ((State(0), Action(1)), vec![(1.0, State(1), 0.0)]),
        ((State(1), Action(1)), vec![(1.0, State(0), 0.0)]),
        ((State(2), Action(0)), vec![(1.0, State(2), 0.0)]),
        ((State(3), Action(0)), vec![(1.0, State(3), 0.0)]),
    ]);

    let terminal_states = vec![State(2), State(3)];

    Mdp {
        transitions: transition_probabilities,
        terminal_states,
        initial_state: State(0),
    }
}

pub fn run_experiment() {
    let mdp = build_mdp(0.001);

    println!("Q-Learning");
    let q_algo = RiggedQLearning::new(0.1, 1.0, 0.2, usize::MAX);
    let mut rng = ChaCha20Rng::seed_from_u64(0);
    let q_map = q_algo.run(&mdp, 1000000, &mut rng);
    println!("Q-Table:");
    print_q_map(&q_map);
    println!();

    println!("Q-Learning Beta");
    let q_beta_algo = QLearningBeta::new(0.1, 1.0, 0.2, usize::MAX);
    let mut rng = ChaCha20Rng::seed_from_u64(0);
    let q_map = q_beta_algo.run(&mdp, 1000000, &mut rng);
    println!("Q-Table:");
    print_q_map(&q_map);
    println!();

    println!("\nTransitions");
    print_transition_map(&mdp);
    let values = value_iteration(&mdp, 0.001, 1.0);
    println!("{:?}", values);
    println!();
}

pub struct RiggedQLearning {
    alpha: f64,
    gamma: f64,
    epsilon: f64,
    max_steps: usize,
}

impl RiggedQLearning {
    pub fn new(alpha: f64, gamma: f64, epsilon: f64, max_steps: usize) -> Self {
        RiggedQLearning {
            alpha,
            gamma,
            epsilon,
            max_steps,
        }
    }
}

impl StateActionAlgorithm for RiggedQLearning {
    fn run_with_q_map(
        &self,
        mdp: &crate::mdp::Mdp,
        episodes: usize,
        rng: &mut rand_chacha::ChaCha20Rng,
        q_map: &mut std::collections::BTreeMap<(crate::mdp::State, crate::mdp::Action), f64>,
    ) {
        for episode in 1..=episodes {
            let mut current_state = mdp.initial_state;
            let mut steps = 0;

            while !mdp.terminal_states.contains(&current_state) && steps < self.max_steps {
                let Some(mut selected_action) = epsilon_greedy_policy(mdp, q_map, current_state, self.epsilon, rng) else {break};
                let (mut next_state, mut reward) =
                    mdp.perform_action((current_state, selected_action), rng);
                // println!("{:?}, {:?}, {:?}", current_state, selected_action, next_state);

                // get high, improbable reward on first episode and first step
                if episode == 2 && steps == 0 {
                    println!("Rigging first action selection!!!");
                    selected_action = Action(0);
                    next_state = State(2);
                    reward = 1000.0;
                }

                // update q_map
                let Some(best_action) = greedy_policy(mdp, q_map, next_state, rng) else {break};
                let best_q = *q_map
                    .get(&(next_state, best_action))
                    .expect("No qmap entry found");

                let current_q = q_map.entry((current_state, selected_action)).or_insert(0.0);
                *current_q = *current_q + self.alpha * (reward + self.gamma * best_q - *current_q);

                current_state = next_state;

                steps += 1;
            }
        }
    }
}
