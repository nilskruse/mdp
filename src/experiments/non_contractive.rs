use crate::{
    algorithms::GenericStateActionAlgorithm,
    mdp::{GenericAction, GenericMdp, GenericState},
};
use std::collections::{BTreeMap, HashSet};

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

use crate::{
    algorithms::{
        q_learning::QLearning, q_learning_beta::QLearningBeta, value_iteration::value_iteration,
    },
    mdp::{IndexAction, IndexMdp, IndexState, Transition},
    policies::{epsilon_greedy_policy, greedy_policy},
    utils::{print_q_map, print_transition_map},
};

fn build_mdp(p: f64) -> IndexMdp {
    let transition_probabilities: BTreeMap<(IndexState, IndexAction), Vec<Transition>> =
        BTreeMap::from([
            (
                (IndexState(0), IndexAction(0)),
                vec![(p, IndexState(2), 1000.0), (1.0 - p, IndexState(3), 1.0)],
            ),
            (
                (IndexState(0), IndexAction(1)),
                vec![(1.0, IndexState(1), 0.0)],
            ),
            (
                (IndexState(1), IndexAction(1)),
                vec![(1.0, IndexState(0), 0.0)],
            ),
            (
                (IndexState(2), IndexAction(0)),
                vec![(1.0, IndexState(2), 0.0)],
            ),
            (
                (IndexState(3), IndexAction(0)),
                vec![(1.0, IndexState(3), 0.0)],
            ),
        ]);

    let terminal_states_vec = vec![IndexState(2), IndexState(3)];

    let terminal_states: HashSet<IndexState> =
        HashSet::from_iter(terminal_states_vec.iter().copied());
    let discount_factor = 1.0;

    IndexMdp {
        transitions: transition_probabilities,
        terminal_states,
        initial_state: IndexState(0),
        discount_factor,
    }
}

pub fn run_experiment() {
    let mdp = build_mdp(0.001);
    let rig = (IndexState(2), IndexAction(0));

    println!("Q-Learning");
    let q_algo = RiggedQLearning::new(0.1, 0.2, usize::MAX);
    let mut rng = ChaCha20Rng::seed_from_u64(0);
    let q_map = q_algo.run(&mdp, 1000000, &mut rng, rig);
    println!("Q-Table:");
    print_q_map(&q_map);
    println!();

    println!("Q-Learning Beta");
    let q_beta_algo = QLearningBeta::new(0.1, 0.2, usize::MAX);
    let mut rng = ChaCha20Rng::seed_from_u64(0);
    let q_map = q_beta_algo.run(&mdp, 2000000, &mut rng, rig);
    println!("Q-Table:");
    print_q_map(&q_map);
    println!();

    println!("\nTransitions");
    print_transition_map(&mdp);
    println!();

    println!("Value iteration:");
    let values = value_iteration(&mdp, 0.001);
    println!("{:?}", values);
    println!();
}

pub struct RiggedQLearning {
    alpha: f64,
    epsilon: f64,
    max_steps: usize,
}

impl RiggedQLearning {
    pub fn new(alpha: f64, epsilon: f64, max_steps: usize) -> Self {
        RiggedQLearning {
            alpha,
            epsilon,
            max_steps,
        }
    }
}

impl RiggedStateActionAlgorithm for RiggedQLearning {
    fn run_with_q_map<
        M: GenericMdp<S, A>,
        S: GenericState,
        A: GenericAction,
        R: Rng + SeedableRng,
    >(
        &self,
        mdp: &M,
        episodes: usize,
        rng: &mut R,
        q_map: &mut BTreeMap<(S, A), f64>,
        rig: (S, A),
    ) {
        for episode in 1..=episodes {
            let mut current_state = mdp.get_initial_state();
            let mut steps = 0;

            while !mdp.is_terminal(current_state) && steps < self.max_steps {
                let Some(mut selected_action) = epsilon_greedy_policy(mdp, q_map, current_state, self.epsilon, rng) else {break};
                let (mut next_state, mut reward) =
                    mdp.perform_action((current_state, selected_action), rng);

                // // get high, improbable reward on first episode and first step
                if episode == 2 && steps == 0 {
                    println!("Rigging first action selection!!!");
                    next_state = rig.0;
                    selected_action = rig.1;
                    reward = 1000.0;
                }

                // update q_map
                let Some(best_action) = greedy_policy(mdp, q_map, next_state, rng) else {break};
                let best_q = *q_map
                    .get(&(next_state, best_action))
                    .expect("No qmap entry found");

                let current_q = q_map.entry((current_state, selected_action)).or_insert(0.0);
                *current_q = *current_q
                    + self.alpha * (reward + mdp.get_discount_factor() * best_q - *current_q);

                current_state = next_state;

                steps += 1;
            }
        }
    }
}

pub trait RiggedStateActionAlgorithm {
    // default implementation
    fn run<M: GenericMdp<S, A>, S: GenericState, A: GenericAction, R: Rng + SeedableRng>(
        &self,
        mdp: &M,
        episodes: usize,
        rng: &mut R,
        rig: (S, A),
    ) -> BTreeMap<(S, A), f64> {
        let mut q_map: BTreeMap<(S, A), f64> = BTreeMap::new();

        mdp.get_all_state_actions_iter().for_each(|state_action| {
            q_map.insert(*state_action, 0.0);
        });

        self.run_with_q_map(mdp, episodes, rng, &mut q_map, rig);

        q_map
    }
    fn run_with_q_map<M, S, A, R>(
        &self,
        mdp: &M,
        episodes: usize,
        rng: &mut R,
        q_map: &mut BTreeMap<(S, A), f64>,
        rig: (S, A),
    ) where
        M: GenericMdp<S, A>,
        S: GenericState,
        A: GenericAction,
        R: Rng + SeedableRng;
}
