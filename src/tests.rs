use std::collections::BTreeMap;

use assert_float_eq::assert_f64_near;
use rand::SeedableRng;

use crate::{
    algorithms::{q_learning::QLearning, sarsa::Sarsa, StateActionAlgorithm},
    mdp::{IndexAction, IndexMdp, IndexState, Transition},
    utils::print_q_map,
};

#[test]
fn test_q_learning() {
    // this will select action 1 on first step and go to state 0
    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(212);
    let mdp = create_test_mdp();
    let algo = QLearning::new(0.1, 0.9, 0.1, 5);
    let q_map = algo.run(&mdp, 1, &mut rng);

    print_q_map(&q_map);

    assert_f64_near!(*q_map.get(&(IndexState(0), IndexAction(0))).unwrap(), 0.181);
    assert_f64_near!(*q_map.get(&(IndexState(0), IndexAction(1))).unwrap(), -0.1);
    assert_f64_near!(*q_map.get(&(IndexState(1), IndexAction(0))).unwrap(), -0.1);
    assert_f64_near!(
        *q_map.get(&(IndexState(1), IndexAction(1))).unwrap(),
        -0.191
    );
}

// TODO: redo the manual calculations because the initial state handling has changed
#[test]
fn test_sarsa() {
    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(8);
    let mdp = create_test_mdp();

    let algo = Sarsa::new(0.1, 0.9, 0.1, 5);
    let q_map = algo.run(&mdp, 1, &mut rng);

    assert_f64_near!(*q_map.get(&(IndexState(0), IndexAction(0))).unwrap(), 0.19);
    assert_f64_near!(*q_map.get(&(IndexState(0), IndexAction(1))).unwrap(), 0.0);
    assert_f64_near!(
        *q_map.get(&(IndexState(1), IndexAction(0))).unwrap(),
        -0.199
    );
    assert_f64_near!(
        *q_map.get(&(IndexState(1), IndexAction(1))).unwrap(),
        -0.191
    );
}

const EPISODES: usize = 1000;

#[test]
fn test_sarsa_equivalence() {
    let mdp = create_test_mdp();
    let algo = Sarsa::new(0.1, 0.9, 0.1, 1000);

    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0);
    let q_map_1 = algo.run(&mdp, EPISODES, &mut rng);

    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0);
    let mut q_map_2 = algo.run(&mdp, EPISODES / 2, &mut rng);
    algo.run_with_q_map(&mdp, EPISODES / 2, &mut rng, &mut q_map_2);

    assert_eq!(q_map_1, q_map_2);
}

#[test]
fn test_q_learning_equivalence() {
    let mdp = create_test_mdp();
    let algo = QLearning::new(0.1, 0.9, 0.1, 1000);

    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0);
    let q_map_1 = algo.run(&mdp, EPISODES, &mut rng);

    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0);
    let mut q_map_2 = algo.run(&mdp, EPISODES / 2, &mut rng);
    algo.run_with_q_map(&mdp, EPISODES / 2, &mut rng, &mut q_map_2);

    assert_eq!(q_map_1, q_map_2);
}

fn create_test_mdp() -> IndexMdp {
    let transition_probabilities: BTreeMap<(IndexState, IndexAction), Vec<Transition>> =
        BTreeMap::from([
            (
                (IndexState(0), IndexAction(0)),
                vec![(0.2, IndexState(1), 1.0), (0.8, IndexState(2), 10.0)],
            ),
            (
                (IndexState(0), IndexAction(1)),
                vec![(1.0, IndexState(0), -1.0)],
            ),
            (
                (IndexState(1), IndexAction(0)),
                vec![(1.0, IndexState(1), -1.0)],
            ),
            (
                (IndexState(1), IndexAction(1)),
                vec![(0.99, IndexState(0), -2.0), (0.01, IndexState(2), 1000.0)],
            ),
        ]);

    let terminal_states = vec![IndexState(2)];

    IndexMdp {
        transitions: transition_probabilities,
        terminal_states,
        initial_state: IndexState(0),
    }
}
