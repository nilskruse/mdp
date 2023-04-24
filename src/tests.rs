use std::collections::HashMap;

use assert_float_eq::assert_f64_near;
use rand::SeedableRng;

use crate::{mdp::{Mdp, State, Action, Transition}, algorithms::q_learning};

#[test]
fn test_q_learning() {
    // this will select action 1 on first step and go to state 0
    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(4);
    let mdp = create_test_mdp();
    let q_map = q_learning(&mdp, 0.1, 0.9, (mdp.initial_state, Action(0)), 1, 1, &mut rng);

    println!("{:?}", q_map);
    assert_f64_near!(*q_map.get(&(State(0), Action(1))).unwrap(), -0.1);
}

fn create_test_mdp() -> Mdp {
    let transition_probabilities: HashMap<(State, Action), Vec<Transition>> = HashMap::from([
        (
            (State(0), Action(0)),
            vec![(0.2, State(1), 1.0), (0.8, State(2), 10.0)],
        ),
        ((State(0), Action(1)), vec![(1.0, State(0), -1.0)]),
        ((State(1), Action(0)), vec![(1.0, State(1), -1.0)]),
        (
            (State(1), Action(1)),
            vec![(0.99, State(0), -2.0), (0.01, State(2), 1000.0)],
        ),
    ]);

    let terminal_states = vec![State(2)];

    Mdp {
        transitions: transition_probabilities,
        terminal_states,
        initial_state: State(0),
    }
}
