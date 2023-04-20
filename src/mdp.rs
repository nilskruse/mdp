use rand::distributions::{Distribution, WeightedIndex};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct State(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Action(pub u32);

// some type aliases for readability
pub type Probability = f64;
pub type Reward = f64;
pub type Transition = (Probability, State, Reward);

pub struct Mdp {
    pub transitions: HashMap<(State, Action), Vec<Transition>>,
    pub terminal_states: Vec<State>,
    pub initial_state: State,
}

impl Mdp {
    pub fn perform_action(&self, state_action: (State, Action)) -> (State, Reward) {
        if let Some(transitions) = self.transitions.get(&state_action) {
            let mut rng = rand::thread_rng();

            // extract probabilities, create distribution and sample
            let probs: Vec<_> = transitions
                .iter()
                .map(|(prob, _, _)| (prob * 100.0) as u32)
                .collect();
            let dist = WeightedIndex::new(&probs).unwrap();
            let state_index = dist.sample(&mut rng);

            //return resulting state and reward
            (transitions[state_index].1, transitions[state_index].2)
        } else {
            // you don't want to be here
            panic!("something went very wrong");
        }
    }

    pub fn get_possible_actions(&self, current_state: State) -> Vec<Action> {
        self.transitions
            .iter()
            .filter_map(|((state, action), _)| {
                if state.eq(&current_state) {
                    Some(*action)
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn new_test_mdp() -> Mdp {
        let transition_probabilities: HashMap<(State, Action), Vec<Transition>> = HashMap::from([
            (
                (State(0), Action(0)),
                vec![(0.2, State(1), 1.0), (0.8, State(1), 10.0)],
            ),
            ((State(0), Action(1)), vec![(1.0, State(0), -1.0)]),
            ((State(1), Action(0)), vec![(1.0, State(1), -1.0)]),
            (
                (State(1), Action(1)),
                vec![(0.99, State(0), -2.0), (0.01, State(2), -1.0)],
            ),
            ((State(2), Action(0)), vec![(1.0, State(0), 1.0)]),
        ]);

        let terminal_states = vec![State(2)];

        Mdp {
            transitions: transition_probabilities,
            terminal_states,
            initial_state: State(0),
        }
    }
}
