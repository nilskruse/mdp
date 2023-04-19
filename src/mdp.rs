use rand::distributions::{Distribution, WeightedIndex};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct State(pub u8);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Action {
    A,
    B,
}

// some type aliases for readability
pub type Probability = f64;
pub type Reward = f64;

pub struct Mdp {
    pub transition_probabilities: HashMap<(State, Action), Vec<(Probability, State, Reward)>>,
}

impl Mdp {
    pub fn perform_action(&self, state_action: (State, Action)) -> (State, Reward) {
        if let Some(transitions) = self.transition_probabilities.get(&state_action) {
            let mut rng = rand::thread_rng();

            // extract probabilities, create distribution and sample
            let probs: Vec<_> = transitions
                .iter()
                .map(|(prob, _, _)| (prob * 100.0) as u32)
                .collect();
            let dist = WeightedIndex::new(&probs).unwrap();
            let state_index = dist.sample(&mut rng);

            //return resulting state and reward
            (State(state_index as u8), transitions[state_index].2)
        } else {
            // you don't want to be here
            panic!("something went very wrong");
        }
    }

    pub fn get_possible_actions(&self, current_state: State) -> Vec<Action> {
        self.transition_probabilities
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
        let transition_probabilities: HashMap<(State, Action), Vec<(Probability, State, Reward)>> =
            HashMap::from([
                (
                    (State(0), Action::A),
                    vec![(0.8, State(1), 1.0), (0.2, State(1), 10.0)],
                ),
                ((State(0), Action::B), vec![(1.0, State(0), -1.0)]),
                ((State(1), Action::A), vec![(1.0, State(1), -1.0)]),
                ((State(1), Action::B), vec![(1.0, State(0), -1.0)]),
                ((State(2), Action::A), vec![(1.0, State(0), -1.0)]),
            ]);

        Mdp {
            transition_probabilities,
        }
    }
}
