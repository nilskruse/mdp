use std::collections::HashMap;
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct State(u8);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Action {
    A,
    B,
}

// some type aliases for readability
pub type Probability = f64;
pub type Reward = f64;

pub struct Mdp {
    pub transition_probabilities: HashMap<(State, Action), Vec<(f64, State, f64)>>,
    pub discount_factor: f64,
}

impl Mdp {
    pub fn value_iteration(&self, tolerance: f64) -> HashMap<State, f64> {
        let mut value_map: HashMap<State, f64> = HashMap::new();
        let mut delta = f64::MAX;

        while delta > tolerance {
            delta = 0.0;

            for (state, _) in self.transition_probabilities.keys() {
                let old_value = *value_map.get(state).unwrap_or(&0.0);
                let new_value = self.best_action_value(*state, &value_map);

                value_map.insert(*state, new_value);
                delta = delta.max((old_value - new_value).abs());
            }
        }

        value_map
    }

    pub fn best_action_value(&self, state: State, value_map: &HashMap<State, f64>) -> f64 {
        self.transition_probabilities
            .iter()
            .filter_map(|((s, _), transitions)| {
                if *s == state {
                    let expected_value: f64 = transitions
                        .iter()
                        .map(|(prob, next_state, reward)| {
                            prob * (reward
                                + self.discount_factor * value_map.get(next_state).unwrap_or(&0.0))
                        })
                        .sum();
                    Some(expected_value)
                } else {
                    None
                }
            })
            .fold(f64::MIN, f64::max)
    }

    pub fn new_test_mdp(discount_factor: f64) -> Mdp {
        let transition_probabilities: HashMap<(State, Action), Vec<(Probability, State, Reward)>> =
            HashMap::from([
                (
                    (State(0), Action::A),
                    vec![(0.8, State(1), 1.0), (0.2, State(1), 10.0)],
                ),
                ((State(0), Action::B), vec![(1.0, State(0), 1.0)]),
                ((State(1), Action::A), vec![(1.0, State(1), -1.0)]),
                ((State(1), Action::B), vec![(1.0, State(0), 1.0)]),
            ]);

        Mdp {
            transition_probabilities,
            discount_factor,
        }
    }
}


