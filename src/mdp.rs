use rand::distributions::{Distribution, WeightedIndex};
use rand_chacha::ChaCha20Rng;
use std::{
    collections::{BTreeMap, HashSet},
    hash::Hash,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct State(pub usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Action(pub usize);

// some type aliases for readability
pub type Probability = f64;
pub type Reward = f64;
pub type Transition = (Probability, State, Reward);

#[derive(Clone)]
pub struct Mdp {
    pub transitions: BTreeMap<(State, Action), Vec<Transition>>,
    pub terminal_states: Vec<State>,
    pub initial_state: State,
}

impl Mdp {
    pub fn perform_action(
        &self,
        state_action: (State, Action),
        rng: &mut ChaCha20Rng,
    ) -> (State, Reward) {
        if let Some(transitions) = self.transitions.get(&state_action) {
            // extract probabilities, create distribution and sample
            let probs: Vec<_> = transitions
                .iter()
                .map(|(prob, _, _)| (prob * 100.0) as u32)
                .collect();
            let dist = WeightedIndex::new(probs).unwrap();
            let state_index = dist.sample(rng);

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
        let transition_probabilities: BTreeMap<(State, Action), Vec<Transition>> =
            BTreeMap::from([
                (
                    (State(0), Action(0)),
                    vec![(0.2, State(1), 1.0), (0.8, State(1), 10.0)],
                ),
                ((State(0), Action(1)), vec![(1.0, State(0), -1.0)]),
                ((State(1), Action(0)), vec![(1.0, State(1), -1.0)]),
                (
                    (State(1), Action(1)),
                    vec![(0.99, State(0), -2.0), (0.01, State(2), 1000.0)],
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

#[derive(Debug, Clone)]
pub struct MapMdp<S: GenericState, A: GenericAction> {
    pub transitions: BTreeMap<(S, A), Vec<(Probability, S, Reward)>>,
    pub terminal_states: std::collections::HashSet<S>,
    pub initial_state: S,
}

impl<S: GenericState, A: GenericAction> MapMdp<S, A> {
    pub fn new(initial_state: S) -> MapMdp<S, A> {
        let transitions: BTreeMap<(S, A), Vec<(Probability, S, Reward)>> = BTreeMap::new();
        let terminal_states: HashSet<S> = HashSet::new();

        MapMdp {
            transitions,
            terminal_states,
            initial_state,
        }
    }
}

pub trait GenericState: Ord + Clone + Hash + Copy {}
impl<T: Ord + Clone + Hash + Copy> GenericState for T {}
pub trait GenericAction: Ord + Copy + Clone {}

impl<T: Ord + Copy + Clone> GenericAction for T {}

pub trait GenericMdp<S: GenericState, A: GenericAction> {
    fn add_transition_vector(
        &mut self,
        sa: (S, A),
        transition: Vec<(Probability, S, Reward)>,
    ) -> anyhow::Result<()>;

    fn add_terminal_state(&mut self, state: S);

    fn perform_action(&self, state_action: (S, A), rng: &mut ChaCha20Rng) -> (S, Reward);

    fn get_possible_actions(&self, current_state: S) -> Vec<A>;

    fn get_all_state_actions(&self) -> Vec<(S, A)>;

    fn is_terminal(&self, state: S) -> bool;

    fn get_initial_sate(&self) -> S;
}

impl<S: GenericState, A: GenericAction> GenericMdp<S, A> for MapMdp<S, A> {
    fn add_transition_vector(
        &mut self,
        sa: (S, A),
        transition: Vec<(Probability, S, Reward)>,
    ) -> anyhow::Result<()> {
        if self.transitions.contains_key(&sa) {
            return Err(anyhow::anyhow!("Duplicate state insert"));
        }
        self.transitions.insert(sa, transition);
        Ok(())
    }

    fn add_terminal_state(&mut self, state: S) {
        self.terminal_states.insert(state);
    }

    fn perform_action(&self, state_action: (S, A), rng: &mut ChaCha20Rng) -> (S, Reward) {
        if let Some(transitions) = self.transitions.get(&state_action) {
            // extract probabilities, create distribution and sample
            let probs: Vec<_> = transitions
                .iter()
                .map(|(prob, _, _)| (prob * 100.0) as u32)
                .collect();
            let dist = WeightedIndex::new(probs).unwrap();
            let state_index = dist.sample(rng);

            //return resulting state and reward
            (transitions[state_index].1, transitions[state_index].2)
        } else {
            // you don't want to be here
            panic!("something went very wrong");
        }
    }

    fn get_possible_actions(&self, current_state: S) -> Vec<A> {
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

    fn get_all_state_actions(&self) -> Vec<(S, A)> {
        self.transitions.keys().cloned().collect()
    }

    fn is_terminal(&self, state: S) -> bool {
        self.terminal_states.contains(&state)
    }

    fn get_initial_sate(&self) -> S {
        self.initial_state
    }
}
