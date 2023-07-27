use rand::{
    distributions::{Distribution, WeightedIndex},
    Rng,
};

use std::{
    collections::{BTreeMap, HashSet},
    hash::Hash,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct IndexState(pub usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct IndexAction(pub usize);

pub type IndexMdp = MapMdp<IndexState, IndexAction>;

// some type aliases for readability
pub type Probability = f64;
pub type Reward = f64;
pub type Transition = (Probability, IndexState, Reward);

#[derive(Debug, Clone)]
pub struct MapMdp<S: GenericState, A: GenericAction> {
    pub transitions: Transitions<S, A>,
    pub terminal_states: std::collections::HashSet<S>,
    pub initial_state: S,
    pub discount_factor: f64,
    pub states_actions: Vec<(S, A)>,
}

type Transitions<S, A> = BTreeMap<(S, A), Vec<(Probability, S, Reward)>>;

impl<S: GenericState, A: GenericAction> MapMdp<S, A> {
    pub fn new(discount_factor: f64, initial_state: S) -> MapMdp<S, A> {
        let transitions: Transitions<S, A> = BTreeMap::new();
        let terminal_states: HashSet<S> = HashSet::new();
        let states_actions = vec![];

        MapMdp {
            transitions,
            terminal_states,
            initial_state,
            discount_factor,
            states_actions,
        }
    }
    pub fn add_transition_vector(
        &mut self,
        sa: (S, A),
        transition: Vec<(Probability, S, Reward)>,
    ) -> anyhow::Result<()> {
        if self.transitions.contains_key(&sa) {
            return Err(anyhow::anyhow!("Duplicate state insert"));
        }
        self.transitions.insert(sa, transition);
        self.states_actions.push(sa);
        Ok(())
    }

    pub fn add_terminal_state(&mut self, state: S) {
        self.terminal_states.insert(state);
    }
}

pub trait GenericState: Ord + Clone + Hash + Copy + std::fmt::Debug {}
impl<T: Ord + Clone + Hash + Copy + std::fmt::Debug> GenericState for T {}

pub trait GenericAction: Ord + Copy + Clone + Hash + std::fmt::Debug {}
impl<T: Ord + Copy + Clone + Hash + std::fmt::Debug> GenericAction for T {}

pub trait GenericMdp<S: GenericState, A: GenericAction> {
    fn perform_action<R: Rng>(&self, state_action: (S, A), rng: &mut R) -> (S, Reward);

    fn get_possible_actions(&self, current_state: S) -> Vec<A>;

    fn get_all_state_actions(&self) -> &[(S, A)];

    fn is_terminal(&self, state: S) -> bool;

    fn get_initial_state<R: Rng>(&self, rng: &mut R) -> S;

    fn get_discount_factor(&self) -> f64;
}

impl<S: GenericState, A: GenericAction> GenericMdp<S, A> for MapMdp<S, A> {
    fn perform_action<R: Rng>(&self, state_action: (S, A), rng: &mut R) -> (S, Reward) {
        if let Some(transitions) = self.transitions.get(&state_action) {
            // extract probabilities, create distribution and sample
            let probs: Vec<_> = transitions.iter().map(|(prob, _, _)| prob).collect();
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

    fn get_all_state_actions(&self) -> &[(S, A)] {
        &self.states_actions
    }

    fn is_terminal(&self, state: S) -> bool {
        self.terminal_states.contains(&state)
    }

    fn get_initial_state<R: Rng>(&self, _: &mut R) -> S {
        self.initial_state
    }

    fn get_discount_factor(&self) -> f64 {
        self.discount_factor
    }
}
