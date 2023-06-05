#![allow(unused_variables)]
use rand::Rng;

use crate::mdp::GenericMdp;

#[derive(Debug, Ord, PartialOrd, Eq, PartialEq, Clone, Hash, Copy)]
struct State {
    light_state: LightState,
    ns_cars: usize,
    ew_cars: usize,
}

#[derive(Debug, Ord, PartialOrd, Eq, PartialEq, Clone, Hash, Copy)]
enum LightState {
    NorthSouth = 0,
    EastWest = 1,
}

#[derive(Debug, Ord, PartialOrd, Eq, PartialEq, Clone, Hash, Copy)]
enum Action {
    Change = 0,
    Stay = 1,
}

pub struct MyIntersectionMdp {}

impl GenericMdp<State, Action> for MyIntersectionMdp {
    fn perform_action<R: rand::SeedableRng + Rng>(
        &self,
        state_action: (State, Action),
        rng: &mut R,
    ) -> (State, crate::mdp::Reward) {
        todo!()
    }

    fn get_possible_actions(&self, current_state: State) -> Vec<Action> {
        todo!()
    }

    fn get_all_state_actions_iter(
        &self,
    ) -> std::collections::btree_map::Keys<
        '_,
        (State, Action),
        Vec<(crate::mdp::Probability, State, crate::mdp::Reward)>,
    > {
        todo!()
    }

    fn is_terminal(&self, state: State) -> bool {
        todo!()
    }

    fn get_initial_state<R: Rng + rand::SeedableRng>(&self, rng: &mut R) -> State {
        todo!()
    }

    fn get_discount_factor(&self) -> f64 {
        todo!()
    }
}
