#![allow(unused_variables)]
use std::collections::{btree_map, BTreeMap};

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
    NorthSouthOpen = 0,
    EastWestOpen = 1,
}

#[derive(Debug, Ord, PartialOrd, Eq, PartialEq, Clone, Hash, Copy)]
enum Action {
    Change = 0,
    Stay = 1,
}

pub struct MyIntersectionMdp {
    new_car_prob_ns: f64,
    new_car_prob_ew: f64,
    max_cars: usize,
}

impl MyIntersectionMdp {
    fn open_road_transition<R: Rng>(&self, old_cars: usize, new_prob: f64, rng: &mut R) -> usize {
        if old_cars == 0 {
            0
        } else if rng.gen_range(0.0..1.0) < (1.0 - new_prob) {
            old_cars - 1
        } else {
            old_cars
        }
    }

    fn closed_road_transition<R: Rng>(&self, old_cars: usize, new_prob: f64, rng: &mut R) -> usize {
        if old_cars == self.max_cars {
            self.max_cars
        } else if rng.gen_range(0.0..1.0) < (1.0 - new_prob) {
            old_cars
        } else {
            old_cars + 1
        }
    }
}

impl GenericMdp<State, Action> for MyIntersectionMdp {
    fn perform_action<R: rand::SeedableRng + Rng>(
        &self,
        state_action: (State, Action),
        rng: &mut R,
    ) -> (State, crate::mdp::Reward) {
        let (state, action) = state_action;

        let new_light_state = match action {
            Action::Change => match state.light_state {
                LightState::NorthSouthOpen => LightState::EastWestOpen,
                LightState::EastWestOpen => LightState::NorthSouthOpen,
            },
            Action::Stay => state.light_state,
        };

        let (new_ns_cars, new_ew_cars) = match new_light_state {
            LightState::NorthSouthOpen => (
                self.open_road_transition(state.ns_cars, self.new_car_prob_ns, rng),
                self.closed_road_transition(state.ew_cars, self.new_car_prob_ew, rng),
            ),
            LightState::EastWestOpen => (
                self.closed_road_transition(state.ns_cars, self.new_car_prob_ns, rng),
                self.open_road_transition(state.ew_cars, self.new_car_prob_ew, rng),
            ),
        };

        let new_state = State {
            light_state: new_light_state,
            ns_cars: new_ns_cars,
            ew_cars: new_ew_cars,
        };

        let reward: f64 = -((new_ns_cars + new_ew_cars) as f64);

        (new_state, reward)
    }

    fn get_possible_actions(&self, current_state: State) -> Vec<Action> {
        vec![Action::Change, Action::Stay]
    }

    fn get_all_state_actions_iter(
        &self,
    ) -> std::collections::btree_map::Keys<
        '_,
        (State, Action),
        Vec<(crate::mdp::Probability, State, crate::mdp::Reward)>,
    > {
        let mut states = vec![];
        for ns_cars in 0..=self.max_cars {
            for ew_cars in 0..=self.max_cars {
                states.push(State {
                    light_state: LightState::NorthSouthOpen,
                    ns_cars,
                    ew_cars,
                });
                states.push(State {
                    light_state: LightState::EastWestOpen,
                    ns_cars,
                    ew_cars,
                });
            }
        }

        let mut states_actions = BTreeMap::new();

        states.iter().for_each(|state| {
            states_actions.insert(
                (*state, Action::Stay),
                vec![(
                    0.0_f64,
                    State {
                        light_state: LightState::NorthSouthOpen,
                        ns_cars: 0,
                        ew_cars: 0,
                    },
                    0.0_f64,
                )],
            );
            states_actions.insert(
                (*state, Action::Change),
                vec![(
                    0.0_f64,
                    State {
                        light_state: LightState::NorthSouthOpen,
                        ns_cars: 0,
                        ew_cars: 0,
                    },
                    0.0_f64,
                )],
            );
        });

        states_actions.clone().keys()
    }

    fn is_terminal(&self, state: State) -> bool {
        false
    }

    fn get_initial_state<R: Rng + rand::SeedableRng>(&self, rng: &mut R) -> State {
        State {
            light_state: LightState::NorthSouthOpen,
            ns_cars: 0,
            ew_cars: 0,
        }
    }

    fn get_discount_factor(&self) -> f64 {
        1.0
    }
}
