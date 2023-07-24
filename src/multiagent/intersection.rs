#![allow(unused_variables)]
use std::{
    collections::{btree_map, BTreeMap},
    slice::Iter,
};

use itertools::{iproduct, Itertools};

use rand::Rng;

use crate::mdp::GenericMdp;

#[derive(Debug, Ord, PartialOrd, Eq, PartialEq, Clone, Hash, Copy)]
pub struct State {
    pub light_state_1: LightState,
    pub light_state_2: LightState,
    ns_cars_1: u8,
    ew_cars_1: u8,
    ns_cars_2: u8,
    ew_cars_2: u8,
}

#[derive(Debug, Ord, PartialOrd, Eq, PartialEq, Clone, Hash, Copy)]
pub enum LightState {
    NorthSouthOpen = 0,
    EastWestOpen = 1,
    ChangingToNS = 2,
    ChangingToEW = 3,
}

impl From<(LightState, LightState, u8, u8, u8, u8)> for State {
    fn from(value: (LightState, LightState, u8, u8, u8, u8)) -> Self {
        Self {
            light_state_1: value.0,
            light_state_2: value.1,
            ns_cars_1: value.2,
            ew_cars_1: value.3,
            ns_cars_2: value.4,
            ew_cars_2: value.5,
        }
    }
}

#[derive(Debug, Ord, PartialOrd, Eq, PartialEq, Clone, Hash, Copy)]
pub enum LightAction {
    Change = 0,
    Stay = 1,
    WaitForChange = 2,
}

#[derive(Debug, Ord, PartialOrd, Eq, PartialEq, Clone, Hash, Copy)]
struct Action(LightAction, LightAction);

pub struct MAIntersectionMdp {
    new_car_prob_ns_1: f64,
    new_car_prob_ew_1: f64,
    new_car_prob_ns_2: f64,
    new_car_prob_ew_2: f64,
    max_cars: u8,
    states_actions: Vec<(State, Action)>,
}

impl MAIntersectionMdp {
    pub fn new(
        new_car_prob_ns_1: f64,
        new_car_prob_ew_1: f64,
        new_car_prob_ns_2: f64,
        new_car_prob_ew_2: f64,
        max_cars: u8,
    ) -> Self {
        // first get all possible states
        let lightstates = [
            LightState::NorthSouthOpen,
            LightState::EastWestOpen,
            LightState::ChangingToNS,
            LightState::ChangingToEW,
        ];
        let car_range = 0..=max_cars;
        let state_iter = iproduct!(
            lightstates.iter().cloned(),
            lightstates.iter().cloned(),
            car_range.clone().into_iter(),
            car_range.clone().into_iter(),
            car_range.clone().into_iter(),
            car_range.clone().into_iter()
        );

        // combine with all possible action pairs for each state
        let states_actions: Vec<(State, Action)> = state_iter
            .flat_map(|state| {
                let ls_1 = state.0;
                let ls_2 = state.1;

                let ls_1_actions = match ls_1 {
                    LightState::NorthSouthOpen | LightState::EastWestOpen => {
                        vec![LightAction::Stay, LightAction::Change]
                    }
                    LightState::ChangingToNS | LightState::ChangingToEW => {
                        vec![LightAction::WaitForChange]
                    }
                };

                let ls_2_actions = match ls_2 {
                    LightState::NorthSouthOpen | LightState::EastWestOpen => {
                        vec![LightAction::Stay, LightAction::Change]
                    }
                    LightState::ChangingToNS | LightState::ChangingToEW => {
                        vec![LightAction::WaitForChange]
                    }
                };
                let action_pairs = iproduct!(ls_1_actions.into_iter(), ls_2_actions.into_iter());
                action_pairs
                    .map(|(a1, a2)| (State::from(state), Action(a1, a2)))
                    .collect_vec()
            })
            .collect_vec();

        println!(
            "multi agent states_actions count: {:?}",
            states_actions.len()
        );

        Self {
            new_car_prob_ns_1,
            new_car_prob_ew_1,
            new_car_prob_ns_2,
            new_car_prob_ew_2,
            max_cars,
            states_actions,
        }
    }

    fn open_road_transition<R: Rng>(&self, old_cars: u8, new_prob: f64, rng: &mut R) -> u8 {
        if old_cars == 0 {
            0
        } else if rng.gen_range(0.0..1.0) < (1.0 - new_prob) {
            old_cars - 1
        } else {
            old_cars
        }
    }

    fn closed_road_transition<R: Rng>(&self, old_cars: u8, new_prob: f64, rng: &mut R) -> u8 {
        if old_cars == self.max_cars {
            self.max_cars
        } else if rng.gen_range(0.0..1.0) < (1.0 - new_prob) {
            old_cars
        } else {
            old_cars + 1
        }
    }

    fn state_transfer(light_state: LightState, action: LightAction) -> LightState {
        match action {
            LightAction::Change => match light_state {
                LightState::NorthSouthOpen => LightState::ChangingToEW,
                LightState::EastWestOpen => LightState::ChangingToNS,
                LightState::ChangingToNS | LightState::ChangingToEW => {
                    panic!("Unreachable state: can't change light mid-cycle")
                }
            },
            LightAction::Stay => light_state,
            LightAction::WaitForChange => match light_state {
                LightState::ChangingToNS => LightState::NorthSouthOpen,
                LightState::ChangingToEW => LightState::EastWestOpen,
                LightState::NorthSouthOpen | LightState::EastWestOpen => {
                    println!("State: {:?}", light_state);
                    panic!("Unreachable state: can't wait for change when lights are not changing")
                }
            },
        }
    }
}

impl GenericMdp<State, Action> for MAIntersectionMdp {
    fn perform_action<R: rand::SeedableRng + Rng>(
        &self,
        state_action: (State, Action),
        rng: &mut R,
    ) -> (State, crate::mdp::Reward) {
        let (state, action) = state_action;
        let (action_1, action_2) = (action.0, action.1);

        let new_light_state_1 = match action_1 {
            LightAction::Change => match state.light_state_1 {
                LightState::NorthSouthOpen => LightState::ChangingToEW,
                LightState::EastWestOpen => LightState::ChangingToNS,
                LightState::ChangingToNS | LightState::ChangingToEW => {
                    panic!("Unreachable state: can't change light mid-cycle")
                }
            },
            LightAction::Stay => state.light_state_1,
            LightAction::WaitForChange => match state.light_state_1 {
                LightState::ChangingToNS => LightState::NorthSouthOpen,
                LightState::ChangingToEW => LightState::EastWestOpen,
                LightState::NorthSouthOpen | LightState::EastWestOpen => {
                    println!("State: {:?}", state);
                    panic!("Unreachable state: can't wait for change when lights are not changing")
                }
            },
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
            LightState::ChangingToNS | LightState::ChangingToEW => (
                self.closed_road_transition(state.ns_cars, self.new_car_prob_ns, rng),
                self.closed_road_transition(state.ew_cars, self.new_car_prob_ew, rng),
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

    fn get_possible_actions(&self, current_state: State) -> Vec<LightAction> {
        match current_state.light_state {
            LightState::NorthSouthOpen | LightState::EastWestOpen => {
                vec![LightAction::Change, LightAction::Stay]
            }
            LightState::ChangingToNS | LightState::ChangingToEW => vec![LightAction::WaitForChange],
        }
    }

    fn get_all_state_actions(&self) -> &[(State, LightAction)] {
        &self.states_actions
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
        0.8
    }
}
