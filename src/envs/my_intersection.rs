#![allow(unused_variables)]

use rand::Rng;

use crate::mdp::GenericMdp;

#[derive(Debug, Ord, PartialOrd, Eq, PartialEq, Clone, Hash, Copy)]
pub struct IntersectionState {
    pub light_state: LightState,
    pub ns_cars: usize,
    pub ew_cars: usize,
}

#[derive(Debug, Ord, PartialOrd, Eq, PartialEq, Clone, Hash, Copy)]
pub enum LightState {
    NorthSouthOpen = 0,
    EastWestOpen = 1,
    ChangingToNS = 2,
    ChangingToEW = 3,
}

#[derive(Debug, Ord, PartialOrd, Eq, PartialEq, Clone, Hash, Copy)]
pub enum LightAction {
    Change = 0,
    Stay = 1,
    WaitForChange = 2,
}

pub struct MyIntersectionMdp {
    new_car_prob_ns: f64,
    new_car_prob_ew: f64,
    max_cars: usize,
    states_actions: Vec<(IntersectionState, LightAction)>,
}

impl MyIntersectionMdp {
    pub fn new(new_car_prob_ns: f64, new_car_prob_ew: f64, max_cars: usize) -> Self {
        let mut states = vec![];
        for ns_cars in 0..=max_cars {
            for ew_cars in 0..=max_cars {
                states.push(IntersectionState {
                    light_state: LightState::NorthSouthOpen,
                    ns_cars,
                    ew_cars,
                });
                states.push(IntersectionState {
                    light_state: LightState::EastWestOpen,
                    ns_cars,
                    ew_cars,
                });
                states.push(IntersectionState {
                    light_state: LightState::ChangingToNS,
                    ns_cars,
                    ew_cars,
                });
                states.push(IntersectionState {
                    light_state: LightState::ChangingToEW,
                    ns_cars,
                    ew_cars,
                });
            }
        }

        let mut states_actions: Vec<(IntersectionState, LightAction)> = vec![];

        states.iter().for_each(|s| match s.light_state {
            LightState::NorthSouthOpen | LightState::EastWestOpen => {
                states_actions.push((*s, LightAction::Stay));
                states_actions.push((*s, LightAction::Change));
            }
            LightState::ChangingToNS | LightState::ChangingToEW => {
                states_actions.push((*s, LightAction::WaitForChange));
            }
        });

        println!(
            "single agent states_actions count: {:?}",
            states_actions.len()
        );
        Self {
            new_car_prob_ns,
            new_car_prob_ew,
            max_cars,
            states_actions,
        }
    }

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

impl GenericMdp<IntersectionState, LightAction> for MyIntersectionMdp {
    fn perform_action<R: Rng>(
        &self,
        state_action: (IntersectionState, LightAction),
        rng: &mut R,
    ) -> (IntersectionState, crate::mdp::Reward) {
        let (state, action) = state_action;

        let new_light_state = match action {
            LightAction::Change => match state.light_state {
                LightState::NorthSouthOpen => LightState::ChangingToEW,
                LightState::EastWestOpen => LightState::ChangingToNS,
                LightState::ChangingToNS | LightState::ChangingToEW => {
                    panic!("Unreachable state: can't change light mid-cycle")
                }
            },
            LightAction::Stay => state.light_state,
            LightAction::WaitForChange => match state.light_state {
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

        let new_state = IntersectionState {
            light_state: new_light_state,
            ns_cars: new_ns_cars,
            ew_cars: new_ew_cars,
        };

        let reward: f64 = -((new_ns_cars + new_ew_cars) as f64);

        (new_state, reward)
    }

    fn get_possible_actions(&self, current_state: IntersectionState) -> Vec<LightAction> {
        match current_state.light_state {
            LightState::NorthSouthOpen | LightState::EastWestOpen => {
                vec![LightAction::Change, LightAction::Stay]
            }
            LightState::ChangingToNS | LightState::ChangingToEW => vec![LightAction::WaitForChange],
        }
    }

    fn get_all_state_actions(&self) -> &[(IntersectionState, LightAction)] {
        &self.states_actions
    }

    fn is_terminal(&self, state: IntersectionState) -> bool {
        false
    }

    fn get_initial_state<R: Rng>(&self, rng: &mut R) -> IntersectionState {
        IntersectionState {
            light_state: LightState::NorthSouthOpen,
            ns_cars: 0,
            ew_cars: 0,
        }
    }

    fn get_discount_factor(&self) -> f64 {
        0.8
    }
}
