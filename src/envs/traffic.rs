#![allow(unused_variables)]
use std::slice::Iter;

use rand::Rng;

use crate::mdp::GenericMdp;

const NUM_FLOWS: usize = 4;
const NUM_LIGHTS: usize = 2;
const MAX_CARS: usize = 4;

#[derive(Debug, Ord, PartialOrd, Eq, PartialEq, Clone, Hash, Copy)]
struct State {
    vehicles: [usize; NUM_FLOWS],
    lights: [Light; NUM_LIGHTS],
}

#[derive(Debug, Ord, PartialOrd, Eq, PartialEq, Clone, Hash, Copy)]
enum Light {
    Green = 0,
    Yellow = 1,
    AllRed = 2,
    Red = 3,
}

type Action = Light;

pub struct TrafficMdp {
    arrival_probs: [f64; NUM_FLOWS],
}

impl TrafficMdp {
    pub fn new(arrival_probs: [f64; NUM_FLOWS]) -> Self {
        Self { arrival_probs }
    }

    fn green_transition<R: Rng>(old_cars: usize, new_prob: f64, rng: &mut R) -> usize {
        if old_cars == 0 {
            0
        } else if rng.gen_range(0.0..1.0) < (1.0 - new_prob) {
            old_cars - 1
        } else {
            old_cars
        }
    }

    fn red_transition<R: Rng>(old_cars: usize, new_prob: f64, rng: &mut R) -> usize {
        if old_cars == MAX_CARS {
            MAX_CARS
        } else if rng.gen_range(0.0..1.0) < (1.0 - new_prob) {
            old_cars
        } else {
            old_cars + 1
        }
    }
}

impl GenericMdp<State, Action> for TrafficMdp {
    fn perform_action<R: rand::SeedableRng + rand::Rng>(
        &self,
        state_action: (State, Action),
        rng: &mut R,
    ) -> (State, crate::mdp::Reward) {
        let (state, action) = state_action;
        for flow in 0..NUM_FLOWS {
            let old_cars = state.vehicles[flow];
            let new_cars = match action {
                Light::Green => Self::green_transition(old_cars, self.arrival_probs[flow], rng),
                Light::Yellow | Light::AllRed | Light::Red => todo!(),
            };
        }
        todo!()
    }

    fn get_possible_actions(&self, current_state: State) -> Vec<Action> {
        todo!()
    }

    fn get_all_state_actions(&self) -> Vec<(State, Action)> {
        todo!()
    }

    fn is_terminal(&self, state: State) -> bool {
        todo!()
    }

    fn get_initial_state<R: rand::Rng + rand::SeedableRng>(&self, rng: &mut R) -> State {
        todo!()
    }

    fn get_discount_factor(&self) -> f64 {
        todo!()
    }
}
