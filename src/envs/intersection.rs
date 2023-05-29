use std::collections::BTreeMap;

use crate::mdp::{self, Probability, Reward, State, Transition};

enum IntersectionAction {
    Stay = 0,
    Change = 1,
}

#[derive(Debug)]
enum TrafficSignalState {
    NorthSouthOpen = 0,
    EastWestOpen = 1,
}

pub fn build_mdp(max_cars: usize) {
    let mut transitions: BTreeMap<(State, mdp::Action), Vec<Transition>> = BTreeMap::new();

    let mut states = vec![];

    let car_combinations = (max_cars + 1).pow(2);
    for ns_cars in 0..=max_cars {
        for ew_cars in 0..=max_cars {
            let index = ns_cars * (max_cars + 1) + ew_cars;

            let ns_state = State(index);
            states.push(ns_state);

            let ew_index = index + car_combinations;

            println!(
                "State: {:?}, ns_cars: {:?}, ew_cars: {:?}",
                ns_state, ns_cars, ew_cars
            );
            reconstruct_state(ns_state, max_cars);
            let ew_state = State(ew_index);
            reconstruct_state(ew_state, max_cars);
            states.push(ew_state);
        }
    }

    println!("{:?}", states);
}

type IntersectionState = (TrafficSignalState, usize, usize);

fn reconstruct_state(state: State, max_cars: usize) -> IntersectionState {
    let mut index = state.0;

    let combinations = (max_cars + 1).pow(2);
    let intersection_state = if index < combinations {
        TrafficSignalState::NorthSouthOpen
    } else {
        index -= combinations;
        TrafficSignalState::EastWestOpen
    };

    let ew_cars = index % (max_cars + 1);
    let ns_cars = index / (max_cars + 1);

    println!(
        "function, ns_cars: {:?}, ew_cars: {:?}, intersection_state: {:?}",
        ns_cars, ew_cars, intersection_state
    );

    return (intersection_state, ns_cars, ew_cars);
}

fn calculate_reward(action: IntersectionAction, intersection_state: IntersectionState) -> Reward {
    let (state, ns_cars, ew_cars) = intersection_state;
    match (action, state) {
        (IntersectionAction::Change, TrafficSignalState::NorthSouthOpen) => {
            if ew_cars >= 2 * ns_cars {
                1.0
            } else {
                -1.0
            }
        }
        (IntersectionAction::Change, TrafficSignalState::EastWestOpen) => {
            if ns_cars >= 2 * ew_cars {
                1.0
            } else {
                -1.0
            }
        }
        (IntersectionAction::Stay, TrafficSignalState::NorthSouthOpen) => {
            if ew_cars >= 2 * ns_cars {
                0.0
            } else {
                1.0
            }
        }
        (IntersectionAction::Stay, TrafficSignalState::EastWestOpen) => {
            if ns_cars >= 2 * ew_cars {
                0.0
            } else {
                1.0
            }
        }
    }
}

fn calc_change_prob(cars: usize, car_change: usize) -> Probability {
    let new_cars = cars + car_change;

    if new_cars == cars {
        1.0 / 2.0_f64.sqrt()
    } else if new_cars == cars + 1 || new_cars == cars - 1 {
        let divisor = if cars == 0 { 1.0 } else { 2.0 };
        (1.0 - (1.0 / 2.0_f64.sqrt())) * (1.0 / divisor)
    } else {
        0.0
    }
}
