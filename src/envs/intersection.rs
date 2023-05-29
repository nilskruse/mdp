use std::collections::BTreeMap;

use crate::mdp::{self, State, Transition};

enum IntersectionAction {
    Stay = 0,
    Change = 1,
}

#[derive(Debug)]
enum IntersectionState {
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

fn reconstruct_state(state: State, max_cars: usize) -> (IntersectionState, usize, usize) {
    let mut index = state.0;

    let combinations = (max_cars + 1).pow(2);
    let intersection_state = if index < combinations {
        IntersectionState::NorthSouthOpen
    } else {
        index -= combinations;
        IntersectionState::EastWestOpen
    };

    let ew_cars = index % (max_cars + 1);
    let ns_cars = index / (max_cars + 1);

    println!("function, ns_cars: {:?}, ew_cars: {:?}, intersection_state: {:?}", ns_cars, ew_cars, intersection_state);

    return (intersection_state, ns_cars, ew_cars);
}
