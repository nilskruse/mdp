use std::collections::BTreeMap;

use crate::{
    mdp::{self, Probability, Reward, State, Transition},
    utils::print_transition_map,
};

enum IntersectionAction {
    Stay = 0,
    Change = 1,
}

#[derive(Debug, Copy, Clone)]
enum TrafficSignalState {
    NorthSouthOpen = 0,
    EastWestOpen = 1,
}

pub fn build_mdp(max_cars: isize) {
    if max_cars < 1 {
        panic!("max_cars needs to be at least 1");
    }
    let mut transitions: BTreeMap<(State, mdp::Action), Vec<Transition>> = BTreeMap::new();

    let mut states = vec![];

    let car_combinations = (max_cars + 1).pow(2);
    for ns_cars in 0..=max_cars {
        for ew_cars in 0..=max_cars {
            let ns_state = build_state(
                TrafficSignalState::NorthSouthOpen,
                ns_cars,
                ew_cars,
                max_cars,
            );
            let ew_state =
                build_state(TrafficSignalState::EastWestOpen, ns_cars, ew_cars, max_cars);

            states.push(ns_state);
            println!(
                "State: {:?}, ns_cars: {:?}, ew_cars: {:?}",
                ns_state, ns_cars, ew_cars
            );
            reconstruct_state(ns_state, max_cars);
            reconstruct_state(ew_state, max_cars);
            states.push(ew_state);
        }
    }

    println!("{:?}", states);
    println!("{:?}", f(3, 3));
    u(3, 3);
}

type IntersectionState = (TrafficSignalState, isize, isize);

fn build_state(
    traffic_signal_state: TrafficSignalState,
    ns_cars: isize,
    ew_cars: isize,
    max_cars: isize,
) -> State {
    match traffic_signal_state {
        TrafficSignalState::NorthSouthOpen => State((ns_cars * (max_cars + 1) + ew_cars) as usize),
        TrafficSignalState::EastWestOpen => {
            State((ns_cars * (max_cars + 1) + ew_cars + (max_cars + 1).pow(2)) as usize)
        }
    }
}

fn reconstruct_state(state: State, max_cars: isize) -> IntersectionState {
    let mut index = state.0 as isize;

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

    (intersection_state, ns_cars, ew_cars)
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

fn calc_change_prob(cars: isize, car_change: isize) -> Probability {
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

fn calc_stay_prob_open_road(car_change: isize) -> Probability {
    match car_change {
        -2 => 3.0 / 16.0,
        -1 | 0 => 9.0 / 32.0,
        2 => 1.0 / 12.0,
        1 => 1.0 / 6.0,
        _ => 0.0,
    }
}

fn calc_stay_prob_closed_road(car_change: isize) -> Probability {
    match car_change {
        -2 => 1.0 / 60.0,
        -1 => 1.0 / 30.0,
        2 => 19.0 / 80.0,
        1 | 0 => 57.0 / 160.0,
        _ => 0.0,
    }
}

fn f(ns_cars: isize, ew_cars: isize) {
    let mut total_prob: f64 = 0.0;
    for i in -2..=2 {
        for j in -2..=2 {
            total_prob += calc_change_prob(ns_cars, i) * calc_change_prob(ew_cars, j);
        }
    }
    println!("total_prob: {total_prob}");
}

fn u(ns_cars: isize, ew_cars: isize) {
    let mut total_prob: f64 = 0.0;
    for i in -2..=2 {
        for j in -2..=2 {
            total_prob += calc_stay_prob_open_road(i) * calc_stay_prob_closed_road(j);
        }
    }
    println!("total_prob: {total_prob}");
}

fn build_transitions(
    intersection_state: IntersectionState,
    intersection_action: IntersectionAction,
    max_cars: isize,
) -> Vec<Transition> {
    let (traffic_signal_state, ns_cars, ew_cars) = intersection_state;
    let mut transitions = vec![];

    let new_traffic_signal_state = match intersection_action {
        IntersectionAction::Stay => traffic_signal_state,
        IntersectionAction::Change => match traffic_signal_state {
            TrafficSignalState::NorthSouthOpen => TrafficSignalState::EastWestOpen,
            TrafficSignalState::EastWestOpen => TrafficSignalState::NorthSouthOpen,
        },
    };

    for ns_car_change in -2..=2 {
        for ew_car_change in -2..=2 {
            let new_state = build_state(
                new_traffic_signal_state,
                ns_cars + ns_car_change,
                ew_cars + ew_car_change,
                max_cars,
            );
            let prob = match intersection_action {
                IntersectionAction::Stay => match traffic_signal_state {
                    TrafficSignalState::NorthSouthOpen => {
                        calc_stay_prob_open_road(ns_car_change)
                            * calc_stay_prob_closed_road(ew_car_change)
                    }
                    TrafficSignalState::EastWestOpen => {
                        calc_stay_prob_open_road(ew_car_change)
                            * calc_stay_prob_closed_road(ns_car_change)
                    }
                },
                IntersectionAction::Change => {
                    calc_change_prob(ns_cars, ns_car_change)
                        * calc_change_prob(ew_cars, ew_car_change)
                }
            };
        }
    }

    transitions
}
