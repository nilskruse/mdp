#![allow(unused_variables)]
use std::{
    collections::{btree_map, BTreeMap},
    slice::Iter,
};

use itertools::{iproduct, Itertools};

use rand::{Rng, SeedableRng};

use crate::{
    algorithms::GenericStateActionAlgorithm, mdp::GenericMdp, policies::epsilon_greedy_policy_ma,
};

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
            car_range.clone(),
            car_range.clone(),
            car_range.clone(),
            car_range.clone()
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

    fn possible_light_actions(light_state: LightState) -> Vec<LightAction> {
        match light_state {
            LightState::NorthSouthOpen | LightState::EastWestOpen => {
                vec![LightAction::Change, LightAction::Stay]
            }
            LightState::ChangingToNS | LightState::ChangingToEW => vec![LightAction::WaitForChange],
        }
    }

    fn all_states(max_cars: u8) -> Vec<State> {
        let lightstates = [
            LightState::NorthSouthOpen,
            LightState::EastWestOpen,
            LightState::ChangingToNS,
            LightState::ChangingToEW,
        ];
        let car_range = 0..=max_cars;
        let states = iproduct!(
            lightstates.iter().cloned(),
            lightstates.iter().cloned(),
            car_range.clone(),
            car_range.clone(),
            car_range.clone(),
            car_range.clone()
        )
        .map(State::from);

        states.collect()
    }
}

impl GenericMdp<State, Action> for MAIntersectionMdp {
    fn perform_action<R: rand::Rng>(
        &self,
        state_action: (State, Action),
        rng: &mut R,
    ) -> (State, crate::mdp::Reward) {
        let (state, action) = state_action;
        let (action_1, action_2) = (action.0, action.1);

        let new_light_state_1 = Self::state_transfer(state.light_state_1, action_1);
        let new_light_state_2 = Self::state_transfer(state.light_state_2, action_2);

        // we need to consider that a car leaving the intersection in the east-west direction can
        // increase the number of cars in the other intersection's east-west road
        let (new_ns_cars_1, new_ew_cars_1) = match new_light_state_1 {
            LightState::NorthSouthOpen => (
                self.open_road_transition(state.ns_cars_1, self.new_car_prob_ns_1, rng),
                self.closed_road_transition(state.ew_cars_1, self.new_car_prob_ew_1, rng),
            ),
            LightState::EastWestOpen => (
                self.closed_road_transition(state.ns_cars_1, self.new_car_prob_ns_1, rng),
                self.open_road_transition(state.ew_cars_1, self.new_car_prob_ew_1, rng),
            ),
            LightState::ChangingToNS | LightState::ChangingToEW => (
                self.closed_road_transition(state.ns_cars_1, self.new_car_prob_ns_1, rng),
                self.closed_road_transition(state.ew_cars_1, self.new_car_prob_ew_1, rng),
            ),
        };

        let (new_ns_cars_2, new_ew_cars_2) = match new_light_state_2 {
            LightState::NorthSouthOpen => (
                self.open_road_transition(state.ns_cars_2, self.new_car_prob_ns_2, rng),
                self.closed_road_transition(state.ew_cars_2, self.new_car_prob_ew_2, rng),
            ),
            LightState::EastWestOpen => (
                self.closed_road_transition(state.ns_cars_2, self.new_car_prob_ns_2, rng),
                self.open_road_transition(state.ew_cars_2, self.new_car_prob_ew_2, rng),
            ),
            LightState::ChangingToNS | LightState::ChangingToEW => (
                self.closed_road_transition(state.ns_cars_2, self.new_car_prob_ns_2, rng),
                self.closed_road_transition(state.ew_cars_2, self.new_car_prob_ew_2, rng),
            ),
        };

        // assuming equal flow in both direction car passes with a 0.5 probability to the other
        // intersection
        let crossed_cars_1 = if new_ew_cars_1 < state.ew_cars_1 {
            let random_value = rng.gen_range(0.0..1.0);
            if random_value < 0.5 {
                if new_ew_cars_2 < self.max_cars {
                    1
                } else {
                    0
                }
            } else {
                0
            }
        } else {
            0
        };

        let crossed_cars_2 = if new_ew_cars_2 < state.ew_cars_2 {
            let random_value = rng.gen_range(0.0..1.0);
            if random_value < 0.5 {
                if new_ew_cars_1 < self.max_cars {
                    1
                } else {
                    0
                }
            } else {
                0
            }
        } else {
            0
        };

        let new_state = State {
            light_state_1: new_light_state_1,
            light_state_2: new_light_state_2,
            ns_cars_1: new_ns_cars_1,
            ew_cars_1: new_ew_cars_1 + crossed_cars_2,
            ns_cars_2: new_ns_cars_2,
            ew_cars_2: new_ew_cars_2 + crossed_cars_1,
        };

        let reward: f64 = -((new_ns_cars_1 + new_ns_cars_2 + new_ew_cars_1 + new_ew_cars_2) as f64);

        (new_state, reward)
    }

    fn get_possible_actions(&self, current_state: State) -> Vec<Action> {
        let light_actions_1 = match current_state.light_state_1 {
            LightState::NorthSouthOpen | LightState::EastWestOpen => {
                vec![LightAction::Change, LightAction::Stay]
            }
            LightState::ChangingToNS | LightState::ChangingToEW => vec![LightAction::WaitForChange],
        };

        let light_actions_2 = match current_state.light_state_2 {
            LightState::NorthSouthOpen | LightState::EastWestOpen => {
                vec![LightAction::Change, LightAction::Stay]
            }
            LightState::ChangingToNS | LightState::ChangingToEW => vec![LightAction::WaitForChange],
        };

        iproduct!(light_actions_1.iter(), light_actions_2.iter())
            .map(|(a1, a2)| Action(*a1, *a2))
            .collect()
    }

    fn get_all_state_actions(&self) -> &[(State, Action)] {
        &self.states_actions
    }

    fn is_terminal(&self, state: State) -> bool {
        false
    }

    fn get_initial_state<R: Rng>(&self, rng: &mut R) -> State {
        State {
            light_state_1: LightState::NorthSouthOpen,
            light_state_2: LightState::NorthSouthOpen,
            ns_cars_1: 0,
            ew_cars_1: 0,
            ns_cars_2: 0,
            ew_cars_2: 0,
        }
    }

    fn get_discount_factor(&self) -> f64 {
        0.8
    }
}

pub struct MAIntersectionRunner<G: GenericStateActionAlgorithm> {
    mdp: MAIntersectionMdp,
    agent_1: G,
    agent_2: G,
    max_steps: usize,
}

impl<G: GenericStateActionAlgorithm> MAIntersectionRunner<G> {
    pub fn new(
        new_car_prob_ns_1: f64,
        new_car_prob_ew_1: f64,
        new_car_prob_ns_2: f64,
        new_car_prob_ew_2: f64,
        max_cars: u8,
        agent_1: G,
        agent_2: G,
        max_steps: usize,
    ) -> Self {
        let mdp = MAIntersectionMdp::new(
            new_car_prob_ns_1,
            new_car_prob_ew_1,
            new_car_prob_ns_2,
            new_car_prob_ew_2,
            max_cars,
        );

        Self {
            mdp,
            agent_1,
            agent_2,
            max_steps,
        }
    }

    pub fn run<R: Rng>(
        &self,
        episodes: usize,
        q_map_1: &mut BTreeMap<(State, LightAction), f64>,
        q_map_2: &mut BTreeMap<(State, LightAction), f64>,
        rng: &mut R,
    ) {
        for _ in 0..episodes {
            let mut current_state: State = self.mdp.get_initial_state(rng);
            let mut steps = 0;

            while !self.mdp.is_terminal(current_state) && steps < self.max_steps {
                let light_state_1 = current_state.light_state_1;
                let light_state_2 = current_state.light_state_2;

                // retrieve possible actions for light 1
                let possible_actions_1 =
                    MAIntersectionMdp::possible_light_actions(current_state.light_state_1);

                // retrieve possible actions for light 2
                let possible_actions_2 =
                    MAIntersectionMdp::possible_light_actions(current_state.light_state_2);

                // select action for intersection 1
                let Some(selected_action_1) = epsilon_greedy_policy_ma(&possible_actions_1, q_map_1, current_state, self.agent_1.get_epsilon(), rng)
                else {
                    panic!("no action possible")
                };

                // select action for intersection 2
                let Some(selected_action_2) = epsilon_greedy_policy_ma(&possible_actions_2, q_map_2, current_state, self.agent_2.get_epsilon(), rng)
                else {
                    panic!("no action possible")
                };

                // execute combined action
                let (next_state, reward) = self.mdp.perform_action(
                    (current_state, Action(selected_action_1, selected_action_2)),
                    rng,
                );

                // perform learning steps
                let next_possible_actions_1 =
                    MAIntersectionMdp::possible_light_actions(next_state.light_state_1);

                let next_possible_actions_2 =
                    MAIntersectionMdp::possible_light_actions(next_state.light_state_2);

                // println!("agent 1");
                self.agent_1.step(
                    q_map_1,
                    &next_possible_actions_1,
                    current_state,
                    selected_action_1,
                    next_state,
                    reward,
                    self.mdp.get_discount_factor(),
                    rng,
                );

                // println!("agent 2");
                self.agent_2.step(
                    q_map_2,
                    &next_possible_actions_2,
                    current_state,
                    selected_action_2,
                    next_state,
                    reward,
                    self.mdp.get_discount_factor(),
                    rng,
                );

                // the usual
                current_state = next_state;
                steps += 1;
            }
        }
    }

    pub fn gen_q_maps(
        &self,
    ) -> (
        BTreeMap<(State, LightAction), f64>,
        BTreeMap<(State, LightAction), f64>,
    ) {
        let states: Vec<State> = MAIntersectionMdp::all_states(self.mdp.max_cars);
        let states_actions_1: Vec<(State, LightAction)> = states
            .iter()
            .flat_map(|state| {
                let possible_actions =
                    MAIntersectionMdp::possible_light_actions(state.light_state_1);
                possible_actions
                    .iter()
                    .map(|action| (*state, *action))
                    .collect_vec()
            })
            .collect();

        let states_actions_2: Vec<(State, LightAction)> = states
            .iter()
            .flat_map(|state| {
                let possible_actions =
                    MAIntersectionMdp::possible_light_actions(state.light_state_2);
                possible_actions
                    .iter()
                    .map(|action| (*state, *action))
                    .collect_vec()
            })
            .collect();

        let mut q_map_1: BTreeMap<(State, LightAction), f64> = BTreeMap::new();
        let mut q_map_2: BTreeMap<(State, LightAction), f64> = BTreeMap::new();

        states_actions_1.iter().for_each(|state_action| {
            q_map_1.insert(*state_action, 0.0);
        });

        states_actions_2.iter().for_each(|state_action| {
            q_map_2.insert(*state_action, 0.0);
        });
        (q_map_1, q_map_2)
    }

    pub fn eval_greedy<R: Rng>(
        &self,
        episodes: usize,
        q_map_1: &BTreeMap<(State, LightAction), f64>,
        q_map_2: &BTreeMap<(State, LightAction), f64>,
        max_steps: usize,
        rng: &mut R,
    ) -> f64 {
        let mut total_reward = 0.0;
        for _ in 0..episodes {
            let mut current_state = self.mdp.get_initial_state(rng);
            let mut episode_reward = 0.0;
            let mut steps = 0;

            while !self.mdp.is_terminal(current_state) && steps < max_steps {
                // retrieve possible actions for light 1
                let possible_actions_1 =
                    MAIntersectionMdp::possible_light_actions(current_state.light_state_1);

                // retrieve possible actions for light 2
                let possible_actions_2 =
                    MAIntersectionMdp::possible_light_actions(current_state.light_state_2);

                // select action for intersection 1
                let Some(selected_action_1) = epsilon_greedy_policy_ma(&possible_actions_1, q_map_1, current_state, self.agent_1.get_epsilon(), rng)
                else {
                    panic!("no action possible")
                };

                // select action for intersection 2
                let Some(selected_action_2) = epsilon_greedy_policy_ma(&possible_actions_2, q_map_2, current_state, self.agent_2.get_epsilon(), rng)
                else {
                    panic!("no action possible")
                };

                let combined_action = Action(selected_action_1, selected_action_2);

                let (next_state, reward) = self
                    .mdp
                    .perform_action((current_state, combined_action), rng);

                episode_reward += reward;
                current_state = next_state;
                steps += 1;
            }
            total_reward += episode_reward;
        }
        total_reward / episodes as f64
    }
}
