use itertools::Itertools;
use std::{cmp::Ordering, collections::BTreeMap};

use crate::mdp::{GenericAction, GenericState, IndexMdp, Reward};

// print q_map in order
pub fn print_q_map<S: GenericState, A: GenericAction>(q_map: &BTreeMap<(S, A), Reward>) {
    q_map
        .iter()
        // .sorted_by(|pair1, pair2| {
        //     let state1 = pair1.0 .0 .0;
        //     let state2 = pair2.0 .0 .0;
        //     let action1 = pair1.0 .1 .0;
        //     let action2 = pair2.0 .1 .0;
        //     if state1 > state2 {
        //         Ordering::Greater
        //     } else if state1 < state2 {
        //         Ordering::Less
        //     } else if action1 > action2 {
        //         Ordering::Greater
        //     } else if action1 < action2 {
        //         Ordering::Less
        //     } else {
        //         Ordering::Equal
        //     }
        // })
        // .sorted()
        .for_each(|q_entry| println!("{:?}", q_entry));
}

pub fn print_transition_map(mdp: &IndexMdp) {
    let _transitions = mdp.transitions.clone();
    mdp.transitions
        .iter()
        .sorted_by(|pair1, pair2| {
            let state1 = pair1.0 .0 .0;
            let state2 = pair2.0 .0 .0;
            let action1 = pair1.0 .1 .0;
            let action2 = pair2.0 .1 .0;
            if state1 > state2 {
                Ordering::Greater
            } else if state1 < state2 {
                Ordering::Less
            } else if action1 > action2 {
                Ordering::Greater
            } else if action1 < action2 {
                Ordering::Less
            } else {
                Ordering::Equal
            }
        })
        .for_each(|entry| println!("{:?}", entry));
}
