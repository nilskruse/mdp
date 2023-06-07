use std::{collections::BTreeMap, iter::zip};

use rand::{Rng, SeedableRng};

use crate::{
    mdp::{GenericAction, GenericMdp, GenericState, IndexAction, IndexState},
    policies::{epsilon_greedy_policy, greedy_policy},
    utils::print_q_map,
};

use super::GenericStateActionAlgorithm;

pub struct QLearningDynamic {
    alpha: f64,
    epsilon: f64,
    max_steps: usize,
}

impl QLearningDynamic {
    pub fn new(alpha: f64, epsilon: f64, max_steps: usize) -> Self {
        QLearningDynamic {
            alpha,
            epsilon,
            max_steps,
        }
    }
}

impl GenericStateActionAlgorithm for QLearningDynamic {
    fn run_with_q_map<
        M: GenericMdp<S, A>,
        S: GenericState,
        A: GenericAction,
        R: Rng + SeedableRng,
    >(
        &self,
        mdp: &M,
        episodes: usize,
        rng: &mut R,
        q_map: &mut BTreeMap<(S, A), f64>,
    ) {
        let mut prev_q_map: BTreeMap<(S, A), f64> = BTreeMap::new();

        mdp.get_all_state_actions().iter().for_each(|state_action| {
            prev_q_map.insert(*state_action, 0.0);
        });

        for episode in 1..=episodes {
            let mut alpha = self.alpha;
            if episode > 1 {
                // calculate the mean squared error
                let mut acc = 0.0;
                let mut max = f64::MIN;
                let non_zero = q_map
                    .values()
                    .fold(0, |acc, elem| if *elem != 0.0 { acc + 1 } else { acc });

                for (entry1, entry2) in zip(prev_q_map.iter(), q_map.iter()) {
                    acc += (*entry1.1 - *entry2.1).powi(2);
                    if *entry2.1 > max {
                        max = *entry2.1;
                    }
                }
                acc /= q_map.len() as f64;
                // acc = acc / (max + 0.0001);

                println!("max = {max}");
                println!("acc = {acc}");
                println!("Non-zero = {non_zero}");
                alpha += acc;
                alpha = alpha.clamp(self.alpha / 2.0, self.alpha * 2.0);
            }
            println!("alpha = {alpha}");

            // prev_q_map.extend(q_map.iter());
            prev_q_map = q_map.clone();
            // println!("q_map:");
            // print_q_map(q_map);
            // println!("prev_q_map:");
            // print_q_map(&prev_q_map);

            let mut current_state = mdp.get_initial_state(rng);
            let mut steps = 0;

            while !mdp.is_terminal(current_state) && steps < self.max_steps {
                let Some(selected_action) = epsilon_greedy_policy(mdp, q_map, current_state, self.epsilon, rng) else {break};
                let (next_state, reward) =
                    mdp.perform_action((current_state, selected_action), rng);

                // update q_map
                let Some(best_action) = greedy_policy(mdp, q_map, next_state, rng) else {break};
                let best_q = *q_map
                    .get(&(next_state, best_action))
                    .expect("No qmap entry found");

                let current_q = q_map.entry((current_state, selected_action)).or_insert(0.0);
                *current_q =
                    *current_q + alpha * (reward + mdp.get_discount_factor() * best_q - *current_q);

                current_state = next_state;

                steps += 1;
            }
        }
    }
}
