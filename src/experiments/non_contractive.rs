use crate::{
    algorithms::{
        dyna_q::{BetaDynaQ, Dyna, DynaQ},
        GenericStateActionAlgorithm, GenericStateActionAlgorithmStateful,
    },
    eval::evaluate_greedy_policy,
    mdp::{GenericAction, GenericMdp, GenericState},
};
use std::collections::{BTreeMap, HashSet};

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

use crate::{
    algorithms::value_iteration::value_iteration,
    mdp::{IndexAction, IndexMdp, IndexState, Transition},
    policies::{epsilon_greedy_policy, greedy_policy},
    utils::{print_q_map, print_transition_map},
};

pub fn build_mdp(p: f64) -> IndexMdp {
    let transition_probabilities: BTreeMap<(IndexState, IndexAction), Vec<Transition>> =
        BTreeMap::from([
            (
                (IndexState(0), IndexAction(0)),
                vec![(p, IndexState(2), 1000.0), (1.0 - p, IndexState(3), 1.0)],
            ),
            (
                (IndexState(0), IndexAction(1)),
                vec![(1.0, IndexState(1), 0.0)],
            ),
            (
                (IndexState(1), IndexAction(1)),
                vec![(1.0, IndexState(0), 0.0)],
            ),
            (
                (IndexState(2), IndexAction(0)),
                vec![(1.0, IndexState(2), 0.0)],
            ),
            (
                (IndexState(3), IndexAction(0)),
                vec![(1.0, IndexState(3), 0.0)],
            ),
        ]);

    let terminal_states_vec = vec![IndexState(2), IndexState(3)];

    let terminal_states: HashSet<IndexState> =
        HashSet::from_iter(terminal_states_vec.iter().copied());
    let discount_factor = 1.0;
    let states_actions = vec![
        (IndexState(0), IndexAction(0)),
        (IndexState(0), IndexAction(1)),
        (IndexState(1), IndexAction(1)),
        (IndexState(2), IndexAction(0)),
        (IndexState(3), IndexAction(0)),
    ];

    IndexMdp {
        transitions: transition_probabilities,
        terminal_states,
        initial_state: IndexState(0),
        discount_factor,
        states_actions,
    }
}

pub fn run_experiment() {
    let mdp = build_mdp(0.001);
    let episodes = 1000000;

    let alpha = 0.1;
    let epsilon = 0.1;
    let k = 5;
    let max_steps = usize::MAX;
    let beta_rate = 1;

    // println!("Monte Carlo");
    // let mc_algo = MonteCarlo::new(epsilon, max_steps);
    // let mut rng = ChaCha20Rng::seed_from_u64(0);
    // let q_map = mc_algo.run(&mdp, episodes, &mut rng);
    // println!("Q-Table:");
    // print_q_map(&q_map);
    // write_csv("MC", &q_map);
    // println!();

    // println!("Q-Learning");
    // let q_algo = QLearning::new(alpha, epsilon, max_steps);
    // let mut rng = ChaCha20Rng::seed_from_u64(0);
    // let q_map = q_algo.run(&mdp, episodes, &mut rng);
    // println!("Q-Table:");
    // print_q_map(&q_map);
    // // write_csv("Q", &q_map);
    // println!();

    // println!("SARSA");
    // let sarsa_algo = Sarsa::new(alpha, epsilon, max_steps);
    // let mut rng = ChaCha20Rng::seed_from_u64(0);
    // // let q_map = q_algo.run(&mdp, episodes, &mut rng, Some(&rig));
    // let q_map = sarsa_algo.run(&mdp, episodes, &mut rng);
    // println!("Q-Table:");
    // print_q_map(&q_map);
    // write_csv("SARSA", &q_map);
    // println!();

    // println!("Q-Learning Beta");
    // let mut q_beta_algo = QLearningBeta::new(alpha, epsilon, max_steps, beta_rate);
    // let mut rng = ChaCha20Rng::seed_from_u64(0);
    // let q_map = q_beta_algo.run(&mdp, episodes, &mut rng);
    // println!("Q-Table:");
    // print_q_map(&q_map);
    // write_csv("q_beta", &q_map);
    // println!();

    // println!("Q-Learning clipped");
    // let q_clipped_algo = QLearningClipped::new(alpha, epsilon, max_steps, 0.1);
    // let mut rng = ChaCha20Rng::seed_from_u64(0);
    // let q_map = q_clipped_algo.run(&mdp, episodes, &mut rng);
    // println!("Q-Table:");
    // print_q_map(&q_map);
    // write_csv("q_clipped_0_5", &q_map);
    // println!();

    // println!("DynaQ, with direct learning");
    // let mut dyna_q_algo = DynaQ::new(alpha, epsilon, k, max_steps, false, true, &mdp);
    // let mut rng = ChaCha20Rng::seed_from_u64(0);
    // let q_map = dyna_q_algo.run(&mdp, episodes, &mut rng);
    // println!("Q-Table:");
    // print_q_map(&q_map);
    // write_csv("DynaQ_direct", &q_map);
    // println!();

    // println!("DynaQ, no direct learning");
    // let mut dyna_q_algo = DynaQ::new(alpha, epsilon, k, max_steps, false, false, &mdp);
    // let mut rng = ChaCha20Rng::seed_from_u64(0);
    // let q_map = dyna_q_algo.run(&mdp, episodes, &mut rng);
    // println!("Q-Table:");
    // print_q_map(&q_map);
    // println!();

    println!("BetaDynaQ, no converging alpha, direct learning step");
    let mut beta_dyna_q_algo = BetaDynaQ::new(alpha, epsilon, k, max_steps, false, &mdp, beta_rate);
    let mut rng = ChaCha20Rng::seed_from_u64(0);
    let q_map = beta_dyna_q_algo.run(&mdp, episodes, &mut rng);
    println!("Q-Table:");
    print_q_map(&q_map);
    write_csv("beta_dyna_q_direct", &q_map);
    println!();

    println!("BetaDynaQ, no converging alpha, no direct learning step");
    let mut beta_dyna_q_algo = BetaDynaQ::new_with_settings(
        alpha, epsilon, k, max_steps, false, &mdp, beta_rate, false, false,
    );
    let mut rng = ChaCha20Rng::seed_from_u64(0);
    let q_map = beta_dyna_q_algo.run(&mdp, episodes, &mut rng);
    println!("Q-Table:");
    print_q_map(&q_map);
    write_csv("beta_dyna_q_no_direct", &q_map);
    println!();

    // println!("BetaDynaQ, with converging alpha, direct learning step");
    // let mut beta_dyna_q_algo = BetaDynaQ::new_with_settings(
    //     alpha, epsilon, k, max_steps, false, &mdp, beta_rate, true, true,
    // );
    // let mut rng = ChaCha20Rng::seed_from_u64(0);
    // let q_map = beta_dyna_q_algo.run(&mdp, episodes, &mut rng);
    // println!("Q-Table:");
    // print_q_map(&q_map);
    // println!();

    // println!("BetaDynaQ, with converging alpha, no direct learning step");
    // let mut beta_dyna_q_algo = BetaDynaQ::new_with_settings(
    //     alpha, epsilon, k, max_steps, false, &mdp, beta_rate, true, false,
    // );
    // let mut rng = ChaCha20Rng::seed_from_u64(0);
    // let q_map = beta_dyna_q_algo.run(&mdp, episodes, &mut rng);
    // println!("Q-Table:");
    // print_q_map(&q_map);
    // println!();

    println!();
    println!("\nTransitions");
    print_transition_map(&mdp);
    println!();

    println!("Value iteration:");
    let values = value_iteration(&mdp, 0.001);
    println!("{:?}", values);
    println!();
}

pub struct QLearningClipped {
    alpha: f64,
    epsilon: f64,
    max_steps: usize,
    clip: f64,
}

impl QLearningClipped {
    pub fn new(alpha: f64, epsilon: f64, max_steps: usize, clip: f64) -> Self {
        QLearningClipped {
            alpha,
            epsilon,
            max_steps,
            clip,
        }
    }
}

impl GenericStateActionAlgorithm for QLearningClipped {
    fn run_with_q_map<M: GenericMdp<S, A>, S: GenericState, A: GenericAction, R: Rng>(
        &self,
        mdp: &M,
        episodes: usize,
        rng: &mut R,
        q_map: &mut BTreeMap<(S, A), f64>,
    ) {
        for _ in 1..=episodes {
            let mut current_state = mdp.get_initial_state(rng);
            let mut steps = 0;

            while !mdp.is_terminal(current_state) && steps < self.max_steps {
                let Some(selected_action) =
                    epsilon_greedy_policy(mdp, q_map, current_state, self.epsilon, rng)
                else {
                    break;
                };
                let (next_state, reward) =
                    mdp.perform_action((current_state, selected_action), rng);

                // update q_map
                let Some(best_action) = greedy_policy(mdp, q_map, next_state, rng) else {
                    break;
                };
                let best_q = *q_map
                    .get(&(next_state, best_action))
                    .expect("No qmap entry found");

                let current_q = q_map.entry((current_state, selected_action)).or_insert(0.0);
                *current_q += (self.alpha
                    * (reward + mdp.get_discount_factor() * best_q - *current_q))
                    .clamp(-self.clip, self.clip);

                // println!(
                //     "update: {:?}, state: {:?}, action: {:?}, clipped reward: {:?}, discounted: {:?}, current_q: {:?} ",
                //     self.alpha
                //         * (reward.clamp(-self.clip, self.clip)
                //             + mdp.get_discount_factor() * best_q
                //             - *current_q),
                //     current_state,
                //     selected_action,
                //     reward.clamp(-self.clip, self.clip),
                //     mdp.get_discount_factor() * best_q,
                //     *current_q
                // );

                current_state = next_state;

                steps += 1;
            }
        }
    }

    fn get_epsilon(&self) -> f64 {
        self.epsilon
    }
}

fn write_csv(algo_name: &str, q_map: &BTreeMap<(IndexState, IndexAction), f64>) {
    let mut path = format!("results/non_contractive_{algo_name}");
    path = path.replace(".", "_");
    path.push_str(".csv");
    dbg!(&path);
    let mut csv_writer = csv::Writer::from_path(path).expect("csv file error");
    csv_writer
        .write_record(["State-action", "0", "1"])
        .expect("csv write record error");

    csv_writer
        .serialize((
            "a",
            q_map.get(&(IndexState(0), IndexAction(0))),
            q_map.get(&(IndexState(1), IndexAction(0))),
        ))
        .expect("csv error");
    csv_writer
        .serialize((
            "b",
            q_map.get(&(IndexState(0), IndexAction(1))),
            q_map.get(&(IndexState(1), IndexAction(1))),
        ))
        .expect("csv error");
}

pub fn slippery() {
    let mdp = crate::envs::slippery_cliff_walking::build_mdp(0.1).unwrap();
    let episodes = 100000;

    let alpha = 0.1;
    let epsilon = 0.1;
    let k = 5;
    let max_steps = 200;
    let beta_rate = 1;

    println!("DynaQ, no converging alpha, direct learning step");
    let mut beta_dyna_q_algo = DynaQ::new(alpha, epsilon, k, max_steps, false, true, &mdp);
    let mut rng = ChaCha20Rng::seed_from_u64(0);
    let q_map = beta_dyna_q_algo.run(&mdp, episodes, &mut rng);
    println!("Q-Table:");
    let avg_reward = evaluate_greedy_policy(&mdp, &q_map, episodes, max_steps, &mut rng);
    dbg!(avg_reward);
    // print_q_map(&q_map);
    println!();

    println!("BetaDynaQ, no converging alpha, no direct learning step");
    let mut beta_dyna_q_algo = BetaDynaQ::new_with_settings(
        alpha, epsilon, k, max_steps, false, &mdp, beta_rate, false, true,
    );
    let mut rng = ChaCha20Rng::seed_from_u64(0);
    let q_map = beta_dyna_q_algo.run(&mdp, episodes, &mut rng);
    let avg_reward = evaluate_greedy_policy(&mdp, &q_map, episodes, max_steps, &mut rng);
    dbg!(avg_reward);
    println!("Q-Table:");
    // print_q_map(&q_map);
    println!();
}
