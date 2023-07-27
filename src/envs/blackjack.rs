#![allow(unused_variables)]
use std::slice::Iter;

use rand::{Rng, SeedableRng};

use crate::mdp::GenericMdp;

pub struct BlackjackMdp;

impl BlackjackMdp {
    pub fn draw_card<R: Rng>(&self, rng: &mut R) -> u8 {
        rng.gen_range(1..=10)
    }
}

//BlackjackState = (current_sum: 12-21, dealer card: 1-10, usable ace: true/false)
#[derive(Debug, Ord, PartialOrd, Eq, PartialEq, Clone, Hash, Copy)]
pub enum BlackjackState {
    Running(u8, u8, bool),
    Win,
    Draw,
    Loss,
}

#[derive(Debug, Ord, PartialOrd, Eq, PartialEq, Clone, Hash, Copy)]
pub enum BlackjackAction {
    Hit = 0,
    Stick = 1,
}

impl GenericMdp<BlackjackState, BlackjackAction> for BlackjackMdp {
    fn perform_action<R: rand::Rng>(
        &self,
        state_action: (BlackjackState, BlackjackAction),
        rng: &mut R,
    ) -> (BlackjackState, crate::mdp::Reward) {
        let (state, action) = state_action;
        let BlackjackState::Running(player, dealer, useable_ace)  = state else { panic!("what are you doing here")};

        let card = match action {
            BlackjackAction::Hit => Some(self.draw_card(rng)),
            BlackjackAction::Stick => None,
        };

        if let Some(card) = card {}

        todo!()
    }

    fn get_possible_actions(&self, _: BlackjackState) -> Vec<BlackjackAction> {
        vec![BlackjackAction::Hit, BlackjackAction::Stick]
    }

    fn get_all_state_actions(&self) -> &[(BlackjackState, BlackjackAction)] {
        todo!()
    }

    fn is_terminal(&self, state: BlackjackState) -> bool {
        match state {
            BlackjackState::Running(_, _, _) => false,
            BlackjackState::Win => true,
            BlackjackState::Draw => true,
            BlackjackState::Loss => true,
        }
    }

    fn get_initial_state<R: Rng>(&self, rng: &mut R) -> BlackjackState {
        let player_card_1 = self.draw_card(rng);
        let player_card_2 = self.draw_card(rng);
        let player_sum = player_card_1 + player_card_2;

        let dealer_card_1 = self.draw_card(rng);
        let dealer_card_2 = self.draw_card(rng);
        let dealer_sum = dealer_card_1 + dealer_card_2;

        // check for immediate end condition
        if player_sum == 11 {
            if dealer_sum == 11 {
                return BlackjackState::Draw;
            } else {
                return BlackjackState::Win;
            }
        }

        let player_state = if player_sum < 12 { 12 } else { player_sum };
        let dealer_state = dealer_card_2;
        let useable_ace = player_card_1 == 1 || player_card_2 == 1;

        BlackjackState::Running(player_state, dealer_state, useable_ace)
    }

    fn get_discount_factor(&self) -> f64 {
        1.0
    }
}
