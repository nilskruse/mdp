#![allow(dead_code)]

#[macro_use]
extern crate assert_float_eq;

pub mod algorithms;
pub mod envs;
pub mod eval;
pub mod generator;
pub mod mdp;
pub mod policies;
pub mod utils;

#[cfg(test)]
pub mod tests;

pub mod benchmarks;

pub mod experiments;

pub mod visualisation;

pub mod multiagent;
