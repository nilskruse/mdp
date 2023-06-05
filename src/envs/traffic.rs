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
