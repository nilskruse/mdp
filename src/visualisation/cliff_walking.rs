use std::collections::BTreeMap;

use eframe::egui;
use egui::{Color32, Pos2, Rgba};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use crate::{
    envs::cliff_walking::{COLS, ROWS},
    mdp::{Action, Mdp, State},
    policies::greedy_policy,
};

pub fn show_strategy(
    mdp: &Mdp,
    q_map: &BTreeMap<(State, Action), f64>,
) -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        // initial_window_size: Some(egui::vec2(320.0, 240.0)),
        ..Default::default()
    };

    let myapp = MyApp::new(mdp.clone(), q_map.clone());
    eframe::run_native("My egui App", options, Box::new(|_cc| Box::from(myapp)))
}

struct MyApp {
    mdp: Mdp,
    q_map: BTreeMap<(State, Action), f64>,
}

impl MyApp {
    fn new(mdp: Mdp, q_map: BTreeMap<(State, Action), f64>) -> Self {
        // Customize egui here with cc.egui_ctx.set_fonts and cc.egui_ctx.set_visuals.
        // Restore app state using cc.storage (requires the "persistence" feature).
        // Use the cc.gl (a glow::Context) to create graphics shaders and buffers that you can use
        // for e.g. egui::PaintCallback.
        //
        Self { mdp, q_map }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Cliff walking greedy strategy");

            let size = 20.0;
            let spacing = -3.0;

            let grid = egui::Grid::new("some_unique_id")
                .min_col_width(size)
                .max_col_width(size)
                .min_row_height(size)
                .spacing((spacing, spacing));

            grid.show(ui, |ui| {
                let mut rng = ChaCha20Rng::seed_from_u64(0);
                for row in 0..ROWS {
                    for col in 0..COLS {
                        let state_index = row * COLS + col;
                        let action =
                            greedy_policy(&self.mdp, &self.q_map, State(state_index), &mut rng);
                        let label = match action {
                            Some(Action(0)) => "⬆",
                            Some(Action(1)) => "⬇",
                            Some(Action(2)) => "⬅",
                            Some(Action(3)) => "➡",
                            _ => "none",
                        };
                        // let label = format!("{}", state_index);
                        let color = match state_index {
                            36 => Color32::BLUE,
                            37..=46 => Color32::RED,
                            47 => Color32::GREEN,
                            _ => Color32::LIGHT_BLUE,
                        };

                        let richtext = egui::RichText::new(label).background_color(color);
                        ui.label(richtext);
                        // let cell_pos = Pos2::new(col as f32 * size , row as f32 * size);
                        // let cell_rect = egui::Rect::from_min_size(cell_pos, egui::vec2(size, size));
                        // ui.painter().rect_filled(cell_rect, 0.0, Color32::RED);

                        // ui.st
                        // ui.label(format!("{}", label));
                    }
                    ui.end_row();
                }
            });
        });
    }
}
