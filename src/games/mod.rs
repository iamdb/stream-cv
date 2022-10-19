use crate::{img::frame::Frame, roi::RegionOfInterestList};

pub mod apex;

#[derive(Debug)]
pub enum SupportedGames {
    Apex,
}

pub trait Game {
    fn regions(&self) -> RegionOfInterestList;
    fn extract_data(&self, frame: Frame);
    fn output(&self) -> &str;
}

pub fn new(game: SupportedGames) -> impl Game {
    match game {
        SupportedGames::Apex => apex::new(),
    }
}
