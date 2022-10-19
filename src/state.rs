use std::{collections::VecDeque, sync::Arc};

use tokio::sync::Mutex;

pub type StringWindow = Arc<Mutex<VecDeque<String>>>;

#[derive(Clone, Debug)]
pub struct GameState {
    _weapon1: StringWindow,
    _weapon2: StringWindow,
}

pub fn new() -> GameState {
    GameState {
        _weapon1: Arc::new(Mutex::new(VecDeque::new())),
        _weapon2: Arc::new(Mutex::new(VecDeque::new())),
    }
}
