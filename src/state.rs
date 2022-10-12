use std::{collections::VecDeque, sync::Arc};

use enum_iterator::Sequence;
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

#[derive(Debug, Copy, Clone, Ord, PartialOrd, PartialEq, Eq, Sequence)]
pub enum Weapon {
    Spitfire,
    Mozambique,
    RE45,
    R301,
    Sentinel,
    TripleTake,
    Devotion,
    Longbow,
    LStar,
    Peacekeeper,
    Mastiff,
    EVA8,
    Volt,
    Bocek,
    Kraber,
    Rampage,
    Wingman,
    Hemlock,
    Prowler,
    Flatline,
    P2020,
    N3030,
    Car,
}

impl Weapon {
    pub fn match_string(s: String) -> Option<(Weapon, f64)> {
        let weapons = enum_iterator::all::<Weapon>().collect::<Vec<_>>();

        if let Some(max) = weapons
            .iter()
            .cloned()
            .map(|w| {
                (
                    w,
                    strsim::jaro_winkler(
                        &s.to_string().to_lowercase(),
                        &w.to_string().to_lowercase(),
                    ),
                )
            })
            .max_by(|w1, w2| w1.1.total_cmp(&w2.1))
        {
            if max.1 > 0.75 {
                Some(max)
            } else {
                None
            }
        } else {
            None
        }
    }
}

impl ToString for Weapon {
    fn to_string(&self) -> String {
        let s = match self {
            Weapon::Spitfire => "Spitfire",
            Weapon::Mozambique => "Mozambique",
            Weapon::RE45 => "RE-45",
            Weapon::R301 => "R-301",
            Weapon::Sentinel => "Sentinel",
            Weapon::TripleTake => "TripleTake",
            Weapon::Longbow => "Longbow",
            Weapon::LStar => "L-STAR",
            Weapon::Peacekeeper => "Peacekeeper",
            Weapon::Mastiff => "Mastiff",
            Weapon::EVA8 => "EVA-8",
            Weapon::Volt => "Volt",
            Weapon::Bocek => "Bocek",
            Weapon::Kraber => "Kraber",
            Weapon::Rampage => "Rampage",
            Weapon::Wingman => "Wingman",
            Weapon::Hemlock => "Hemlock",
            Weapon::Prowler => "Prowler",
            Weapon::Flatline => "Flatline",
            Weapon::P2020 => "P-2020",
            Weapon::N3030 => "30-30",
            Weapon::Car => "Car",
            Weapon::Devotion => "Devotion",
        };

        s.to_string()
    }
}
