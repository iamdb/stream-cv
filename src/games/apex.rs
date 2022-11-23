use enum_iterator::Sequence;

use crate::{
    games::Game,
    roi::{self, StreamResolution},
};

pub struct Apex;

pub fn new() -> Apex {
    Apex
}

impl Game for Apex {
    fn regions(&self) -> crate::roi::RegionOfInterestList {
        let mut list = roi::new_region_list();

        list.add_new_region(
            "loaded_mag_size".to_string(),
            1720,
            960,
            62,
            40,
            StreamResolution::HD1080p,
        );
        list.add_new_region(
            "total_ammo".to_string(),
            1720,
            998,
            62,
            30,
            StreamResolution::HD1080p,
        );
        list.add_new_region(
            "weapon_1_name".to_string(),
            1555,
            1034,
            110,
            24,
            StreamResolution::HD1080p,
        );
        list.add_new_region(
            "weapon_2_name".to_string(),
            1715,
            1034,
            110,
            24,
            StreamResolution::HD1080p,
        );
        list.add_new_region(
            "compass_number".to_string(),
            935,
            90,
            50,
            32,
            StreamResolution::HD1080p,
        );

        list
    }

    fn extract_data(&self, _frame: crate::img::frame::Frame) {
        todo!()
    }

    fn output(&self) -> &str {
        todo!()
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
    G7Scout,
    Havoc,
}

impl Weapon {
    pub fn match_string(mut s: String) -> Option<(Weapon, f64)> {
        let weapons = enum_iterator::all::<Weapon>().collect::<Vec<_>>();
        s.make_ascii_lowercase();

        if let Some(max) = weapons
            .iter()
            .cloned()
            .map(|w| {
                (
                    w,
                    strsim::jaro_winkler(&s.to_string(), &w.to_string().to_lowercase()),
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
            Weapon::G7Scout => "G7 Scout",
            Weapon::Havoc => "Havoc",
        };

        s.to_string()
    }
}
