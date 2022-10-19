use opencv::core::{Rect_, Vector};
use std::collections::{hash_map::IntoIter, HashMap};

#[derive(Clone, Debug)]
pub enum RegionOfInterestType {
    Text,
}

#[derive(Clone, Debug)]
pub struct RegionOfInterest {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
    pub roi_type: RegionOfInterestType,
    pub result: Option<String>,
    pub name: String,
    pub base_resolution: StreamResolution,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum StreamResolution {
    HD720p,
    HD1080p,
}

pub fn new_region(
    name: String,
    x: i32,
    y: i32,
    width: i32,
    height: i32,
    res: StreamResolution,
) -> RegionOfInterest {
    RegionOfInterest {
        x,
        y,
        width,
        height,
        roi_type: RegionOfInterestType::Text,
        result: None,
        name,
        base_resolution: res,
    }
}

impl RegionOfInterest {
    pub fn set_result(&mut self, result: String) {
        self.result = Some(result);
    }
}

#[derive(Clone, Debug)]
pub struct RegionOfInterestList {
    list: HashMap<String, RegionOfInterest>,
}

pub fn new_region_list() -> RegionOfInterestList {
    RegionOfInterestList {
        list: HashMap::new(),
    }
}

impl RegionOfInterestList {
    pub fn add_region(&mut self, region: RegionOfInterest) {
        self.list.insert(region.clone().name, region);
    }

    pub fn add_new_region(
        &mut self,
        name: String,
        x: i32,
        y: i32,
        width: i32,
        height: i32,
        res: StreamResolution,
    ) {
        let region = new_region(name, x, y, width, height, res);

        self.add_region(region);
    }

    pub fn get_value(&self, region_name: String) -> Option<&RegionOfInterest> {
        self.list.get(&region_name)
    }

    pub fn iter(&self) -> IntoIter<String, RegionOfInterest> {
        self.list.clone().into_iter()
    }

    pub fn len(&self) -> i32 {
        self.list.len() as i32
    }

    pub fn is_empty(&self) -> bool {
        self.list.is_empty()
    }

    pub fn get_log(&self) -> String {
        self.list
            .iter()
            .map(|i| format!("{}\t{}", i.1.name, i.1.result.clone().unwrap()))
            .collect()
    }

    pub fn vec_of_rects(&self) -> Vector<Rect_<i32>> {
        self.list
            .iter()
            .map(|i| Rect_::new(i.1.x, i.1.y, i.1.width, i.1.height))
            .collect()
    }
}
