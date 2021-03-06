use std::slice::Iter;

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
    pub result_text: Option<String>,
    pub result_number: Option<i32>,
    pub name: String,
}

pub fn new_region(name: String, x: i32, y: i32, width: i32, height: i32) -> RegionOfInterest {
    RegionOfInterest {
        x,
        y,
        width,
        height,
        roi_type: RegionOfInterestType::Text,
        result_text: None,
        result_number: None,
        name,
    }
}

impl RegionOfInterest {
    pub fn set_text_result(&mut self, result: String) {
        self.result_text = Some(result);
    }
}

#[derive(Clone, Debug)]
pub struct RegionOfInterestList {
    list: Vec<RegionOfInterest>,
}

pub fn new_region_list() -> RegionOfInterestList {
    RegionOfInterestList { list: Vec::new() }
}

#[allow(dead_code)]
impl RegionOfInterestList {
    pub fn add_region(&mut self, region: RegionOfInterest) {
        self.list.push(region);
    }

    pub fn add_new_region(&mut self, name: String, x: i32, y: i32, width: i32, height: i32) {
        let region = new_region(name, x, y, width, height);

        self.add_region(region);
    }

    pub fn iter(&self) -> Iter<RegionOfInterest> {
        self.list.iter()
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
            .map(|i| format!("{}\t{}", i.name, i.result_text.clone().unwrap()))
            .collect()
    }
}
