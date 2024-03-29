use chrono::{DateTime, Utc};
use opencv::{
    core::{
        add_weighted, bitwise_and, bitwise_not, Mat, Point as OpenCVPoint, Range, Rect_, Scalar,
        Size as OpenCVSize, ToInputArray, ToInputOutputArray, ToOutputArray, UMat, UMatUsageFlags,
        Vector, BORDER_DEFAULT,
    },
    dnn,
    imgproc::{
        bilateral_filter, canny, cvt_color, dilate as dilate_image, get_structuring_element,
        put_text, rectangle, threshold, COLOR_BGR2GRAY, COLOR_GRAY2BGR, COLOR_GRAY2RGB,
        FONT_HERSHEY_SIMPLEX, MORPH_DILATE, THRESH_BINARY,
    },
    photo::{detail_enhance, inpaint, INPAINT_TELEA},
    prelude::*,
};
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::roi::{RegionOfInterest, RegionOfInterestList};

type Size = OpenCVSize;
type Point = OpenCVPoint;

#[derive(Clone)]
pub struct Frame {
    pub mat: Mat,
    pub processed_mat: UMat,
    pub num: i64,
    pub text: String,
    pub start_date: DateTime<Utc>,
    pub end_date: Option<DateTime<Utc>>,
    pub results: RegionOfInterestList,
}

unsafe impl Send for Frame {}
unsafe impl Sync for Frame {}

impl Eq for Frame {}

impl PartialEq for Frame {
    fn eq(&self, other: &Self) -> bool {
        self.num.eq(&other.num)
    }
}

impl PartialOrd for Frame {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.num.cmp(&other.num))
    }
}

impl Ord for Frame {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.num.cmp(&other.num)
    }
}

#[allow(unused)]
impl Frame {
    pub fn set_start_date(&mut self, date: DateTime<Utc>) {
        self.start_date = date;
    }

    pub async fn convert_to_gray(mut self) -> Frame {
        cvt_color(
            &self.processed_mat.input_array().unwrap(),
            &mut self.processed_mat.output_array().unwrap(),
            COLOR_BGR2GRAY,
            0,
        )
        .unwrap();

        self
    }

    pub async fn convert_to_bgr(mut self) -> Frame {
        cvt_color(
            &self.processed_mat.input_array().unwrap(),
            &mut self.processed_mat.output_array().unwrap(),
            COLOR_GRAY2BGR,
            0,
        )
        .unwrap();

        self
    }

    pub async fn convert_to_rgb(mut self) -> Frame {
        cvt_color(
            &self.processed_mat.input_array().unwrap(),
            &mut self.processed_mat.output_array().unwrap(),
            COLOR_GRAY2RGB,
            0,
        )
        .unwrap();

        self
    }

    pub async fn detail_enhance(&mut self, sigma_s: f32, sigma_r: f32) -> Frame {
        detail_enhance(
            &self.processed_mat.input_array().unwrap(),
            &mut self.processed_mat.output_array().unwrap(),
            sigma_s,
            sigma_r,
        )
        .unwrap();

        self.to_owned()
    }

    pub async fn bilateral_filter(&mut self, d: i32, sigma_color: f64, sigma_space: f64) -> Frame {
        let mut filtered = UMat::new(UMatUsageFlags::USAGE_DEFAULT);

        match bilateral_filter(
            &self.processed_mat.input_array().unwrap(),
            &mut filtered.output_array().unwrap(),
            d,
            sigma_color,
            sigma_space,
            BORDER_DEFAULT,
        ) {
            Ok(_) => {
                self.processed_mat = filtered;
                self.clone()
            }
            Err(err) => {
                error!("error applying bilateral filter\t{}", err.message);
                self.clone()
            }
        }
    }

    pub async fn dilate(mut self, size: Size, iterations: i32, point: Point) -> Frame {
        dilate_image(
            &self.processed_mat.input_array().unwrap(),
            &mut self.processed_mat.output_array().unwrap(),
            &get_structuring_element(MORPH_DILATE, size, Point { x: -1, y: -1 }).unwrap(),
            point,
            iterations,
            BORDER_DEFAULT,
            Scalar::new(0.0, 0.0, 0.0, 0.0),
        )
        .unwrap();

        self
    }

    pub async fn canny(mut self) -> Frame {
        let mut canny_mat = UMat::new(UMatUsageFlags::USAGE_DEFAULT);
        canny(&self.processed_mat, &mut canny_mat, 100.0, 100.0, 3, false).unwrap();

        self.processed_mat = canny_mat;

        self
    }

    pub async fn add_weighted(mut self, overlay: UMat, alpha: f64, beta: f64, gamma: f64) -> Frame {
        add_weighted(
            &self.processed_mat.input_array().unwrap(),
            alpha,
            &overlay.input_array().unwrap(),
            beta,
            gamma,
            &mut self.processed_mat.output_array().unwrap(),
            0,
        )
        .unwrap();

        self
    }

    pub async fn inpaint(mut self, mask: UMat) -> Frame {
        inpaint(
            &self.processed_mat.input_array().unwrap(),
            &mask,
            &mut self.processed_mat.output_array().unwrap(),
            3.0,
            INPAINT_TELEA,
        )
        .unwrap();

        self
    }

    pub fn extract_roi(&self, region: RegionOfInterest) -> UMat {
        let mut cropped = self
            .processed_mat
            .clone()
            .col_range(&Range::new(region.x, region.x + region.width).unwrap())
            .unwrap();

        cropped = cropped
            .row_range(&Range::new(region.y, region.y + region.height).unwrap())
            .unwrap();

        cropped
    }

    pub async fn adjust_contrast(&mut self, amount: f64) -> Frame {
        let base_mat = self.processed_mat.clone();
        base_mat
            .convert_to(
                &mut self.processed_mat.output_array().unwrap(),
                -1,
                amount,
                amount * -2.,
            )
            .unwrap();

        self.to_owned()
    }

    pub async fn adjust_brightness(mut self, amount: f64) -> Frame {
        let base_mat = self.processed_mat.clone();
        base_mat
            .convert_to(
                &mut self.processed_mat.output_array().unwrap(),
                -1,
                1.,
                amount,
            )
            .unwrap();

        self
    }

    pub async fn threshold(mut self) -> Frame {
        threshold(
            &self.processed_mat.input_array().unwrap(),
            &mut self.processed_mat.output_array().unwrap(),
            120.0,
            255.0,
            THRESH_BINARY,
        )
        .unwrap();

        self
    }

    pub async fn bitwise_not(mut self, mask: Option<UMat>) -> Frame {
        if let Some(mask) = mask {
            bitwise_not(
                &self.processed_mat.input_array().unwrap(),
                &mut self.processed_mat.output_array().unwrap(),
                &mask,
            )
            .unwrap();
        } else {
            let empty = Mat::zeros(
                self.processed_mat.rows(),
                self.processed_mat.cols(),
                self.processed_mat.typ(),
            )
            .unwrap();

            bitwise_not(
                &self.processed_mat.input_array().unwrap(),
                &mut self.processed_mat.output_array().unwrap(),
                &empty,
            )
            .unwrap();
        }

        self
    }

    pub async fn bitwise_and(mut self, mask: UMat) -> Frame {
        let empty = Mat::zeros(
            self.processed_mat.rows(),
            self.processed_mat.cols(),
            self.processed_mat.typ(),
        )
        .unwrap();

        bitwise_and(
            &self.processed_mat.input_array().unwrap(),
            &empty.input_array().unwrap(),
            &mut self.processed_mat.output_array().unwrap(),
            &mask,
        )
        .unwrap();

        self
    }

    pub async fn list_text_recognition<'f>(
        &mut self,
        region_list: RegionOfInterestList,
        shared_recognizer: &Arc<Mutex<dnn::TextRecognitionModel>>,
    ) {
        let array: Vector<Rect_<i32>> = region_list.vec_of_rects();
        let mut results: Vector<String> = Vector::new();

        let recognizer = shared_recognizer.lock().await;
        recognizer
            .recognize_1(&self.processed_mat, &array, &mut results)
            .unwrap();

        for (i, region) in region_list.iter().enumerate() {
            let (_, mut new_region) = region.clone();
            new_region.set_result(results.get(i).unwrap());
            self.results.add_region(new_region);
        }
    }

    pub async fn text_recognition(
        mut self,
        mut region: RegionOfInterest,
        shared_recognizer: &Arc<Mutex<dnn::TextRecognitionModel>>,
    ) -> Frame {
        let mat = self.extract_roi(region.clone());

        let recognizer = shared_recognizer.lock().await;
        let recognition_result = recognizer.recognize(&mat).unwrap();

        region.set_result(recognition_result);

        self.results.add_region(region);

        self
    }

    pub async fn add_result(mut self, region: RegionOfInterest) {
        self.results.add_region(region);
    }

    pub fn highlight_regions(&mut self) {
        for (_, region) in self.results.iter() {
            let rect = Rect_::new(region.x, region.y, region.width, region.height);

            rectangle(
                &mut self.processed_mat.input_output_array().unwrap(),
                rect,
                Scalar::new(0., 255., 0.0, 1.0),
                1,
                0,
                0,
            )
            .unwrap();

            if region.result.is_some() {
                put_text(
                    &mut self.processed_mat.input_output_array().unwrap(),
                    region.result.as_ref().unwrap(),
                    opencv::core::Point_ {
                        x: region.x + region.width + 5,
                        y: region.y + region.height,
                    },
                    FONT_HERSHEY_SIMPLEX,
                    0.75,
                    Scalar::new(0., 255., 0.0, 1.0),
                    1,
                    0,
                    false,
                )
                .unwrap();
            }
        }
    }
}
