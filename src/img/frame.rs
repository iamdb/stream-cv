use std::{fs::File, io::BufRead, io::BufReader, path::PathBuf};

use chrono::{DateTime, Utc};
use libvips::{ops, VipsImage};
use opencv::{
    core::{
        add_weighted, bitwise_and, bitwise_not, Mat, Point as OpenCVPoint, Point2f, Rect2d, Scalar,
        Size as OpenCVSize, ToInputArray, ToInputOutputArray, ToOutputArray, UMat, VecN,
        BORDER_CONSTANT, BORDER_DEFAULT, DECOMP_LU,
    },
    dnn::{self, TextDetectionModel_EAST},
    imgproc::{
        self, bilateral_filter, canny, cvt_color, dilate as dilate_image, get_structuring_element,
        rectangle_points, threshold, COLOR_BGR2GRAY, COLOR_GRAY2BGR, COLOR_GRAY2RGB, MORPH_DILATE,
        THRESH_BINARY,
    },
    photo::{detail_enhance, inpaint, INPAINT_TELEA},
    prelude::*,
    types::{VectorOfPoint2f, VectorOfString, VectorOfVectorOfPoint},
};

use crate::roi::RegionOfInterest;

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
    pub result_text: Vec<String>,
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

fn make_vips(mat: Mat) -> VipsImage {
    let frame_size = mat.size().unwrap();
    let frame_bytes = mat.data_bytes().unwrap();

    VipsImage::new_from_memory(
        frame_bytes,
        frame_size.width,
        frame_size.height,
        1,
        ops::BandFormat::Uchar,
    )
    .unwrap()
}

#[allow(dead_code)]
impl Frame {
    pub async fn run_fullframe_ocr(&self) -> Frame {
        let mut mutable_frame = self.clone();
        let vips_image = make_vips(self.mat.clone());
        let png_image = ops::pngsave_buffer(&vips_image).unwrap();

        let mut lt = leptess::LepTess::new(None, "eng").unwrap();
        lt.set_image_from_mem(&png_image).unwrap();

        let ocr_text = lt.get_utf8_text().unwrap();

        if !ocr_text.is_empty() {
            mutable_frame.text = ocr_text;
        }

        mutable_frame
    }

    pub async fn process_region_ocr(&mut self, region: &RegionOfInterest) -> String {
        let roi_region = self.extract_roi(region);
        let vips_image = make_vips(roi_region);

        let png_image = ops::pngsave_buffer(&vips_image).unwrap();

        let mut lt = leptess::LepTess::new(None, "eng").unwrap();
        lt.set_image_from_mem(&png_image).unwrap();

        lt.get_utf8_text().unwrap()
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

    pub async fn detail_enhance(mut self, sigma_s: f32, sigma_r: f32) -> Frame {
        detail_enhance(
            &self.processed_mat.input_array().unwrap(),
            &mut self.processed_mat.output_array().unwrap(),
            sigma_s,
            sigma_r,
        )
        .unwrap();

        self
    }

    pub async fn bilateral_filter(mut self, d: i32, sigma_color: f64, sigma_space: f64) -> Frame {
        let mut filtered_image = UMat::new(opencv::core::UMatUsageFlags::USAGE_DEFAULT);

        bilateral_filter(
            &self.processed_mat,
            &mut filtered_image,
            d,
            sigma_color,
            sigma_space,
            BORDER_DEFAULT,
        )
        .unwrap();

        self.processed_mat = filtered_image;

        self
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
        let mut canny_mat = UMat::new(opencv::core::UMatUsageFlags::USAGE_DEFAULT);
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

    pub fn extract_roi(&mut self, region: &RegionOfInterest) -> Mat {
        // let rect = Rect2d::new(
        //     region.x as f64,
        //     region.y as f64,
        //     region.width as f64,
        //     region.height as f64,
        // );
        //
        // let pt1 = VecN::new(region.x, region.y, region.width, region.height);
        //
        // rectangle_points(
        //     &mut self.processed_mat.input_output_array().unwrap(),
        //     Point::from_vec2(pt1),
        //     pt2,
        //     color,
        //     thickness,
        //     line_type,
        //     shift,
        // );

        self.mat
            .adjust_roi(
                region.y,
                region.y + region.height,
                region.x,
                region.x + region.width,
            )
            .unwrap()
    }

    pub async fn adjust_contrast(mut self, amount: f64) -> Frame {
        let base_mat = self.processed_mat.clone();
        base_mat
            .convert_to(
                &mut self.processed_mat.output_array().unwrap(),
                -1,
                amount,
                amount * -2.,
            )
            .unwrap();

        self.clone()
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

    pub async fn text_detection(&self, region: &RegionOfInterest) {
        let (detector, recognizer) = make_text_detector();
        text_detection(self.clone(), detector, recognizer, region).await;
    }

    // pub async fn adjust_gamma(mut self) -> Frame {
    //     let mut hsv = Mat::default();
    //     cvt_color(&self.processed_mat, &mut hsv, COLOR_BGR2HSV, 0).unwrap();
    //
    //     let mid = 0.5;
    //     let mean: i32 = 32;
    //
    //     self
    // }
}

pub fn make_text_detector() -> (TextDetectionModel_EAST, dnn::TextRecognitionModel) {
    let conf_threshold = 0.5;
    let nms_threshold = 0.4;
    let width = 640;
    let height = 640;

    let det_model_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("frozen_east_text_detection.pb");
    let rec_model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("crnn.onnx");
    let voc_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("alphabet_36.txt");

    // Load networks.
    let mut detector =
        dnn::TextDetectionModel_EAST::from_file(det_model_path.to_str().unwrap(), "").unwrap();
    detector
        .set_confidence_threshold(conf_threshold)
        .unwrap()
        .set_nms_threshold(nms_threshold)
        .unwrap();
    let mut recognizer =
        dnn::TextRecognitionModel::from_file(rec_model_path.to_str().unwrap(), "").unwrap();

    // Load vocabulary
    let mut vocabulary = VectorOfString::new();
    let voc_file = BufReader::new(File::open(voc_path).unwrap());
    for voc_line in voc_file.lines() {
        vocabulary.push(&voc_line.unwrap());
    }
    recognizer
        .set_vocabulary(&vocabulary)
        .unwrap()
        .set_decode_type("CTC-greedy")
        .unwrap();

    // Parameters for Recognition
    let rec_scale = 1. / 127.5;
    let rec_mean = Scalar::from((127.5, 127.5, 127.5));
    let rec_input_size = Size::new(100, 32);
    recognizer
        .set_input_params(rec_scale, rec_input_size, rec_mean, false, false)
        .unwrap();

    // Parameters for Detection
    let det_scale = 1.;
    let det_input_size = Size::new(width, height);
    let det_mean = Scalar::from((123.68, 116.78, 103.94));
    let swap_rb = true;
    detector
        .set_input_params(det_scale, det_input_size, det_mean, swap_rb, false)
        .unwrap();

    (detector, recognizer)
}

async fn text_detection(
    mut frame: Frame,
    detector: TextDetectionModel_EAST,
    recognizer: dnn::TextRecognitionModel,
    region: &RegionOfInterest,
) {
    let mat = frame.extract_roi(region);
    // Detection
    let mut det_results = VectorOfVectorOfPoint::new();
    detector.detect(&mat, &mut det_results).unwrap();

    if !det_results.is_empty() {
        // Text Recognition
        let mut rec_input = Mat::default();
        imgproc::cvt_color(&mat, &mut rec_input, imgproc::COLOR_BGR2GRAY, 0).unwrap();

        for quadrangle in &det_results {
            let mut quadrangle_2f = VectorOfPoint2f::new();
            for pt in &quadrangle {
                quadrangle_2f.push(Point2f::new(pt.x as f32, pt.y as f32))
            }
            let cropped = four_points_transform(&rec_input, quadrangle_2f.as_slice());
            let recognition_result = recognizer.recognize(&cropped).unwrap();

            debug!("Recognition Result: {}", recognition_result);

            frame.result_text.push(recognition_result);
        }
    }
}

fn four_points_transform(frame: &Mat, vertices: &[Point2f]) -> Mat {
    let output_size = Size::new(100, 32);
    let target_vertices = [
        Point2f::new(0., (output_size.height - 1) as f32),
        Point2f::new(0., 0.),
        Point2f::new((output_size.width - 1) as f32, 0.),
        Point2f::new(
            (output_size.width - 1) as f32,
            (output_size.height - 1) as f32,
        ),
    ];
    let rotation_matrix =
        imgproc::get_perspective_transform_slice(vertices, &target_vertices, DECOMP_LU).unwrap();
    let mut out = Mat::default();
    imgproc::warp_perspective(
        frame,
        &mut out,
        &rotation_matrix,
        output_size,
        imgproc::INTER_LINEAR,
        BORDER_CONSTANT,
        Scalar::default(),
    )
    .unwrap();

    out
}
