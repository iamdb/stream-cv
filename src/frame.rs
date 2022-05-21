use chrono::{DateTime, Utc};
use libvips::{ops, VipsImage};
use opencv::{
    core::{
        add_weighted, Mat, Point, Scalar, Size, ToInputArray, ToOutputArray, UMat, BORDER_DEFAULT,
    },
    imgproc::{
        bilateral_filter, canny, cvt_color, dilate as dilate_image, get_structuring_element,
        COLOR_BGR2GRAY, COLOR_GRAY2BGR, MORPH_DILATE,
    },
    photo::{detail_enhance, inpaint, INPAINT_TELEA},
    prelude::*,
};

#[derive(Clone)]
pub struct Frame {
    pub mat: Mat,
    pub processed_mat: UMat,
    pub num: i64,
    pub text: String,
    pub start_date: DateTime<Utc>,
    pub end_date: Option<DateTime<Utc>>,
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

#[allow(dead_code)]
impl Frame {
    fn make_vips(&self) -> VipsImage {
        let frame_size = self.mat.size().unwrap();
        let frame_bytes = self.mat.data_bytes().unwrap();

        VipsImage::new_from_memory(
            frame_bytes,
            frame_size.width,
            frame_size.height,
            1,
            ops::BandFormat::Uchar,
        )
        .unwrap()
    }
    pub async fn run_ocr(&self) -> Frame {
        let mut mutable_frame = self.clone();
        let vips_image = self.make_vips();
        let png_image = ops::pngsave_buffer(&vips_image).unwrap();

        let mut lt = leptess::LepTess::new(None, "eng").unwrap();
        lt.set_image_from_mem(&png_image).unwrap();

        let ocr_text = lt.get_utf8_text().unwrap();

        if !ocr_text.is_empty() {
            mutable_frame.text = ocr_text;
        }

        mutable_frame
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
    pub async fn detail_enhance(mut self) -> Frame {
        detail_enhance(
            &self.processed_mat.input_array().unwrap(),
            &mut self.processed_mat.output_array().unwrap(),
            5.0,
            5.0,
        )
        .unwrap();

        self
    }
    pub async fn bilateral_filter(&self) -> Frame {
        let mut filtered_image = UMat::new(opencv::core::UMatUsageFlags::USAGE_DEFAULT);
        let mut mutable_frame = self.clone();

        bilateral_filter(
            &mutable_frame.processed_mat,
            &mut filtered_image,
            3,
            120.0,
            120.0,
            BORDER_DEFAULT,
        )
        .unwrap();

        mutable_frame.processed_mat = filtered_image;

        mutable_frame
    }
    pub async fn dilate(mut self) -> Frame {
        dilate_image(
            &self.processed_mat.input_array().unwrap(),
            &mut self.processed_mat.output_array().unwrap(),
            &get_structuring_element(
                MORPH_DILATE,
                Size {
                    width: 2,
                    height: 2,
                },
                Point { x: -1, y: -1 },
            )
            .unwrap(),
            Point { x: -1, y: -1 },
            1,
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
    pub async fn add_weighted(mut self, overlay: UMat, alpha: f64, beta: f64) -> Frame {
        add_weighted(
            &self.processed_mat.input_array().unwrap(),
            alpha,
            &overlay.input_array().unwrap(),
            beta,
            2.0,
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
}
