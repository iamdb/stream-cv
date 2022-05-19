use libvips::{ops, VipsImage};
use opencv::{
    core::{add_weighted, Mat, Point, Scalar, Size, ToOutputArray, BORDER_DEFAULT},
    imgproc::{
        bilateral_filter, canny, cvt_color, dilate as dilate_image, get_structuring_element,
        COLOR_BGR2GRAY, COLOR_GRAY2BGR, MORPH_DILATE,
    },
    photo::{detail_enhance, inpaint, INPAINT_TELEA},
    prelude::MatTraitConstManual,
};

#[derive(Clone, Debug)]
pub struct Frame {
    pub mat: Mat,
    pub processed_mat: Mat,
    pub num: i64,
    pub text: String,
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
    pub async fn run_ocr(&self) -> Frame {
        let mut mutable_frame = self.clone();
        let frame_size = mutable_frame.mat.size().unwrap();
        let frame_bytes = mutable_frame.mat.data_bytes().unwrap();
        let vips_image = VipsImage::new_from_memory(
            frame_bytes,
            frame_size.width,
            frame_size.height,
            1,
            ops::BandFormat::Uchar,
        )
        .unwrap();
        let png_image = ops::pngsave_buffer(&vips_image).unwrap();

        let mut lt = leptess::LepTess::new(None, "eng").unwrap();
        lt.set_image_from_mem(&png_image).unwrap();

        let ocr_text = lt.get_utf8_text().unwrap();

        if !ocr_text.is_empty() {
            mutable_frame.text = ocr_text;
        }

        mutable_frame
    }
    pub async fn to_gray(&self) -> Frame {
        let mut gray = Mat::default();
        let mut mutable_frame = self.clone();

        cvt_color(&mutable_frame.processed_mat, &mut gray, COLOR_BGR2GRAY, 0).unwrap();

        mutable_frame.processed_mat = gray;

        mutable_frame
    }
    pub async fn to_bgr(&self) -> Frame {
        let mut bgr = Mat::default();
        let mut mutable_frame = self.clone();

        cvt_color(&mutable_frame.processed_mat, &mut bgr, COLOR_GRAY2BGR, 0).unwrap();

        mutable_frame.processed_mat = bgr;

        mutable_frame
    }
    pub async fn detail_enhance(&self) -> Frame {
        let mut bgr = Mat::default();
        let mut mutable_frame = self.clone();

        detail_enhance(&mutable_frame.processed_mat, &mut bgr, 12.0, 5.0).unwrap();

        mutable_frame.processed_mat = bgr;

        mutable_frame
    }
    pub async fn bilateral_filter(&self) -> Frame {
        let mut filtered_image = Mat::default();
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
    pub async fn dilate(&self) -> Frame {
        let mut dilated = Mat::default();
        let output_array = &mut dilated.output_array().unwrap();
        let mut mutable_frame = self.clone();

        dilate_image(
            &self.processed_mat,
            output_array,
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

        mutable_frame.processed_mat = dilated;

        mutable_frame
    }
    pub async fn canny(&self) -> Frame {
        let mut canny_mat = Mat::default();
        let mut mutable_frame = self.clone();
        canny(&self.processed_mat, &mut canny_mat, 100.0, 100.0, 3, false).unwrap();

        mutable_frame.processed_mat = canny_mat;

        mutable_frame
    }
    pub async fn add_weighted(&self, overlay: Mat, alpha: f64, beta: f64) -> Frame {
        let mut weighted = Mat::default();
        let mut mutable_frame = self.clone();

        add_weighted(
            &self.processed_mat,
            alpha,
            &overlay,
            beta,
            2.0,
            &mut weighted,
            0,
        )
        .unwrap();

        mutable_frame.processed_mat = weighted;

        mutable_frame
    }

    pub fn inpaint(&self, mask: Mat) -> Frame {
        let mut painted = Mat::default();
        let mut mutable_frame = self.clone();

        inpaint(
            &mutable_frame.processed_mat,
            &mask,
            &mut painted,
            3.0,
            INPAINT_TELEA,
        )
        .unwrap();

        mutable_frame.processed_mat = painted;

        mutable_frame
    }
}
