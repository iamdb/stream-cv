use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

use opencv::core::{Scalar, Size};
use opencv::dnn;
use opencv::prelude::*;
use opencv::types::VectorOfString;

pub mod frame;

pub fn make_text_recognizer() -> dnn::TextRecognitionModel {
    debug!("loading models for text recognition");
    let rec_model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models")
        .join("crnn.onnx");
    let voc_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models")
        .join("alphabet_36.txt");

    let mut recognizer =
        dnn::TextRecognitionModel::from_file(rec_model_path.to_str().unwrap(), "").unwrap();

    recognizer
        .set_preferable_target(dnn::Target::DNN_TARGET_CUDA)
        .unwrap()
        .set_preferable_backend(dnn::Backend::DNN_BACKEND_CUDA)
        .unwrap();

    // Load vocabulary
    let mut vocabulary = VectorOfString::new();
    let voc_file = BufReader::new(File::open(voc_path).unwrap());
    for voc_line in voc_file.lines() {
        vocabulary.push(&voc_line.unwrap());
    }

    // Parameters for Recognition
    let rec_scale = 1. / 127.5;
    let rec_mean = Scalar::from((127.5, 127.5, 127.5));
    let rec_input_size = Size::new(100, 32);
    recognizer
        .set_vocabulary(&vocabulary)
        .unwrap()
        .set_decode_type("CTC-prefix-beam-search")
        .unwrap()
        .set_input_params(rec_scale, rec_input_size, rec_mean, false, false)
        .unwrap();

    recognizer
}
