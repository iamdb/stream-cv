extern crate ffmpeg_next as ffmpeg;

use crate::pipeline::Pipeline;
use ffmpeg::format::{input, Pixel};
use ffmpeg::frame::Video;
use ffmpeg::software::scaling::{context::Context as FFContext, flag::Flags};
use ffmpeg::sys::{av_log_set_level, AV_LOG_WARNING};
use opencv::core::{Mat, CV_8UC3};
use opencv::imgproc::{cvt_color, COLOR_RGB2BGR};
use std::ffi::c_void;

pub struct VideoStream<'p> {
    _width: i32,
    _height: i32,
    _framerate: i32,
    pub decoding: bool,
    url: String,
    pub frame_index: i32,
    scaler: Option<FFContext>,
    pipe: &'p Pipeline,
}

impl<'p> VideoStream<'p> {
    pub fn new(url: String, pipe: &'p Pipeline) -> Self {
        VideoStream {
            _width: 0,
            _height: 0,
            _framerate: 0,
            decoding: false,
            scaler: None,
            url,
            frame_index: 0,
            pipe,
        }
    }
    pub async fn decode(&mut self) {
        ffmpeg::init().unwrap();

        unsafe { av_log_set_level(AV_LOG_WARNING) }

        println!("****starting decoder*****");
        self.decoding = true;
        if let Ok(mut ictx) = input(&self.url) {
            println!("*****building input*****");
            let streams = ictx.streams();

            let inputs = streams
                .filter(|s| {
                    let meta = s.metadata();
                    let stream_meta = meta.into_iter().map(|a| a.1);

                    stream_meta.count() > 1
                })
                .collect::<Vec<ffmpeg::Stream>>();

            let input = inputs.first().unwrap();

            let video_stream_index = input.index();
            println!("{:?}", input.metadata());

            let context_decoder =
                ffmpeg::codec::context::Context::from_parameters(input.parameters()).unwrap();
            let mut decoder = context_decoder.decoder().video().unwrap();

            self.scaler = Some(
                FFContext::get(
                    decoder.format(),
                    decoder.width(),
                    decoder.height(),
                    Pixel::RGB24,
                    decoder.width(),
                    decoder.height(),
                    Flags::LANCZOS,
                )
                .unwrap(),
            );

            for (stream, packet) in ictx.packets() {
                if stream.index() == video_stream_index {
                    decoder.send_packet(&packet).unwrap();
                    self.receive_and_process_decoded_frames(&mut decoder)
                        .await
                        .unwrap();
                }
            }

            decoder.send_eof().unwrap();
            self.receive_and_process_decoded_frames(&mut decoder)
                .await
                .unwrap();

            self.decoding = false;
        }
    }
    async fn receive_and_process_decoded_frames(
        &mut self,
        decoder: &mut ffmpeg::decoder::Video,
    ) -> Result<(), ffmpeg::Error> {
        let mut decoded = Video::empty();
        while decoder.receive_frame(&mut decoded).is_ok() {
            let mut rgb_frame = Video::empty();
            self.scaler
                .as_mut()
                .unwrap()
                .run(&decoded, &mut rgb_frame)?;
            let rgb_mat: Mat;
            let mut bgr_mat = Mat::default();
            unsafe {
                let raw_rgb_frame = rgb_frame.as_ptr().as_ref().unwrap();

                rgb_mat = Mat::new_rows_cols_with_data(
                    raw_rgb_frame.height,
                    raw_rgb_frame.width,
                    CV_8UC3,
                    raw_rgb_frame.data[0] as *mut c_void,
                    raw_rgb_frame.linesize[0].try_into().unwrap(),
                )
                .unwrap();
            }
            cvt_color(&rgb_mat, &mut bgr_mat, COLOR_RGB2BGR, 0).unwrap();
            let new_frame = crate::Frame {
                mat: bgr_mat.clone(),
                num: self.frame_index,
                processed_mat: bgr_mat.clone(),
            };

            println!("sending frame {} to processor", new_frame.num);

            self.pipe.send(new_frame).await;

            self.frame_index += 1;
        }
        Ok(())
    }
}