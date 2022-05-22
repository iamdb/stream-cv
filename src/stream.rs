extern crate ffmpeg_next as ffmpeg;

use crate::frame::Frame;
use crate::pipeline::Pipeline;
use chrono::Utc;
use ffmpeg::format::{input, Pixel};
use ffmpeg::frame::Video;
use ffmpeg::software::scaling::{context::Context as FFContext, flag::Flags};
use ffmpeg::sys::{av_log_set_level, AV_LOG_QUIET};
use futures::Stream;
use opencv::core::{Mat, UMat, CV_8UC3};
use opencv::imgproc::{cvt_color, COLOR_RGB2BGR};
use std::collections::VecDeque;
use std::ffi::c_void;

pub struct VideoStream<'p> {
    _width: i32,
    _height: i32,
    _framerate: i32,
    pub decoding: bool,
    url: String,
    pub frame_index: i64,
    scaler: Option<FFContext>,
    pipe: &'p Pipeline,
    queue: VecDeque<Frame>,
}

impl<'p> Stream for VideoStream<'p> {
    type Item = Frame;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        _: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        if self.queue.is_empty() {
            std::task::Poll::Pending
        } else {
            std::task::Poll::Ready(self.queue.pop_front())
        }
    }
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
            queue: VecDeque::new(),
        }
    }
    pub async fn decode(&mut self) {
        ffmpeg::init().unwrap();

        unsafe { av_log_set_level(AV_LOG_QUIET) }

        info!("****setting up decoder*****");
        if let Ok(mut ictx) = input(&self.url) {
            info!("*****building input*****");
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
            info!("Stream spec: {}", input.metadata().get("comment").unwrap());

            let mut context_decoder =
                ffmpeg::codec::context::Context::from_parameters(input.parameters()).unwrap();

            context_decoder.set_threading(ffmpeg::threading::Config {
                count: 1,
                kind: ffmpeg::threading::Type::Frame,
                safe: true,
            });

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

            info!("decoding stream");
            self.decoding = true;

            for (stream, packet) in ictx.packets() {
                if stream.index() == video_stream_index {
                    match decoder.send_packet(&packet) {
                        Ok(_) => self.receive_and_process_decoded_frames(&mut decoder).await,
                        Err(error) => error!("{}", error.to_string()),
                    }
                }
            }

            decoder.send_eof().unwrap();
            self.receive_and_process_decoded_frames(&mut decoder).await;

            self.decoding = false;
        }
    }

    async fn receive_and_process_decoded_frames(&mut self, decoder: &mut ffmpeg::decoder::Video) {
        let mut decoded = Video::empty();
        while decoder.receive_frame(&mut decoded).is_ok() {
            let mut rgb_frame = Video::empty();

            self.scaler
                .as_mut()
                .unwrap()
                .run(&decoded, &mut rgb_frame)
                .unwrap();

            let rgb_mat: Mat;
            let mut bgr_umat = UMat::new(opencv::core::UMatUsageFlags::USAGE_DEFAULT);

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

            cvt_color(&rgb_mat, &mut bgr_umat, COLOR_RGB2BGR, 0).unwrap();

            let new_frame = crate::Frame {
                mat: rgb_mat.clone(),
                num: self.frame_index,
                processed_mat: bgr_umat,
                text: "".to_string(),
                start_date: Utc::now(),
                end_date: None,
            };

            self.pipe.decode_send(new_frame).await;

            self.frame_index += 1;
        }
    }
}
